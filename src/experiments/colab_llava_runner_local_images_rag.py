import os
import glob
import json
import pandas as pd
import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
import time
import logging
from typing import List, Dict, Any
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- RAG Knowledge Base Loader ---
def load_external_knowledge(knowledge_dir: str) -> List[str]:
    """Loads all relevant text fields from all .json files in the directory."""
    knowledge_texts = []
    for file in glob.glob(os.path.join(knowledge_dir, '*.json')):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Collect all text fields from known structures
            for key in data:
                if isinstance(data[key], list):
                    for entry in data[key]:
                        # For Guidelines
                        if 'text' in entry:
                            knowledge_texts.append(entry['text'])
                        # For Misconceptions
                        if 'misconception' in entry and 'correction' in entry:
                            knowledge_texts.append(f"MISCONCEPTION: {entry['misconception']} CORRECTION: {entry['correction']}")
    logger.info(f"Loaded {len(knowledge_texts)} knowledge entries from {knowledge_dir}")
    return knowledge_texts

# --- LLaVA Answer Parser (as before, for clarity) ---
class LLaVAAnswerParser:
    """Parser for LLaVA outputs, 'Yes' only without uncertainty words as positive"""
    def extract_prediction(self, generated_text: str) -> tuple[int, float, str]:
        text = generated_text.lower().strip()
        if text.startswith('yes'):
            if any(word in text for word in ['maybe', 'partially', 'somewhat', 'related', 'unclear']):
                return 0, 0.95, "Model gave an uncertain or partial match response"
            else:
                return 1, 0.95, "Model gave a clear positive response"
        else:
            return 0, 0.95, "Model gave a negative or unclear response"

def retrieve_relevant_knowledge(knowledge_texts: list, sample_text: str, top_k: int = 3) -> list:
    """Simple relevance search: Choose the top_k guidelines/misconceptions with the most keyword overlaps to the sample text."""
    sample_words = set(re.findall(r'\w+', sample_text.lower()))
    scored = []
    for entry in knowledge_texts:
        entry_words = set(re.findall(r'\w+', entry.lower()))
        score = len(sample_words & entry_words)
        scored.append((score, entry))
    # Sort by score, then by length (prefer shorter in case of tie)
    scored = sorted(scored, key=lambda x: (-x[0], len(x[1])))
    return [entry for score, entry in scored[:top_k] if score > 0] or scored[:top_k]

# --- ColabLLaVARunner with RAG ---
class ColabLLaVARunnerRAG:
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", knowledge_dir: str = "data/external_knowledge"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.knowledge_texts = load_external_knowledge(knowledge_dir)
        # Prompt with Additional Context
        self.prompt_template = (
            "USER: <image>\n"
            "Text: '{text}'\n"
            "Metadata: {metadata}\n"
            "Additional context: {additional_context}\n"
            "Does the text accurately describe the image and metadata? Answer 'Yes' only if the text clearly and specifically matches the image and metadata. If you are unsure or the match is only partial or vague, answer 'No'. Start your answer with 'Yes' or 'No' and provide a short explanation.\n"
            "ASSISTANT:"
        )
        logger.info(f"Initializing LLaVA RAG runner with device: {self.device}")

    def load_model(self):
        logger.info(f"Loading LLaVA model: {self.model_name}")
        self.processor = LlavaProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info("Model loaded successfully")

    def load_local_image(self, image_id: str) -> Image.Image:
        import glob
        image_pattern = os.path.join("colab_images", f"{image_id}.*")
        matching_files = glob.glob(image_pattern)
        if matching_files:
            return Image.open(matching_files[0]).convert('RGB')
        else:
            logger.warning(f"No image found for ID {image_id}")
            return Image.new('RGB', (224, 224), color='gray')

    def create_metadata_string(self, row: pd.Series) -> str:
        metadata_parts = []
        if pd.notna(row.get('created_utc')):
            metadata_parts.append(f"created_utc: {row['created_utc']}")
        if pd.notna(row.get('domain')):
            metadata_parts.append(f"domain: {row['domain']}")
        if pd.notna(row.get('author')):
            metadata_parts.append(f"author: {row['author']}")
        if pd.notna(row.get('subreddit')):
            metadata_parts.append(f"subreddit: {row['subreddit']}")
        return "; ".join(metadata_parts) if metadata_parts else "No metadata available"

    def process_sample(self, row: pd.Series) -> Dict[str, Any]:
        try:
            image = self.load_local_image(row['id'])
            metadata = self.create_metadata_string(row)
            # Select relevant guidelines/misconceptions for this sample
            relevant_knowledge = retrieve_relevant_knowledge(self.knowledge_texts, row['clean_title'], top_k=3)
            additional_context = " \n".join(relevant_knowledge)
            prompt = self.prompt_template.format(
                text=row['clean_title'],
                metadata=metadata,
                additional_context=additional_context
            )
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False
                )
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text.strip()
            parser = LLaVAAnswerParser()
            predicted_label, confidence, explanation = parser.extract_prediction(response)
            return {
                'id': row['id'],
                'text': row['clean_title'],
                'image_url': row['image_url'],
                'true_label': row['2_way_label'],
                'generated_text': response,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'parsing_explanation': explanation,
                'metadata': metadata,
                'retrieved_knowledge': additional_context
            }
        except Exception as e:
            logger.error(f"Error processing sample {row['id']}: {e}")
            return {
                'id': row['id'],
                'text': row['clean_title'],
                'image_url': row['image_url'],
                'true_label': row['2_way_label'],
                'generated_text': f"Error: {e}",
                'predicted_label': -1,
                'confidence': 0.0,
                'parsing_explanation': f"Processing error: {e}",
                'metadata': self.create_metadata_string(row),
                'retrieved_knowledge': ''
            }

    def run_experiment(self, csv_file: str, num_samples: int = 100) -> pd.DataFrame:
        logger.info(f"Starting RAG experiment with {num_samples} samples")
        if not os.path.exists("colab_images"):
            logger.error("colab_images folder not found. Please upload and extract colab_images.zip")
            return pd.DataFrame()
        df = pd.read_csv(csv_file)
        df = df.head(num_samples)
        self.load_model()
        results = []
        start_time = time.time()
        for idx, row in df.iterrows():
            logger.info(f"Processing sample {idx+1}/{len(df)}: {row['id']}")
            result = self.process_sample(row)
            results.append(result)
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (len(df) - idx - 1)
                logger.info(f"Progress: {idx+1}/{len(df)} samples. Avg time per sample: {avg_time:.1f}s. Estimated remaining: {remaining/60:.1f} minutes")
        results_df = pd.DataFrame(results)
        valid_results = results_df[results_df['predicted_label'] != -1]
        if len(valid_results) > 0:
            accuracy = (valid_results['predicted_label'] == valid_results['true_label']).mean()
            logger.info(f"Accuracy: {accuracy:.3f} ({len(valid_results)} valid samples)")
        return results_df

def main():
    CSV_FILE = "test_balanced_pairs_clean.csv"
    NUM_SAMPLES = 100
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    OUTPUT_FILE = "llava_results_local_images_rag.csv"
    if not os.path.exists(CSV_FILE):
        logger.error(f"CSV file {CSV_FILE} not found. Please upload it to Colab.")
        return
    if not os.path.exists("colab_images"):
        logger.error("colab_images folder not found. Please upload and extract colab_images.zip")
        logger.info("Run this command in Colab: !unzip -o colab_images.zip -d colab_images")
        return
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available. This will be very slow!")
    runner = ColabLLaVARunnerRAG(MODEL_NAME)
    results = runner.run_experiment(CSV_FILE, NUM_SAMPLES)
    results.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Results saved to {OUTPUT_FILE}")
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Total samples: {len(results)}")
    print(f"Valid predictions: {len(results[results['predicted_label'] != -1])}")
    valid_results = results[results['predicted_label'] != -1]
    if len(valid_results) > 0:
        accuracy = (valid_results['predicted_label'] == valid_results['true_label']).mean()
        print(f"Accuracy: {accuracy:.3f}")
        tp = ((valid_results['predicted_label'] == 1) & (valid_results['true_label'] == 1)).sum()
        tn = ((valid_results['predicted_label'] == 0) & (valid_results['true_label'] == 0)).sum()
        fp = ((valid_results['predicted_label'] == 1) & (valid_results['true_label'] == 0)).sum()
        fn = ((valid_results['predicted_label'] == 0) & (valid_results['true_label'] == 1)).sum()
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
    print("="*50)

main() 