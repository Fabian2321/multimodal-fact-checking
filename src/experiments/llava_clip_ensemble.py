# --- Colab-kompatibles LLaVA+CLIP Ensemble Skript ---
# Vor Ausführung: !pip install transformers torch pillow pandas scikit-learn
import os
import glob
import json
import pandas as pd
import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPProcessor, CLIPModel
from PIL import Image
import time
import logging
import numpy as np
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_local_image(image_id: str) -> Image.Image:
    image_pattern = os.path.join("colab_images", f"{image_id}.*")
    matching_files = glob.glob(image_pattern)
    if matching_files:
        return Image.open(matching_files[0]).convert('RGB')
    else:
        print(f"No image found for ID {image_id}")
        return Image.new('RGB', (224, 224), color='gray')

def create_metadata_string(row: pd.Series) -> str:
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

class LLaVAAnswerParser:
    def extract_prediction(self, generated_text: str) -> tuple[int, float, str]:
        text = generated_text.lower().strip()
        if text.startswith('yes'):
            if any(word in text for word in ['maybe', 'partially', 'somewhat', 'related', 'unclear']):
                return 0, 0.95, "Model gave an uncertain or partial match response"
            else:
                return 1, 0.95, "Model gave a clear positive response"
        else:
            return 0, 0.95, "Model gave a negative or unclear response"

class LLaVAHandler:
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = LlavaProcessor.from_pretrained(self.model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.prompt_template = (
            "USER: <image>\n"
            "Text: '{text}'\n"
            "Metadata: {metadata}\n"
            "Does the text accurately describe the image and metadata? Answer 'Yes' only if the text clearly and specifically matches the image and metadata. If you are unsure or the match is only partial or vague, answer 'No'. Start your answer with 'Yes' or 'No' and provide a short explanation.\n"
            "ASSISTANT:"
        )
        self.parser = LLaVAAnswerParser()

    def predict(self, text: str, metadata: str, image: Image.Image) -> Dict[str, Any]:
        prompt = self.prompt_template.format(text=text, metadata=metadata)
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
        predicted_label, confidence, explanation = self.parser.extract_prediction(response)
        return {
            'generated_text': response,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'parsing_explanation': explanation
        }

class CLIPHandler:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)

    def predict_similarity(self, text: str, image: Image.Image) -> float:
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
            similarity = (image_embeds @ text_embeds.T).cpu().item()
        return similarity

    def find_optimal_threshold(self, similarities: list, true_labels: list) -> float:
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(true_labels, similarities)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx]

# --- Hauptfunktion für Colab ---
def main():
    CSV_FILE = "test_balanced_pairs_clean.csv"
    NUM_SAMPLES = 100
    OUTPUT_FILE = "llava_clip_ensemble_results.csv"
    if not os.path.exists(CSV_FILE):
        print(f"CSV file {CSV_FILE} not found. Bitte hochladen!")
        return
    if not os.path.exists("colab_images"):
        print("colab_images folder not found. Bitte colab_images.zip entpacken!")
        print("Run this command in Colab: !unzip -o colab_images.zip -d colab_images")
        return
    df = pd.read_csv(CSV_FILE).head(NUM_SAMPLES)
    llava = LLaVAHandler()
    clip = CLIPHandler()
    results = []
    similarities = []
    true_labels = []
    print("Running LLaVA and CLIP predictions...")
    for idx, row in df.iterrows():
        image = load_local_image(row['id'])
        metadata = create_metadata_string(row)
        # LLaVA
        llava_result = llava.predict(row['clean_title'], metadata, image)
        # CLIP
        sim = clip.predict_similarity(row['clean_title'], image)
        similarities.append(sim)
        true_labels.append(row['2_way_label'])
        results.append({
            'id': row['id'],
            'text': row['clean_title'],
            'image_url': row['image_url'],
            'true_label': row['2_way_label'],
            'llava_predicted_label': llava_result['predicted_label'],
            'llava_generated_text': llava_result['generated_text'],
            'llava_confidence': llava_result['confidence'],
            'llava_parsing_explanation': llava_result['parsing_explanation'],
            'clip_similarity': sim,
        })
    # Schwellenwert für CLIP bestimmen
    threshold = clip.find_optimal_threshold(similarities, true_labels)
    print(f"Optimaler CLIP-Schwellenwert: {threshold:.3f}")
    # CLIP-Predictions setzen
    for r in results:
        r['clip_predicted_label'] = int(r['clip_similarity'] >= threshold)
    # Ensemble-Logik
    for r in results:
        r['ensemble_and'] = int(r['llava_predicted_label'] == 1 and r['clip_predicted_label'] == 1)
        r['ensemble_or'] = int(r['llava_predicted_label'] == 1 or r['clip_predicted_label'] == 1)
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Results saved to {OUTPUT_FILE}")
    for key in ['llava_predicted_label', 'clip_predicted_label', 'ensemble_and', 'ensemble_or']:
        valid = results_df[key] != -1
        acc = (results_df.loc[valid, key] == results_df.loc[valid, 'true_label']).mean()
        print(f"{key}: Accuracy = {acc:.3f}")
    print("\nEnsemble-Experiment abgeschlossen. Siehe Output-Datei für Details.")

# In Colab einfach main() aufrufen
main() 