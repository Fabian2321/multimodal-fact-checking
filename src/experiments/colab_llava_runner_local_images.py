#!/usr/bin/env python3
"""
Google Colab LLaVA Runner for Text-Image Matching (Local Images)

This script runs the LLaVA-1.5-7B experiment on Google Colab with GPU acceleration.
It uses local images from the colab_images/ folder instead of downloading them.

Usage in Colab:
1. Upload this script to Colab
2. Upload the test_balanced_pairs_clean.csv file
3. Upload and extract colab_images.zip to create colab_images/ folder
4. Run the script
"""

import os
import pandas as pd
import torch
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from PIL import Image
import time
from typing import List, Dict, Any
import logging
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class ColabLLaVARunner:
    """LLaVA experiment runner for Google Colab with local images"""
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf"):
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Prompt: 'Yes' only for clear, specific matches
        self.prompt_template = """USER: <image>\nText: '{text}'\nMetadata: {metadata}\nDoes the text accurately describe the image and metadata? Answer 'Yes' only if the text clearly and specifically matches the image and metadata. If you are unsure or the match is only partial or vague, answer 'No'. Start your answer with 'Yes' or 'No' and provide a short explanation.\nASSISTANT:"""
        
        logger.info(f"Initializing LLaVA runner with device: {self.device}")
    
    def load_model(self):
        """Load LLaVA model and processor"""
        logger.info(f"Loading LLaVA model: {self.model_name}")
        
        try:
            self.processor = LlavaProcessor.from_pretrained(self.model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def load_local_image(self, image_id: str) -> Image.Image:
        """Load image from local colab_images folder"""
        try:
            # Look for image with the given ID (with any extension)
            image_pattern = os.path.join("colab_images", f"{image_id}.*")
            matching_files = glob.glob(image_pattern)
            
            if matching_files:
                image_path = matching_files[0]
                image = Image.open(image_path).convert('RGB')
                logger.debug(f"Loaded image: {image_path}")
                return image
            else:
                logger.warning(f"No image found for ID {image_id}")
                return Image.new('RGB', (224, 224), color='gray')
                
        except Exception as e:
            logger.warning(f"Could not load image for ID {image_id}: {e}")
            return Image.new('RGB', (224, 224), color='gray')
    
    def create_metadata_string(self, row: pd.Series) -> str:
        """Create metadata string from row data"""
        metadata_parts = []
        
        # Add relevant metadata fields
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
        """Process a single sample"""
        try:
            # Load local image using the ID
            image = self.load_local_image(row['id'])
            
            # Create metadata string
            metadata = self.create_metadata_string(row)
            
            # Create prompt
            prompt = self.prompt_template.format(
                text=row['clean_title'],
                metadata=metadata
            )
            
            # Process with LLaVA
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False
                )
            
            # Decode response
            generated_text = self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Extract the assistant's response (after "ASSISTANT:")
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text.strip()
            
            # Parse prediction
            parser = LLaVAAnswerParser()
            predicted_label, confidence, explanation = parser.extract_prediction(response)
            
            return {
                'id': row['id'],
                'text': row['clean_title'],
                'image_url': row['image_url'],  # Keep for reference
                'true_label': row['2_way_label'],
                'generated_text': response,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'parsing_explanation': explanation,
                'metadata': metadata
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
                'metadata': self.create_metadata_string(row)
            }
    
    def run_experiment(self, csv_file: str, num_samples: int = 100) -> pd.DataFrame:
        """Run the complete experiment"""
        logger.info(f"Starting experiment with {num_samples} samples")
        
        # Check if colab_images folder exists
        if not os.path.exists("colab_images"):
            logger.error("colab_images folder not found. Please upload and extract colab_images.zip")
            return pd.DataFrame()
        
        # Load data
        df = pd.read_csv(csv_file)
        df = df.head(num_samples)  # Limit to requested number of samples
        
        # Load model
        self.load_model()
        
        # Process samples
        results = []
        start_time = time.time()
        
        for idx, row in df.iterrows():
            logger.info(f"Processing sample {idx+1}/{len(df)}: {row['id']}")
            
            result = self.process_sample(row)
            results.append(result)
            
            # Log progress
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (len(df) - idx - 1)
                logger.info(f"Progress: {idx+1}/{len(df)} samples. "
                          f"Avg time per sample: {avg_time:.1f}s. "
                          f"Estimated remaining: {remaining/60:.1f} minutes")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        valid_results = results_df[results_df['predicted_label'] != -1]
        if len(valid_results) > 0:
            accuracy = (valid_results['predicted_label'] == valid_results['true_label']).mean()
            logger.info(f"Accuracy: {accuracy:.3f} ({len(valid_results)} valid samples)")
        
        return results_df

def main():
    """Main function to run the experiment"""
    
    # Configuration
    CSV_FILE = "test_balanced_pairs_clean.csv"  # Upload this to Colab
    NUM_SAMPLES = 100
    MODEL_NAME = "llava-hf/llava-1.5-7b-hf"
    OUTPUT_FILE = "llava_results_local_images.csv"
    
    # Check if CSV file exists
    if not os.path.exists(CSV_FILE):
        logger.error(f"CSV file {CSV_FILE} not found. Please upload it to Colab.")
        return
    
    # Check if colab_images folder exists
    if not os.path.exists("colab_images"):
        logger.error("colab_images folder not found. Please upload and extract colab_images.zip")
        logger.info("Run this command in Colab: !unzip -o colab_images.zip -d colab_images")
        return
    
    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available. This will be very slow!")
    
    # Run experiment
    runner = ColabLLaVARunner(MODEL_NAME)
    results = runner.run_experiment(CSV_FILE, NUM_SAMPLES)
    
    # Save results
    results.to_csv(OUTPUT_FILE, index=False)
    logger.info(f"Results saved to {OUTPUT_FILE}")
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Total samples: {len(results)}")
    print(f"Valid predictions: {len(results[results['predicted_label'] != -1])}")
    
    valid_results = results[results['predicted_label'] != -1]
    if len(valid_results) > 0:
        accuracy = (valid_results['predicted_label'] == valid_results['true_label']).mean()
        print(f"Accuracy: {accuracy:.3f}")
        
        # Confusion matrix
        tp = ((valid_results['predicted_label'] == 1) & (valid_results['true_label'] == 1)).sum()
        tn = ((valid_results['predicted_label'] == 0) & (valid_results['true_label'] == 0)).sum()
        fp = ((valid_results['predicted_label'] == 1) & (valid_results['true_label'] == 0)).sum()
        fn = ((valid_results['predicted_label'] == 0) & (valid_results['true_label'] == 1)).sum()
        
        print(f"True Positives: {tp}")
        print(f"True Negatives: {tn}")
        print(f"False Positives: {fp}")
        print(f"False Negatives: {fn}")
    
    print("="*50)

# In Colab, call main() directly
main() 