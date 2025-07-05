# --- DEBUG BLIP2 RESPONSES ---
# Analyzes what BLIP2 actually responds

import os
import glob
import pandas as pd
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import re
from typing import List

def load_local_image(image_id: str) -> Image.Image:
    """Loads local images from colab_images/ folder"""
    image_pattern = os.path.join("colab_images", f"{image_id}.*")
    matching_files = glob.glob(image_pattern)
    if matching_files:
        return Image.open(matching_files[0]).convert('RGB')
    else:
        print(f"No image found for ID {image_id}")
        return Image.new('RGB', (224, 224), color='gray')

def create_debug_prompts(text: str) -> List[str]:
    """Debug prompts to analyze BLIP2 responses"""
    prompts = []
    
    # Basic cleaning
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\-\.\,\!\?]', '', text)
    
    # Test different prompt types
    prompts.append(f"Question: Does this image match the text '{text}'? Answer yes or no. Answer:")
    prompts.append(f"Question: Is this image related to '{text}'? Answer yes or no. Answer:")
    prompts.append(f"Question: Is the text '{text}' true based on this image? Answer yes or no. Answer:")
    prompts.append(f"Question: Can you verify if '{text}' is correct by looking at this image? Answer:")
    prompts.append(f"Question: Is this image fake news or real news? Caption: {text} Answer:")
    
    return prompts

class DebugBLIP2Handler:
    def __init__(self):
        """Debug BLIP2 Handler"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load BLIP2 model
        print("Loading BLIP2 model: Salesforce/blip2-opt-2.7b")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        print("BLIP2 model loaded successfully!")

    def debug_responses(self, text: str, image: Image.Image, num_samples: int = 5):
        """Debug BLIP2 responses for different prompts"""
        
        prompts = create_debug_prompts(text)
        
        print(f"\n🔍 DEBUGGING BLIP2 RESPONSES")
        print(f"Text: {text}")
        print(f"Image ID: {image}")
        print("="*60)
        
        for i, prompt in enumerate(prompts):
            print(f"\n📝 Prompt {i+1}: {prompt}")
            
            try:
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=30,
                        num_beams=3,
                        do_sample=False,
                        temperature=1.0,
                        repetition_penalty=1.0
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                print(f"🤖 BLIP2 Response: '{response}'")
                
                # Response analysis
                response_lower = response.lower()
                print(f"📊 Response Analysis:")
                print(f"  - Contains 'yes': {'yes' in response_lower}")
                print(f"  - Contains 'no': {'no' in response_lower}")
                print(f"  - Length: {len(response.split())} words")
                print(f"  - Response type: {self.analyze_response_type(response)}")
                
            except Exception as e:
                print(f"❌ Error: {e}")

    def analyze_response_type(self, response: str) -> str:
        """Analyzes the type of BLIP2 response"""
        response_lower = response.lower()
        
        if 'yes' in response_lower and 'no' not in response_lower:
            return "CLEAR_YES"
        elif 'no' in response_lower and 'yes' not in response_lower:
            return "CLEAR_NO"
        elif 'yes' in response_lower and 'no' in response_lower:
            return "CONFLICTING"
        elif any(word in response_lower for word in ['fake', 'false', 'misleading']):
            return "NEGATIVE_INDICATORS"
        elif any(word in response_lower for word in ['real', 'true', 'accurate']):
            return "POSITIVE_INDICATORS"
        elif len(response.split()) < 5:
            return "SHORT_RESPONSE"
        else:
            return "DESCRIPTIVE_RESPONSE"

def main():
    """Debug BLIP2 responses"""
    
    # Parameters
    CSV_FILE = "test_balanced_pairs_clean.csv"
    NUM_SAMPLES = 5  # Only 5 samples for debug
    
    print("Debug BLIP2 Responses")
    print("="*30)
    
    # File checks
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV file {CSV_FILE} not found!")
        return
    
    if not os.path.exists("colab_images"):
        print("❌ colab_images folder not found!")
        return
    
    # Load data
    print(f"📊 Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE).head(NUM_SAMPLES)
    print(f"✅ Loaded {len(df)} samples for debugging")
    
    # Initialize BLIP2
    blip2 = DebugBLIP2Handler()
    
    # Debug for each sample
    for idx, row in df.iterrows():
        print(f"\n{'='*80}")
        print(f"🔍 DEBUGGING SAMPLE {idx+1}/{len(df)}")
        print(f"ID: {row['id']}")
        print(f"Text: {row['clean_title']}")
        print(f"True Label: {row['2_way_label']}")
        print(f"{'='*80}")
        
        image = load_local_image(row['id'])
        blip2.debug_responses(row['clean_title'], image)
        
        if idx >= 4:  # Only first 5 samples
            break
    
    print(f"\n✅ Debug completed!")
    print(f"📋 Check the responses above to understand BLIP2's behavior")

if __name__ == "__main__":
    main() 