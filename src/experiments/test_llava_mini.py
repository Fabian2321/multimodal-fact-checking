#!/usr/bin/env python3
"""
Minimal LLaVA-1.5-7B test with only 3 samples to verify technical functionality.
This is a safety test before running the full experiment on the laptop.
"""

import os
import sys
import torch
import pandas as pd
from PIL import Image
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_handler import load_llava, process_batch_for_llava
from src.data_loader import FakedditDataset
from src.prompts import LLAVA_PROMPTS
from src.evaluation import evaluate_model_outputs, compute_qualitative_stats
from src.blip_parser import LLaVAAnswerParser
from src.utils import setup_logger

# Setup logging
logger = setup_logger(__name__)

def test_llava_mini():
    """Test LLaVA-1.5-7B with only 3 samples"""
    
    print("=== LLaVA-1.5-7B Mini Test (3 Samples) ===")
    
    # Configuration for mini test
    config = {
        'model_name': 'llava-hf/llava-1.5-7b-hf',
        'prompt_name': 'llava_match_metadata',
        'num_samples': 3,
        'batch_size': 1,  # Small batch size for laptop
        'max_new_tokens': 100,  # Longer responses for detailed analysis
        'device': 'cpu'  # Force CPU for safety
    }
    
    print(f"Config: {config}")
    
    # 1. Load model and processor
    print("\n1. Loading LLaVA model...")
    model, processor = load_llava(config['model_name'])
    
    if model is None or processor is None:
        print("❌ Failed to load LLaVA model")
        return False
    
    print("✅ LLaVA model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Processor type: {type(processor).__name__}")
    
    # Move to device
    model.to(config['device'])
    model.eval()
    
    # 2. Load minimal dataset (3 samples)
    print("\n2. Loading minimal dataset...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    metadata_dir = os.path.join(project_root, 'data', 'processed')
    downloaded_image_dir = os.path.join(project_root, 'data', 'downloaded_fakeddit_images')
    
    dataset = FakedditDataset(
        metadata_dir=metadata_dir,
        metadata_file_name="test_balanced_pairs_clean.csv",
        downloaded_image_dir=downloaded_image_dir,
        transform=None,  # Get PIL images
        text_col='clean_title',
        label_col='2_way_label',
        extra_metadata_fields=['created_utc', 'domain', 'author', 'subreddit']
    )
    
    if len(dataset) == 0:
        print("❌ Dataset is empty")
        return False
    
    print(f"✅ Dataset loaded with {len(dataset)} total samples")
    
    # Automatically choose 3 valid examples: ID in dataset & image file exists, >10KB, and loadable with PIL
    img_dir = os.path.join(project_root, 'data', 'downloaded_fakeddit_images')
    available_ids = []
    for fname in os.listdir(img_dir):
        if not fname.endswith('.jpg'):
            continue
        path = os.path.join(img_dir, fname)
        if os.path.getsize(path) <= 10*1024:
            continue
        try:
            with Image.open(path) as im:
                im.verify()  # checks if image is really readable
            available_ids.append(fname[:-4])
        except Exception as e:
            print(f"❌ Image {fname} is corrupted or not a real image: {e}")
    test_samples = []
    for sample in dataset:
        if sample['id'] in available_ids:
            test_samples.append(sample)
        if len(test_samples) == 3:
            break
    print(f"✅ Automatically chose 3 PIL-readable samples: {[s['id'] for s in test_samples]}")
    
    # 3. Get prompt template
    print("\n3. Setting up prompt...")
    # Use the LLaVA Metadata Prompt for better clarity
    prompt_template = LLAVA_PROMPTS.get(config['prompt_name'])
    if not prompt_template:
        print(f"❌ Prompt '{config['prompt_name']}' not found!")
        return False
    print(f"✅ Using LLaVA metadata prompt: {config['prompt_name']}")
    print(f"   Template: {prompt_template}")
    
    # 4. Process samples
    print("\n4. Processing samples...")
    all_results = []
    
    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            print(f"\n   Processing sample {i+1}/{len(test_samples)}")
            print(f"   ID: {sample['id']}")
            print(f"   Text: {sample['text'][:100]}...")
            print(f"   Label: {sample['label']}")
            
            # Create batch format with metadata
            batch = {
                'id': [sample['id']],
                'text': [sample['text']],
                'image': [sample['image']],
                'label': [sample['label']],
                'created_utc': [sample.get('created_utc', 'N/A')],
                'domain': [sample.get('domain', 'N/A')],
                'author': [sample.get('author', 'N/A')],
                'subreddit': [sample.get('subreddit', 'N/A')]
            }
            
            # Check image size and mode and adjust if necessary (224x224, RGB)
            for idx in range(len(batch['image'])):
                img = batch['image'][idx]
                print(f"      Image type: {type(img)}")
                # Resize if necessary
                if hasattr(img, 'size') and img.size != (224, 224):
                    img = img.resize((224, 224), Image.BICUBIC)
                    print(f"      Image was resized to 224x224.")
                # Check mode and convert if necessary
                if hasattr(img, 'mode'):
                    print(f"      Image mode: {img.mode}")
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        print(f"      Image was converted to RGB.")
                else:
                    print(f"      Image has no mode attribute")
                batch['image'][idx] = img
                print(f"      Image size after adjustment: {img.size}")
            # Process with LLaVA (pass images as simple list)
            try:
                batch_for_llava = batch.copy()
                batch_for_llava['image'] = batch['image']
                
                # Prepare metadata string for the prompt
                metadata_parts = []
                for field in ['created_utc', 'domain', 'author', 'subreddit']:
                    if field in batch and batch[field][0] != 'N/A':
                        metadata_parts.append(f"{field}: {batch[field][0]}")
                metadata_string = "; ".join(metadata_parts) if metadata_parts else "No metadata available"
                
                # Format prompt with metadata
                formatted_prompt = prompt_template.format(text=batch['text'][0], metadata=metadata_string)
                
                inputs = process_batch_for_llava(
                    batch_for_llava, 
                    processor, 
                    config['device'], 
                    formatted_prompt
                )
                
                if inputs is None:
                    print(f"   ❌ Failed to process batch for sample {i+1}")
                    continue
                
                # Move inputs to device
                inputs = {k: v.to(config['device']) for k, v in inputs.items()}
                
                # Generate response
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=config['max_new_tokens']
                )
                
                generated_texts = processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
                
                generated_text = generated_texts[0].strip()
                print(f"   ✅ Generated: {generated_text}")
                
                # Store result
                result = {
                    'id': sample['id'],
                    'text': sample['text'],
                    'image_path': dataset.get_image_path(sample['id']),
                    'true_label': sample['label'],
                    'generated_text': generated_text
                }
                all_results.append(result)
                
            except Exception as e:
                print(f"   ❌ Error processing sample {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 5. Save results
    print("\n5. Saving results...")
    results_df = pd.DataFrame(all_results)
    
    # Create output directory
    output_dir = os.path.join(project_root, 'results', 'llava', 'mini_test')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    output_csv_path = os.path.join(output_dir, "mini_test_results.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"✅ Results saved to: {output_csv_path}")
    
    # 6. Basic evaluation
    print("\n6. Basic evaluation...")
    if len(all_results) > 0:
        print(f"✅ Successfully processed {len(all_results)}/{config['num_samples']} samples")
        
        # Show sample results
        print("\nSample Results:")
        for i, result in enumerate(all_results):
            print(f"\nSample {i+1}:")
            print(f"  ID: {result['id']}")
            print(f"  True Label: {result['true_label']}")
            print(f"  Generated: {result['generated_text']}")
        
        # --- Professionelle Auswertung mit LLaVAAnswerParser ---
        print("\n7. Professionelle Auswertung mit LLaVAAnswerParser:")
        llava_parser = LLaVAAnswerParser()
        
        # Parse alle generierten Texte
        parsed_results = []
        for idx, row in results_df.iterrows():
            generated_text = row.get('generated_text', '')
            predicted_label, confidence, explanation = llava_parser.extract_prediction(generated_text)
            parsed_results.append({
                'predicted_label': predicted_label,
                'confidence': confidence,
                'parsing_explanation': explanation
            })
        
        # Ergebnisse zum DataFrame hinzufügen
        results_df['predicted_label'] = [r['predicted_label'] for r in parsed_results]
        results_df['confidence'] = [r['confidence'] for r in parsed_results]
        results_df['parsing_explanation'] = [r['parsing_explanation'] for r in parsed_results]
        
        # Nur sichere Vorhersagen für Accuracy-Berechnung
        certain_predictions = results_df[results_df['predicted_label'].notna()]
        uncertain_count = len(results_df) - len(certain_predictions)
        
        if len(certain_predictions) > 0:
            accuracy = (certain_predictions['predicted_label'] == certain_predictions['true_label']).mean()
            print(f"LLaVA Parser: {len(certain_predictions)} sichere Vorhersagen, {uncertain_count} unsichere")
            print(f"Accuracy (nur sichere): {accuracy*100:.1f}%")
        else:
            print("LLaVA Parser: Keine sicheren Vorhersagen gefunden")
            accuracy = 0.0
        
        # Zeige detaillierte Ergebnisse
        print("\nDetaillierte Ergebnisse:")
        for idx, row in results_df.iterrows():
            print(f"ID: {row['id']}")
            print(f"  True Label: {row['true_label']}")
            print(f"  Generated: {row['generated_text'][:100]}...")
            print(f"  Parsed Label: {row['predicted_label']}")
            print(f"  Confidence: {row['confidence']}")
            print(f"  Explanation: {row['parsing_explanation']}")
            print()
        
        print(f"Gesamt-Accuracy: {accuracy*100:.1f}% bei {len(certain_predictions)} von {len(results_df)} Samples")
        print(f"Unsichere Modellantworten: {uncertain_count}")
        
        # Optional: Speichere die Auswertung mit ab
        results_df.to_csv(output_csv_path, index=False)
        
        return True
    else:
        print("❌ No samples were successfully processed")
        return False

if __name__ == "__main__":
    success = test_llava_mini()
    if success:
        print("\n🎉 LLaVA mini test PASSED! Ready for full experiment.")
    else:
        print("\n❌ LLaVA mini test FAILED! Check issues before full experiment.") 