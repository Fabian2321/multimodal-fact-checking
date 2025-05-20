# pipeline.py
import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd # For storing results

from data_loader import FakedditDataset # Assuming it's in the same directory or PYTHONPATH is set
from model_handler import (
    load_clip, process_batch_for_clip, 
    load_blip_conditional, process_batch_for_blip_conditional,
    load_bert_classifier, process_batch_for_bert_classifier # Added BERT imports
)

# Determine device
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

def collate_pil_batch(batch):
    """Custom collate_fn for batches where 'image' contains PIL Images."""
    items = [item for item in batch if item is not None]
    if not items:
        return None
    
    ids = [item['id'] for item in items]
    images = [item['image'] for item in items] # List of PIL Images
    texts = [item['text'] for item in items]
    labels = torch.tensor([item['label'] for item in items], dtype=torch.long)
    
    return {'id': ids, 'image': images, 'text': texts, 'label': labels}

def main(args):
    print(f"Starting pipeline with arguments: {args}")

    # --- 1. Load Data ---
    print("\n--- Loading Data ---")
    # Construct full paths based on project structure
    # Assumes this script is in multimodal-fact-checking/src/
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    metadata_dir = os.path.join(project_root, args.data_dir, 'raw')
    downloaded_image_dir = os.path.join(project_root, args.data_dir, 'downloaded_images_pipeline')
    
    print(f"Metadata directory: {metadata_dir}")
    print(f"Image download directory: {downloaded_image_dir}")

    # Initialize dataset to return PIL images for model processors
    dataset = FakedditDataset(
        metadata_dir=metadata_dir,
        metadata_file_name=args.metadata_file,
        downloaded_image_dir=downloaded_image_dir,
        transform=None, # Crucial: Get PIL images
        text_col=args.text_column,
        label_col=args.label_column
    )

    if not dataset or len(dataset) == 0:
        print("Dataset could not be loaded or is empty. Exiting.")
        return

    print(f"Dataset loaded with {len(dataset)} samples.")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # Usually False for validation/testing, True for training
        num_workers=args.num_workers, 
        collate_fn=collate_pil_batch
    )

    # --- 2. Load Model & Processor ---
    print("\n--- Loading Model & Processor ---")
    model = None
    processor = None # General term, could be CLIPProcessor or BlipProcessor or BertTokenizer
    process_batch_fn = None
    all_results = [] # To store predictions and labels for evaluation

    if args.model_type == 'clip':
        model, processor = load_clip(args.clip_model_name)
        process_batch_fn = process_batch_for_clip
    elif args.model_type == 'blip':
        model, processor = load_blip_conditional(args.blip_model_name)
        def blip_process_wrapper(batch, proc, dev):
            return process_batch_for_blip_conditional(batch, proc, task=args.blip_task, device=dev, max_length=args.blip_max_text_length)
        process_batch_fn = blip_process_wrapper
    elif args.model_type == 'bert':
        model, processor = load_bert_classifier(args.bert_model_name, num_labels=args.num_labels) # processor is bert_tokenizer here
        process_batch_fn = process_batch_for_bert_classifier
    else:
        print(f"Unsupported model type: {args.model_type}")
        return

    if model is None or processor is None:
        print("Model or processor could not be loaded. Exiting.")
        return
    
    model.to(DEVICE)
    model.eval() # Set to evaluation mode

    # --- 3. Process Batches & Run Model (Example) ---
    print("\n--- Processing Batches & Running Model (Example) ---")
    num_batches_to_process = args.num_test_batches

    with torch.no_grad(): # Ensure no gradients are computed if just doing inference
        for i, batch in enumerate(dataloader):
            if i >= num_batches_to_process:
                break
            if batch is None: # From collate_fn if all items in batch failed
                print(f"Skipping empty batch {i+1}.")
                continue

            print(f"\nProcessing batch {i+1} with {len(batch['id'])} samples.")
            
            try:
                # Model-specific batch processing
                inputs = process_batch_fn(batch, processor, DEVICE)
                print(f"  Batch processed for {args.model_type}. Input keys: {list(inputs.keys())}")

                # Forward pass & Prediction Gathering
                if args.model_type == 'clip':
                    outputs = model(**inputs)
                    # For CLIP, similarity scores (logits_per_image) are often used.
                    # Higher score means more similar. For binary fake/real, this needs careful mapping.
                    # Example: if text is "a factual post" vs "a misleading post", and compare image to these two.
                    # For simplicity here, let's assume a placeholder for actual classification logic with CLIP.
                    # We would typically compare image embeddings to text embeddings of "real" and "fake" prompts.
                    # This is a complex step, so for now, we'll simulate predictions.
                    # Placeholder: In a real scenario, you'd derive binary predictions from logits.
                    # For now, let's assume logits_per_image directly relates to class prediction if we craft text prompts carefully
                    # or use a different head. This part needs more sophisticated handling for actual classification.
                    print(f"  CLIP Output logits_per_image shape: {outputs.logits_per_image.shape}")
                    # Simulated predictions for CLIP (needs proper implementation)
                    # Assuming the first logit corresponds to some notion of "realness"
                    # This is NOT a valid way to get predictions from raw CLIP for fake news detection without further setup.
                    # It's a placeholder to make the pipeline runnable.
                    # A proper way would be to compute similarity with prompts like "this is real news" and "this is fake news".
                    predictions = torch.argmax(outputs.logits_per_image, dim=1) 

                elif args.model_type == 'blip':
                    # BLIP outputs for classification or VQA can vary.
                    # For classification_prompted, we'd expect generated text like "yes"/"no" or "factual"/"misleading".
                    generated_texts_for_eval = []
                    if args.blip_task == "captioning":
                        generated_ids = model.generate(**inputs, max_length=args.blip_max_text_length + 20)
                        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                        for idx, gen_text in enumerate(generated_texts):
                            print(f"  BLIP Generated Caption for ID {batch['id'][idx]}: {gen_text.strip()}")
                            generated_texts_for_eval.append(gen_text.strip())
                        # For captioning, direct prediction of fake/real isn't straightforward.
                        # We'd need to evaluate the quality of captions or use them in a downstream task.
                        # Placeholder predictions for now.
                        predictions = torch.randint(0, args.num_labels, (len(batch['id']),)).to(DEVICE)
                    elif args.blip_task == "vqa" or args.blip_task == "classification_prompted":
                        answer_ids = model.generate(**inputs, max_length=args.blip_max_text_length + 10) # Max length for answer
                        answers = processor.batch_decode(answer_ids, skip_special_tokens=True)
                        for idx, ans_text in enumerate(answers):
                            print(f"  BLIP Generated Answer for ID {batch['id'][idx]} (Q: {batch['text'][idx][:50]}...): {ans_text.strip()}")
                            generated_texts_for_eval.append(ans_text.strip())
                        # Convert VQA answers (e.g., "yes"/"no") to numerical predictions
                        # This mapping needs to be defined based on expected answers
                        # Placeholder: maps "yes" to 1 (fake), others to 0 (real), needs refinement
                        current_preds = []
                        for ans in answers:
                            if "yes" in ans.lower(): current_preds.append(1)
                            elif "true" in ans.lower(): current_preds.append(1) # Assuming 1 is "misleading/fake"
                            elif "factual" in ans.lower(): current_preds.append(0) # Assuming 0 is "real"
                            elif "real" in ans.lower(): current_preds.append(0)
                            else: current_preds.append(0) # Default for unclear answers
                        predictions = torch.tensor(current_preds).to(DEVICE)
                    else:
                        print(f"  BLIP task '{args.blip_task}' basic processing done. Add specific output handling.")
                        predictions = torch.randint(0, args.num_labels, (len(batch['id']),)).to(DEVICE) # Placeholder
                        generated_texts_for_eval = [None] * len(batch['id'])

                elif args.model_type == 'bert':
                    outputs = model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=1)
                    generated_texts_for_eval = [None] * len(batch['id']) # BERT doesn't generate text in this setup

                # Store results for this batch
                for idx in range(len(batch['id'])):
                    item_id = batch['id'][idx]
                    true_label = batch['label'][idx].item()
                    pred_label = predictions[idx].item()
                    # Store generated text if available (e.g., from BLIP)
                    generated_text = generated_texts_for_eval[idx] if generated_texts_for_eval else None
                    all_results.append({
                        'id': item_id,
                        'true_label': true_label,
                        'predicted_label': pred_label,
                        'generated_text': generated_text # For BLIP explanations/captions
                    })

            except Exception as e:
                print(f"Error during batch {i+1} processing or model inference: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next batch if one fails, or re-raise if critical

    print("\n--- Pipeline Finished (Example Run) ---")

    # --- 4. Save Results ---
    if args.results_file:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        results_df = pd.DataFrame(all_results)
        # Ensure results directory exists
        results_path = os.path.join(project_root, args.results_dir, args.results_file)
        os.makedirs(os.path.join(project_root, args.results_dir), exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")

        # Optional: Call evaluation script here if it's ready
        from evaluation import evaluate_model_outputs # Assuming evaluation.py is in the same directory
        report_dir = os.path.join(project_root, args.results_dir, 'reports')
        figures_dir = os.path.join(project_root, args.results_dir, 'figures', args.model_type) # Model-specific figures
        report_file_name = f"{os.path.splitext(args.results_file)[0]}_evaluation_report.txt"
        
        evaluate_model_outputs(
            results_df,
            true_label_col='true_label',
            pred_label_col='predicted_label',
            report_path=os.path.join(report_dir, report_file_name),
            figures_dir=figures_dir
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multimodal Fact-Checking Pipeline")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory for data (containing raw, processed, etc.) relative to project root.")
    parser.add_argument("--metadata_file", type=str, default="multimodal_train.csv", help="Name of the metadata CSV file in data_dir/raw/")
    parser.add_argument("--text_column", type=str, default="clean_title", help="Column name for text in metadata.")
    parser.add_argument("--label_column", type=str, default="2_way_label", help="Column name for labels in metadata.")

    # Model choice and arguments
    parser.add_argument("--model_type", type=str, choices=['clip', 'blip', 'bert'], default='clip', help="Type of model to use.")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name from Hugging Face.")
    parser.add_argument("--blip_model_name", type=str, default="Salesforce/blip-vqa-base", help="BLIP model name (e.g., Salesforce/blip-vqa-base, Salesforce/blip-image-captioning-base).")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased", help="BERT model name for text classification.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for classification (e.g., 2 for fake/real).")
    parser.add_argument("--blip_task", type=str, default="vqa", choices=["vqa", "captioning", "classification_prompted"], help="Task for BLIP model conditional processing.")
    parser.add_argument("--blip_max_text_length", type=int, default=32, help="Max text length for BLIP processor.")

    # DataLoader and Processing arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader. 0 for main process.")
    parser.add_argument("--num_test_batches", type=int, default=2, help="Number of batches to process for this example run.")

    # Results and Evaluation arguments
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results and evaluation outputs, relative to project root.")
    parser.add_argument("--results_file", type=str, default=None, help="Filename to save the prediction results (e.g., 'clip_predictions.csv'). If None, results are not saved to a file.")

    args = parser.parse_args()
    main(args)
