# pipeline.py
import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd # For storing results
import json
from PIL import Image

from data_loader import FakedditDataset # Assuming it's in the same directory or PYTHONPATH is set
from model_handler import (
    load_clip, process_batch_for_clip, 
    load_blip_conditional, process_batch_for_blip_conditional,
    load_bert_classifier, process_batch_for_bert_classifier,
    load_llava, process_batch_for_llava # Added LLaVA imports
)
from utils import setup_logger # Import the logger setup function
from evaluation import evaluate_model_outputs # Import the evaluation function

# Setup logger for this module
logger = setup_logger(__name__) # Use __name__ for the logger name, or a custom one

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

    # --- Adjust Output Directory based on Model Type and Experiment Name ---
    # Base model directory (e.g., results/clip or results/blip)
    base_model_output_dir = os.path.join(args.output_dir, args.model_type)
    
    # Final output path including experiment name if provided
    if args.experiment_name:
        final_output_path = os.path.join(base_model_output_dir, args.experiment_name)
    else:
        # If no experiment name, outputs go directly into the base model directory
        final_output_path = base_model_output_dir 
    
    args.output_dir = final_output_path # Update args.output_dir for the rest of the script
    
    logger.info(f"Outputs for model type '{args.model_type}' (Experiment: {args.experiment_name or 'default'}) will be saved to: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True) # Ensure the specific directory exists

    # --- 1. Load Data ---
    print("\n--- Loading Data ---")
    # Construct full paths based on project structure
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
    experiment_config = { # Store experiment configuration
        "model_type": args.model_type,
        "data_dir": args.data_dir,
        "metadata_file": args.metadata_file,
        "num_samples": args.num_samples if args.num_samples != -1 else 'all',
        "batch_size": args.batch_size,
        "clip_model_name": args.clip_model_name if args.model_type == 'clip' else None,
        "blip_model_name": args.blip_model_name if args.model_type == 'blip' else None,
        "bert_model_name": args.bert_model_name if args.model_type == 'bert' else None,
        "llava_model_name": args.llava_model_name if args.model_type == 'llava' else None,
    }

    logger.info(f"Starting pipeline with config: {json.dumps(experiment_config, indent=2)}")

    current_device = DEVICE
    # if args.model_type == 'llava':
    #     logger.warning("LLaVA model is large. Forcing CPU for LLaVA to potentially avoid MPS memory/performance issues. This will be slower.")
    #     current_device = "cpu"

    if args.model_type == 'clip':
        model, processor = load_clip(args.clip_model_name)
        process_batch_fn = process_batch_for_clip
    elif args.model_type == 'blip':
        model, processor = load_blip_conditional(args.blip_model_name)
        # Define a wrapper for BLIP processing to match expected signature by main loop
        def blip_process_wrapper(batch, proc, dev):
            pil_images = batch['image']
            original_texts = batch['text']
            batch_size = len(original_texts)
            
            final_generated_texts = []

            # Step 1: Get Yes/No answer
            yes_no_prompts = [f"Does the provided image perfectly and completely match the claim made in the text: '{t}'? Answer with only 'No.' if there is *any* mismatch, discrepancy, or missing element, however small. Otherwise, answer 'Yes.'." for t in original_texts]
            inputs_step1 = proc(images=pil_images, text=yes_no_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs_step1 = {k: v.to(dev) for k, v in inputs_step1.items()}
            
            # Use a small number of max_new_tokens for the Yes/No answer
            generated_ids_step1 = model.generate(**inputs_step1, max_new_tokens=20)
            yes_no_answers_raw = proc.batch_decode(generated_ids_step1, skip_special_tokens=True)
            
            # Normalize and clean Yes/No answers
            yes_no_answers = []
            for ans_raw in yes_no_answers_raw:
                ans_clean = ans_raw.strip().lower()
                if ans_clean.startswith("yes"):
                    yes_no_answers.append("Yes.")
                elif ans_clean.startswith("no"):
                    yes_no_answers.append("No.")
                else:
                    logger.warning(f"Could not parse Yes/No answer: '{ans_raw}'. Defaulting to 'No.' for explanation prompt.")
                    yes_no_answers.append("No.") # Default or handle as error

            # Step 2: Get Explanation based on Yes/No answer
            explanation_prompts = []
            for i in range(batch_size):
                text = original_texts[i]
                answer = yes_no_answers[i]
                if answer == "Yes.":
                    explanation_prompts.append(f"The image seems to perfectly and completely match the text: '{text}'. Explain concisely what specific elements in the image confirm the text's main claims without ambiguity.")
                else: # Covers "No." and defaults from parsing issues
                    explanation_prompts.append(f"The image appears to have mismatches or missing elements compared to the text: '{text}'. Explain clearly and concisely what the *key* mismatches, discrepancies, or missing elements are.")

            inputs_step2 = proc(images=pil_images, text=explanation_prompts, return_tensors="pt", padding=True, truncation=True)
            inputs_step2 = {k: v.to(dev) for k, v in inputs_step2.items()}
            
            # Use args.max_new_tokens_blip for the explanation
            generated_ids_step2 = model.generate(**inputs_step2, max_new_tokens=args.max_new_tokens_blip)
            explanations = proc.batch_decode(generated_ids_step2, skip_special_tokens=True)

            # Combine Yes/No answer with explanation
            for i in range(batch_size):
                combined_text = f"{yes_no_answers[i]} {explanations[i].strip()}"
                final_generated_texts.append(combined_text)
            
            return final_generated_texts, None # scores are None for this BLIP setup

        process_batch_fn = blip_process_wrapper
        logger.info(f"Using BLIP model: {args.blip_model_name} with two-step explanation-focused prompt.")
    elif args.model_type == 'llava': # New LLaVA branch
        model, processor = load_llava(args.llava_model_name)
        def llava_process_wrapper(batch, proc, dev):
            # The llava_prompt_template comes from args
            inputs = process_batch_for_llava(batch, proc, dev, args.llava_prompt_template)
            if inputs is None: # Error occurred in processing
                return ["Error processing batch for LLaVA"] * len(batch['id']), None 
            
            # Ensure inputs are on the correct device (model.generate expects this)
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens_llava)
            generated_texts = proc.batch_decode(generated_ids, skip_special_tokens=True)
            return generated_texts, None

        process_batch_fn = llava_process_wrapper
        logger.info(f"Using LLaVA model: {args.llava_model_name} with prompt template: {args.llava_prompt_template}")
    elif args.model_type == 'bert':
        model, processor = load_bert_classifier(args.bert_model_name, num_labels=args.num_labels) # processor is bert_tokenizer here
        process_batch_fn = process_batch_for_bert_classifier
    else:
        print(f"Unsupported model type: {args.model_type}")
        return

    if model is None or processor is None:
        print("Model or processor could not be loaded. Exiting.")
        return
    
    model.to(current_device)
    model.eval() # Set to evaluation mode

    # --- 3. Process Batches & Run Model (Example) ---
    print("\n--- Processing Batches & Running Model (Example) ---")
    if args.num_samples == -1:
        num_batches_to_process = float('inf') # Process all batches
        print(f"Processing all samples from the dataset.")
    else:
        num_batches_to_process = (args.num_samples + args.batch_size - 1) // args.batch_size
        print(f"Processing approximately {args.num_samples} samples in {num_batches_to_process} batches.")

    with torch.no_grad(): # Ensure no gradients are computed if just doing inference
        for i, batch in enumerate(dataloader):
            if i >= num_batches_to_process:
                break
            if batch is None: # From collate_fn if all items in batch failed
                print(f"Skipping empty batch {i+1}.")
                continue

            print(f"\nProcessing batch {i+1} with {len(batch['id'])} samples.")
            
            try:
                scores = None # Initialize scores
                generated_texts = None # Initialize generated_texts
                predictions = None # Initialize predictions
                inputs = None # Initialize inputs

                if args.model_type == 'clip':
                    inputs = process_batch_fn(batch, processor, current_device) # Use current_device
                    outputs = model(**inputs)
                    # logits_per_image shape: (image_batch_size, text_batch_size)
                    # Assuming one text per image, and they are aligned in the batch by the processor
                    if inputs['input_ids'].shape[0] == inputs['pixel_values'].shape[0]:
                        scores = outputs.logits_per_image.diag() # Paired similarity
                    else:
                        # Fallback or specific logic for multiple texts per image needed here
                        logger.warning("CLIP: Mismatch in image/text batch sizes for logits. Using first text logit.")
                        scores = outputs.logits_per_image[:, 0] 
                    logger.info(f"  CLIP: logits_per_image shape {outputs.logits_per_image.shape}, extracted scores shape {scores.shape}")
                    # Placeholder for actual prediction logic based on scores and task
                    # The binary prediction based on threshold is done later during evaluation prep
                    # predictions = (scores >= 0.3).int() # Example, this depends on score scaling and meaning

                elif args.model_type == 'blip':
                    # BLIP model loading already handles device_map="auto" or .to(DEVICE)
                    # For consistency, ensure its process_batch_fn also uses current_device if it were ever not 'auto'
                    # However, BLIP's device_map='auto' inside load_blip_conditional might conflict if we force CPU here.
                    # Let's assume BLIP is fine on MPS for now and only force CPU for LLaVA.
                    # If BLIP also causes issues, its device logic might need review.
                    generated_texts, _ = process_batch_fn(batch, processor, DEVICE) # Keep DEVICE for BLIP unless issues arise
                    # For BLIP with explanation, 'predictions' are qualitative (the text itself)
                    # No direct numerical predictions unless we parse generated_texts for yes/no etc.
                    for idx_text, gen_text in enumerate(generated_texts):
                        logger.info(f"  BLIP Generated Text for ID {batch['id'][idx_text]}: '{gen_text.strip()}'")

                elif args.model_type == 'bert':
                    inputs = process_batch_fn(batch, processor, current_device) # Use current_device
                    outputs = model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=1)
                
                elif args.model_type == 'llava':
                    generated_texts, _ = process_batch_fn(batch, processor, current_device) # Use current_device ('cpu')
                    for idx_text, gen_text in enumerate(generated_texts):
                        logger.info(f"  LLaVA Generated Text for ID {batch['id'][idx_text]}: '{gen_text.strip()}'")
                
                logger.info(f"  Batch {i+1} processed for {args.model_type}.")

                # Store results for this batch
                for idx in range(len(batch['id'])):
                    item_id = batch['id'][idx]
                    true_label = batch['label'][idx].item()
                    
                    # Initialize per-item results
                    pred_label_item = None
                    score_item = None
                    generated_text_item = None

                    if args.model_type == 'clip' and scores is not None:
                        score_item = scores[idx].item()
                        # Binary prediction from score is handled later before evaluation function call
                    elif args.model_type == 'bert' and predictions is not None:
                        pred_label_item = predictions[idx].item()
                    elif args.model_type == 'blip' and generated_texts is not None:
                        generated_text_item = generated_texts[idx]
                    elif args.model_type == 'llava' and generated_texts is not None: # Handling LLaVA results
                        generated_text_item = generated_texts[idx]
                    
                    result_entry = {
                        'id': item_id,
                        'text': batch['text'][idx],
                        'image_path': dataset.get_image_path(item_id),
                        'true_label': true_label,
                    }
                    if pred_label_item is not None:
                        result_entry['predicted_label'] = pred_label_item
                    if score_item is not None:
                        result_entry['scores'] = score_item # This is the CLIP similarity score
                    if generated_text_item is not None:
                        result_entry['generated_text'] = generated_text_item
                    
                    all_results.append(result_entry)

            except Exception as e:
                print(f"Error during batch {i+1} processing or model inference: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next batch if one fails, or re-raise if critical

    print("\n--- Pipeline Finished (Example Run) ---")

    # --- 4. Save Results & Evaluate ---
    results_df = pd.DataFrame(all_results)
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Clean up image_path to just be the filename for cleaner CSVs
    if 'image_path' in results_df.columns:
        results_df['image_path'] = results_df['image_path'].apply(lambda x: os.path.basename(x) if isinstance(x, str) else x)

    # Save all raw outputs
    output_csv_path = os.path.join(args.output_dir, "all_model_outputs.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nAll model outputs saved to {output_csv_path}")

    # --- Evaluation ---
    # Determine true and predicted labels based on model type
    # For CLIP, we might use a threshold on similarity scores to get binary predictions.
    # For BLIP, we might parse the generated_text or use it directly for qualitative eval.
    # For BERT, predictions are direct class labels.

    true_labels_col = 'true_label' # Assuming 'true_label' is the GT column from the result_entry

    if args.model_type == 'clip' and 'scores' in results_df.columns:
        # Example: Convert similarity scores to binary predictions using a threshold
        # This threshold might need tuning or be part of the analysis.
        # IMPORTANT: Based on initial data exploration, for THIS dataset,
        # higher CLIP scores seem to correlate with true_label == 1 (e.g., "fake").
        # Therefore, the prediction logic is inverted compared to a typical similarity search.
        similarity_threshold = 27.5 # Refined based on analysis of score distributions
        results_df['predicted_labels'] = results_df['scores'].apply(lambda x: 1 if x >= similarity_threshold else 0)
        logger.info(f"Derived 'predicted_labels' for CLIP using threshold {similarity_threshold}. Higher score predicts 'fake' (1) due to observed data characteristic.")
        pred_labels_col = 'predicted_labels'
        # For the report, we might want to show the score directly
        evaluation_report_path = os.path.join(args.output_dir, "reports", f"{args.model_type}_{args.clip_model_name.replace('/', '_')}_evaluation_report.txt")
        figures_dir = os.path.join(args.output_dir, "figures", f"{args.model_type}_{args.clip_model_name.replace('/', '_')}")
    elif args.model_type == 'blip' and 'generated_text' in results_df.columns:
        # For BLIP, 'predicted_labels' might be harder to derive automatically for quantitative metrics
        # without parsing the 'generated_text'. For the mini-experiment, qualitative is key.
        # We will include generated_text in the report.
        logger.info("BLIP run: 'generated_text' available. Quantitative metrics for classification may require parsing this text.")
        pred_labels_col = None # No direct classification prediction column yet for BLIP
        evaluation_report_path = os.path.join(args.output_dir, "reports", f"{args.model_type}_{args.blip_model_name.replace('/', '_')}_evaluation_report.txt")
        figures_dir = os.path.join(args.output_dir, "figures", f"{args.model_type}_{args.blip_model_name.replace('/', '_')}")
    elif args.model_type == 'bert' and 'predicted_labels' in results_df.columns: # Bert directly outputs class
        pred_labels_col = 'predicted_labels'
        logger.info("BERT run: Using 'predicted_labels' from model output.")
        evaluation_report_path = os.path.join(args.output_dir, "reports", f"{args.model_type}_{args.bert_model_name.replace('/', '_')}_evaluation_report.txt")
        figures_dir = os.path.join(args.output_dir, "figures", f"{args.model_type}_{args.bert_model_name.replace('/', '_')}")
    elif args.model_type == 'llava' and 'generated_text' in results_df.columns: # Evaluation for LLaVA
        logger.info("LLaVA run: 'generated_text' available. Quantitative metrics for classification may require parsing this text.")
        pred_labels_col = None 
        evaluation_report_path = os.path.join(args.output_dir, "reports", f"{args.model_type}_{args.llava_model_name.replace('/', '_')}_evaluation_report.txt")
        figures_dir = os.path.join(args.output_dir, "figures", f"{args.model_type}_{args.llava_model_name.replace('/', '_')}")
    else:
        logger.warning("Could not determine predicted labels column for evaluation or suitable data not found.")
        pred_labels_col = None
        evaluation_report_path = os.path.join(args.output_dir, "reports", f"{args.model_type}_evaluation_report.txt")
        figures_dir = os.path.join(args.output_dir, "figures", f"{args.model_type}")

    os.makedirs(os.path.dirname(evaluation_report_path), exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if true_labels_col in results_df.columns and pred_labels_col and pred_labels_col in results_df.columns:
        logger.info(f"Running evaluation with true_labels='{true_labels_col}' and predicted_labels='{pred_labels_col}'")
        evaluate_model_outputs(
            results_df,
            true_label_col=true_labels_col,
            pred_label_col=pred_labels_col,
            generated_text_col='generated_text' if 'generated_text' in results_df.columns else None,
            report_path=evaluation_report_path,
            figures_dir=figures_dir
        )
    elif 'generated_text' in results_df.columns and (args.model_type == 'blip' or args.model_type == 'llava'):
        # For BLIP/LLaVA, even without derived predicted_labels, save a report with generated text
        logger.info(f"{args.model_type.upper()} run: Saving a report focusing on generated text as 'predicted_labels' are not derived.")
        # Create a dummy 'predicted_labels' if evaluate_model_outputs requires it, or modify evaluation.
        # For now, let's assume qualitative analysis for BLIP/LLaVA.
        # We can still save the generated text to a file.
        explanation_file_name = f"{args.model_type}_{ (args.blip_model_name if args.model_type == 'blip' else args.llava_model_name).replace('/', '_') }_explanations.txt"
        explanations_path = os.path.join(args.output_dir, "reports", explanation_file_name)
        with open(explanations_path, 'w') as f:
            for index, row in results_df.iterrows():
                f.write(f"ID: {row.get('id', index)}\n")
                f.write(f"Text: {row.get('text', 'N/A')}\n")
                f.write(f"True Label: {row.get(true_labels_col, 'N/A')}\n")
                f.write(f"Generated Explanation: {row.get('generated_text', 'N/A')}\n")
                f.write("-" * 30 + "\n")
        logger.info(f"{(args.blip_model_name if args.model_type == 'blip' else args.llava_model_name).replace('/', '_')}_explanations saved to {explanations_path}")
    else:
        logger.warning("Evaluation skipped: True labels or predicted labels column not found or not applicable.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multimodal Fact-Checking Pipeline")
    
    # Data Args
    parser.add_argument("--data_dir", type=str, default="data", help="Directory for data, relative to project root.")
    parser.add_argument("--metadata_file", type=str, default="multimodal_train.csv", help="Metadata CSV file name in data_dir/raw.")
    parser.add_argument("--text_column", type=str, default="clean_title", help="Column name for text in metadata CSV.")
    parser.add_argument("--label_column", type=str, default="2_way_label", help="Column name for labels in metadata CSV.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process. -1 for all.")

    # Model Args
    parser.add_argument("--model_type", type=str, choices=['clip', 'blip', 'bert', 'llava'], required=True, help="Type of model to use.")
    parser.add_argument("--clip_model_name", type=str, default="openai/clip-vit-base-patch32", help="CLIP model name from Hugging Face.")
    parser.add_argument("--blip_model_name", type=str, default="Salesforce/blip-vqa-base", help="BLIP model name (e.g., Salesforce/blip-vqa-base, Salesforce/blip-image-captioning-base).")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased", help="BERT model name for text classification.")
    parser.add_argument("--llava_model_name", type=str, default="llava-hf/llava-1.5-7b-hf", help="LLaVA model name from Hugging Face.")
    parser.add_argument("--num_labels", type=int, default=2, help="Number of labels for classification (e.g., 2 for fake/real).")
    parser.add_argument("--blip_task", type=str, default="vqa", choices=["vqa", "captioning", "classification_prompted"], help="Task for BLIP model conditional processing.")
    parser.add_argument("--blip_max_text_length", type=int, default=32, help="Max text length for BLIP processor (old, might be deprecated by specific model processing).")
    parser.add_argument("--llava_prompt_template", type=str, 
                        default="USER: <image>\nGiven the image and the caption '{text}', is this post misleading? Why or why not?\nASSISTANT:", 
                        help="Prompt template for LLaVA. Use '{text}' for caption and ensure '<image>' token is present for the processor.")
    parser.add_argument("--max_new_tokens_blip", type=int, default=50, help="Max new tokens for BLIP processor for the explanation part in two-step.")
    parser.add_argument("--max_new_tokens_llava", type=int, default=100, help="Max new tokens for LLaVA generation.")

    # DataLoader and Processing arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")

    # Output Args
    parser.add_argument("--output_dir", type=str, default="results", help="Base directory to save all outputs. Model-specific subfolders will be created here.")
    parser.add_argument("--experiment_name", type=str, default=None, help="Optional name for the specific experiment run, creating a subfolder within results/<model_type>/")

    args = parser.parse_args()
    main(args)
