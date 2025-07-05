# pipeline.py
import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd # For storing results
import json
from PIL import Image
import sys
import logging
import numpy as np

from .data_loader import FakedditDataset # Fixed import
from .model_handler import (
    load_clip, process_batch_for_clip, 
    load_blip_conditional, process_batch_for_blip_conditional,
    load_bert_classifier, process_batch_for_bert_classifier,
    load_llava, process_batch_for_llava # Added LLaVA imports
)
from .utils import setup_logger # Import the logger setup function
from .evaluation import evaluate_model_outputs, compute_qualitative_stats, save_metrics_table # Import the evaluation function
from ..models.prompts import BLIP_PROMPTS, LLAVA_PROMPTS, CLIP_PROMPTS, FEW_SHOT_EXAMPLES, build_blip2_true_false_fewshot_prompt, prompt_library
from ..models.rag_handler import RAGHandler, RAGConfig  # Import RAG components
from ..models.blip_parser import BLIPAnswerParser, parse_blip_answer, LLaVAAnswerParser # Import BLIP and LLaVA parsers

# Setup logger for this module
logger = setup_logger(__name__)

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

    # --- Define extra metadata fields to use ---
    extra_metadata_fields = ['created_utc', 'domain', 'author', 'subreddit', 'title', 'num_comments', 'score', 'upvote_ratio', 'linked_submission_id']

    # --- Prepare Few-Shot Data (if applicable) ---
    few_shot_images = []
    few_shot_context = {}
    few_shot_examples_dynamic = None
    if args.use_few_shot and args.prompt_name == 'dynamic_blip2_true_false_fewshot':
        # Dynamische Auswahl: 2 True, 8 Fake aus Testset
        # Lade Testset
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        metadata_dir = os.path.join(project_root, args.data_dir, 'processed')
        testset_df = pd.read_csv(os.path.join(metadata_dir, args.metadata_file))
        true_samples = testset_df[testset_df[args.label_column] == 1].sample(n=2, random_state=42)
        fake_samples = testset_df[testset_df[args.label_column] == 0].sample(n=8, random_state=42)
        few_shot_examples_dynamic = []
        for _, row in pd.concat([true_samples, fake_samples]).sample(frac=1, random_state=42).iterrows():
            few_shot_examples_dynamic.append({
                'text': row[args.text_column],
                'label': 'True' if row[args.label_column] == 1 else 'False'
            })
        logger.info(f"Dynamisch gezogene Few-Shot-Beispiele (2T/8F): {few_shot_examples_dynamic}")
    elif args.use_few_shot:
        logger.info("--- Preparing Few-Shot Examples ---")
        # Load images and assemble text for prompt formatting
        for ex in FEW_SHOT_EXAMPLES:
            try:
                img = Image.open(ex["image_path"]).convert("RGB")
                few_shot_images.append(img)
                logger.info(f"Loaded few-shot image: {ex['image_path']}")
            except FileNotFoundError:
                logger.error(f"Few-shot image not found at {ex['image_path']}. Cannot proceed with few-shot.")
                return
        
        # Prepare a dictionary for easy .format() replacement in prompts
        real_ex = FEW_SHOT_EXAMPLES[0]
        fake_ex = FEW_SHOT_EXAMPLES[1]
        few_shot_context = {
            "real_example_text": real_ex["text"],
            "real_explanation_blip": real_ex["explanation_blip"],
            "real_explanation_llava": real_ex["explanation_llava"],
            "fake_example_text": fake_ex["text"],
            "fake_explanation_blip": fake_ex["explanation_blip"],
            "fake_explanation_llava": fake_ex["explanation_llava"],
        }
        logger.info("Few-shot examples prepared successfully.")

    # --- Initialize RAG if enabled ---
    rag_handler = None
    if args.use_rag:
        logger.info("Initializing RAG system")
        rag_config = RAGConfig(
            embedding_model=args.rag_embedding_model,
            top_k=args.rag_top_k,
            similarity_threshold=args.rag_similarity_threshold,
            knowledge_base_path=args.rag_knowledge_base_path
        )
        rag_handler = RAGHandler(rag_config)
        
        # Add initial knowledge base documents if provided
        if args.rag_initial_docs:
            logger.info(f"Loading initial knowledge base documents from {args.rag_initial_docs}")
            with open(args.rag_initial_docs, 'r') as f:
                initial_docs = json.load(f)
            rag_handler.add_documents(initial_docs)

    # --- 1. Load Data ---
    print("\n--- Loading Data ---")
    # Construct full paths based on project structure
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    metadata_dir = os.path.join(project_root, args.data_dir, 'processed')
    downloaded_image_dir = os.path.join(project_root, args.data_dir, 'downloaded_fakeddit_images')
    
    print(f"Metadata directory: {metadata_dir}")
    print(f"Image download directory: {downloaded_image_dir}")

    # Initialize dataset to return PIL images for model processors
    # Use balanced test data for better evaluation
    metadata_file_name = "test_balanced_pairs_clean.csv" if args.metadata_file == "test_pairs_clean.csv" else args.metadata_file
    
    dataset = FakedditDataset(
        metadata_dir=metadata_dir,
        metadata_file_name=metadata_file_name,
        downloaded_image_dir=downloaded_image_dir,
        transform=None, # Crucial: Get PIL images
        text_col=args.text_column,
        label_col=args.label_column,
        extra_metadata_fields=extra_metadata_fields
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
        "prompt_name": args.prompt_name if args.model_type in ['blip', 'llava'] else None, # Log the prompt name
        "use_few_shot": args.use_few_shot,
        "use_rag": args.use_rag,
        "rag_embedding_model": args.rag_embedding_model if args.use_rag else None,
        "rag_top_k": args.rag_top_k if args.use_rag else None,
        "rag_similarity_threshold": args.rag_similarity_threshold if args.use_rag else None,
    }

    logger.info(f"Starting pipeline with config: {json.dumps(experiment_config, indent=2)}")

    current_device = DEVICE
    # if args.model_type == 'llava':
    #     logger.warning("LLaVA model is large. Forcing CPU for LLaVA to potentially avoid MPS memory/performance issues. This will be slower.")
    #     current_device = "cpu"

    if args.model_type == 'clip': # Clip branch
        model, processor = load_clip(args.clip_model_name)
        
        # Get CLIP prompt template if specified
        clip_prompt_template = None
        if args.prompt_name:
            clip_prompt_template = prompt_library.get_template(args.prompt_name)
            if not clip_prompt_template:
                logger.warning(f"Prompt name '{args.prompt_name}' not found in prompt library. Using default text processing.")
        
        def clip_process_wrapper(batch, proc, dev):
            # Prepare metadata string for each sample
            metadata_strings = []
            for idx in range(len(batch['id'])):
                meta_parts = []
                for field in extra_metadata_fields:
                    if field in batch and len(batch[field]) > idx:
                        meta_parts.append(f"{field}: {batch[field][idx]}")
                metadata_strings.append("; ".join(meta_parts))
            
            # Process texts with RAG if enabled
            if args.use_rag and clip_prompt_template and clip_prompt_template.supports_rag:
                processed_texts = []
                for i, text in enumerate(batch['text']):
                    rag_query = f"{text} | {metadata_strings[i]}" if metadata_strings[i] else text
                    retrieved_docs = rag_handler.retrieve(rag_query)
                    rag_context = rag_handler.format_rag_prompt(text, retrieved_docs)
                    
                    # Format with RAG template
                    if args.prompt_name in ['clip_rag_metadata']:
                        processed_text = clip_prompt_template.rag_template.format(
                            text=text, 
                            rag_context=rag_context, 
                            metadata=metadata_strings[i]
                        )
                    else:
                        processed_text = clip_prompt_template.rag_template.format(
                            text=text, 
                            rag_context=rag_context
                        )
                    processed_texts.append(processed_text)
            elif clip_prompt_template:
                # Use regular template
                if args.prompt_name in ['clip_rag_metadata']:
                    processed_texts = [clip_prompt_template.template.format(text=t, metadata=metadata_strings[i]) 
                                     for i, t in enumerate(batch['text'])]
                else:
                    processed_texts = [clip_prompt_template.template.format(text=t) for t in batch['text']]
            else:
                # Default: use text as is
                processed_texts = batch['text']
            
            # Process with CLIP
            inputs = proc(images=batch['image'], text=processed_texts, return_tensors="pt", padding=True, truncation=True)
            # Move all inputs to the same device as the model
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            return inputs, None
        
        process_batch_fn = clip_process_wrapper
        logger.info(f"Using CLIP model: {args.clip_model_name} with prompt: '{args.prompt_name}' (RAG: {args.use_rag})")
    elif args.model_type == 'blip': # Blip branch
        model, processor = load_blip_conditional(args.blip_model_name)
        
        prompt_template = BLIP_PROMPTS.get(args.prompt_name)
        if not prompt_template and args.prompt_name != "dynamic_blip2_true_false_fewshot":
            logger.error(f"Prompt name '{args.prompt_name}' not found in BLIP_PROMPTS. Exiting.")
            return

        # Remove global formatting for few-shot prompt (fs_yesno_justification)
        if args.use_few_shot and args.prompt_name in ['fs_yesno_justification', 'fs_blip2_true_false', 'fs_blip2_accurate', 'fs_blip2_match', 'fs_blip2_true_false_balanced', 'fs_blip2_true_false_balanced_v2', 'fs_blip2_true_false_balanced_v3', 'fs_blip2_true_false_balanced_v4', 'fs_blip2_true_false_balanced_v5', 'blip2_fs_research_balanced', 'blip_enhanced_fewshot']:
            pass  # Do not format here; handled per-sample in batch wrapper
        elif args.use_few_shot and args.prompt_name != 'dynamic_blip2_true_false_fewshot':
            # Format the entire template with the few-shot examples first (for other few-shot prompts)
            prompt_template = prompt_template.format(
                real_example_text=few_shot_context["real_example_text"],
                real_explanation_blip=few_shot_context["real_explanation_blip"],
                fake_example_text=few_shot_context["fake_example_text"],
                fake_explanation_blip=few_shot_context["fake_explanation_blip"]
            )

        if args.prompt_name == 'dynamic_blip2_true_false_fewshot' and args.use_few_shot:
            # Prompt wird pro Sample dynamisch gebaut
            def blip_process_wrapper(batch, proc, dev):
                prompts = [build_blip2_true_false_fewshot_prompt(few_shot_examples_dynamic, text) for text in batch['text']]
                inputs = proc(images=batch['image'], text=prompts, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens_blip,
                    do_sample=True,
                    temperature=1.0,
                    top_k=10,
                    repetition_penalty=1.2
                )
                generated_texts = proc.batch_decode(generated_ids, skip_special_tokens=True)
                return generated_texts, None
            process_batch_fn = blip_process_wrapper
            logger.info("BLIP2 Few-Shot mit dynamischem 4-True-6-Fake Prompt aktiviert.")
        else:
            def blip_process_wrapper(batch, proc, dev):
                original_texts = batch['text']
                # Prepare metadata string for each sample
                metadata_strings = []
                for idx in range(len(batch['id'])):
                    meta_parts = []
                    for field in extra_metadata_fields:
                        if field in batch and len(batch[field]) > idx:
                            meta_parts.append(f"{field}: {batch[field][idx]}")
                    metadata_strings.append("; ".join(meta_parts))

                # --- Two-step approach: first yes/no, then justification ---
                if args.prompt_name in ['zs_yesno_justification', 'fs_yesno_justification']:
                    # Step 1: Yes/No prediction
                    yesno_prompts = []
                    for i, text in enumerate(original_texts):
                        if args.prompt_name == 'fs_yesno_justification':
                            base_template = "Example 1:\nText: 'A dog sitting on a couch'\nMetadata: location: living room; time: day\nQuestion: Does the text match the image and metadata?\nAnswer: Yes.\n\nExample 2:\nText: 'A cat playing the piano'\nMetadata: location: concert hall; time: night\nQuestion: Does the text match the image and metadata?\nAnswer: No.\n\nNow, answer the following:\nText: {text}\nMetadata: {metadata}\nQuestion: Does the text match the image and metadata?\nAnswer (Yes or No):"
                            yesno_prompt = base_template.format(text=text, metadata=metadata_strings[i])
                        else:
                            concise_template = "Text: {text}\nMetadata: {metadata}\nQuestion: Does the text match the image and metadata?\nAnswer (Yes or No):"
                            yesno_prompt = concise_template.format(text=text, metadata=metadata_strings[i])
                        yesno_prompts.append(yesno_prompt)
                    # Generate yes/no answers
                    inputs = proc(images=batch['image'], text=yesno_prompts, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=5,  # Short output for yes/no
                        do_sample=True,
                        temperature=0.7,
                        top_k=10,
                        repetition_penalty=1.2
                    )
                    yesno_answers = proc.batch_decode(generated_ids, skip_special_tokens=True)
                    # Step 2: Justification
                    justification_prompts = []
                    for i, text in enumerate(original_texts):
                        answer = yesno_answers[i].strip().split(". ")[0]  # Get 'Yes' or 'No'
                        if args.prompt_name == 'fs_yesno_justification':
                            base_template = "Example 1:\nText: 'A dog sitting on a couch'\nMetadata: location: living room; time: day\nQuestion: Does the text match the image and metadata?\nAnswer: Yes. The image shows a dog on a couch in a living room during the day.\n\nExample 2:\nText: 'A cat playing the piano'\nMetadata: location: concert hall; time: night\nQuestion: Does the text match the image and metadata?\nAnswer: No. The image does not show a cat or a piano.\n\nNow, answer the following:\nText: {text}\nMetadata: {metadata}\nQuestion: Does the text match the image and metadata?\nAnswer: {answer}. Justification:"
                            justification_prompt = base_template.format(text=text, metadata=metadata_strings[i], answer=answer)
                        else:
                            concise_template = "Text: {text}\nMetadata: {metadata}\nQuestion: Does the text match the image and metadata?\nAnswer: {answer}. Justification:"
                            justification_prompt = concise_template.format(text=text, metadata=metadata_strings[i], answer=answer)
                        justification_prompts.append(justification_prompt)
                    # Generate justifications
                    inputs = proc(images=batch['image'], text=justification_prompts, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=args.max_new_tokens_blip,
                        do_sample=True,
                        temperature=0.7,
                        top_k=10,
                        repetition_penalty=1.2
                    )
                    justifications = proc.batch_decode(generated_ids, skip_special_tokens=True)
                    # Combine yes/no and justification
                    generated_texts = [f"{yesno_answers[i].strip()}. {justifications[i].strip()}" for i in range(len(yesno_answers))]
                    return generated_texts, None
                # --- End two-step approach ---

                # --- Handle new few-shot BLIP2 prompts ---
                if args.use_few_shot and args.prompt_name in ['fs_blip2_true_false', 'fs_blip2_accurate', 'fs_blip2_match', 'blip2_fs_research_balanced', 'blip2_fs_extended', 'blip_enhanced_fewshot']:
                    prompts = []
                    for text in original_texts:
                        if args.prompt_name == 'fs_blip2_true_false':
                            prompt = prompt_template.format(text=text)
                        elif args.prompt_name == 'fs_blip2_accurate':
                            prompt = prompt_template.format(text=text)
                        elif args.prompt_name == 'fs_blip2_match':
                            prompt = prompt_template.format(text=text)
                        elif args.prompt_name == 'blip2_fs_research_balanced':
                            prompt = prompt_template.format(text=text)
                        elif args.prompt_name == 'blip_enhanced_fewshot':
                            prompt = prompt_template.format(text=text)
                        prompts.append(prompt)
                    
                    # Process the few-shot prompts
                    inputs = proc(images=batch['image'], text=prompts, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=args.max_new_tokens_blip,
                        do_sample=True,
                        temperature=1.0,
                        top_k=10,
                        repetition_penalty=1.2
                    )
                    generated_texts = proc.batch_decode(generated_ids, skip_special_tokens=True)
                    return generated_texts, None
                # --- NEW: Research-Optimized BLIP2 Processing ---
                if args.prompt_name.startswith('blip2_research_') or args.prompt_name.startswith('blip2_ultra_') or args.prompt_name.startswith('blip2_cot_'):
                    # Research-based prompts: Use optimized generation parameters
                    prompts = [prompt_template.format(text=t) for t in original_texts]
                    inputs = proc(images=batch['image'], text=prompts, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=args.max_new_tokens_blip,
                        do_sample=False,  # Deterministic for research prompts
                        temperature=0.0,  # No randomness for consistent results
                        top_k=1,  # Greedy decoding
                        repetition_penalty=1.0  # No repetition penalty needed
                    )
                    generated_texts = proc.batch_decode(generated_ids, skip_special_tokens=True)
                    return generated_texts, None
                # --- End Research-Optimized Processing ---
                # Original logic for other prompts
                elif args.use_rag:
                    prompts = []
                    for i, text in enumerate(original_texts):
                        rag_query = f"{text} | {metadata_strings[i]}" if metadata_strings[i] else text
                        retrieved_docs = rag_handler.retrieve(rag_query)
                        rag_prompt = rag_handler.format_rag_prompt(text, retrieved_docs)
                        if args.prompt_name in ['zs_metadata_check', 'blip_match_metadata', 'blip_match_metadata_few_shot']:
                            final_prompt = prompt_template.format(text=rag_prompt, metadata=metadata_strings[i])
                        else:
                            final_prompt = prompt_template.format(text=rag_prompt)
                        prompts.append(final_prompt)
                else:
                    if args.prompt_name in ['zs_metadata_check', 'blip_match_metadata', 'blip_match_metadata_few_shot']:
                        prompts = [prompt_template.format(text=t, metadata=metadata_strings[i]) for i, t in enumerate(original_texts)]
                    else:
                        prompts = [prompt_template.format(text=t) for t in original_texts]
                if args.prompt_name not in ['zs_yesno_justification', 'fs_yesno_justification', 'fs_blip2_true_false', 'fs_blip2_accurate', 'fs_blip2_match']:
                    inputs = proc(images=batch['image'], text=prompts, return_tensors="pt", padding=True, truncation=True)
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                    generated_ids = model.generate(
                        **inputs, 
                        max_new_tokens=args.max_new_tokens_blip,
                        do_sample=True,
                        temperature=1.0,
                        top_k=10,
                        repetition_penalty=1.2
                    )
                    generated_texts = proc.batch_decode(generated_ids, skip_special_tokens=True)
                    return generated_texts, None

            process_batch_fn = blip_process_wrapper
            logger.info(f"Using BLIP model: {args.blip_model_name} with prompt: '{args.prompt_name}' (Few-shot: {args.use_few_shot})")
    elif args.model_type == 'llava': # LLaVA branch
        model, processor = load_llava(args.llava_model_name)
        
        prompt_template = LLAVA_PROMPTS.get(args.prompt_name)
        if not prompt_template:
            logger.error(f"Prompt name '{args.prompt_name}' not found in LLAVA_PROMPTS. Exiting.")
            return

        if args.use_few_shot:
            prompt_template = prompt_template.format(**few_shot_context)

        def llava_process_wrapper(batch, proc, dev):
            # Prepare metadata string for each sample
            metadata_strings = []
            for idx in range(len(batch['id'])):
                meta_parts = []
                for field in extra_metadata_fields:
                    if field in batch and len(batch[field]) > idx:
                        meta_parts.append(f"{field}: {batch[field][idx]}")
                metadata_strings.append("; ".join(meta_parts))
            # Apply RAG if enabled
            if args.use_rag:
                rag_prompts = []
                for i, text in enumerate(batch['text']):
                    rag_query = f"{text} | {metadata_strings[i]}" if metadata_strings[i] else text
                    retrieved_docs = rag_handler.retrieve(rag_query)
                    rag_prompt = rag_handler.format_rag_prompt(text, retrieved_docs)
                    if args.prompt_name in ['zs_metadata_check', 'llava_match_metadata']:
                        final_prompt = prompt_template.format(text=rag_prompt, metadata=metadata_strings[i])
                    else:
                        final_prompt = prompt_template.format(text=rag_prompt)
                    rag_prompts.append(final_prompt)
                # Pass the prompt template string, not a list, to process_batch_for_llava
                inputs = process_batch_for_llava(batch, proc, dev, prompt_template, few_shot_images if args.use_few_shot else None, metadata_strings)
            else:
                # Pass the prompt template string, not a list, to process_batch_for_llava
                inputs = process_batch_for_llava(batch, proc, dev, prompt_template, few_shot_images if args.use_few_shot else None, metadata_strings)
            if inputs is None:
                return ["Error processing batch for LLaVA"] * len(batch['id']), None
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens_llava)
            generated_texts = proc.batch_decode(generated_ids, skip_special_tokens=True)
            return generated_texts, None

        process_batch_fn = llava_process_wrapper
        logger.info(f"Using LLaVA model: {args.llava_model_name} with prompt: '{args.prompt_name}' (Few-shot: {args.use_few_shot})")
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
                    inputs, _ = process_batch_fn(batch, processor, current_device) # Use current_device
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
                    # Add extra metadata fields if present in batch
                    for field in extra_metadata_fields:
                        if field in batch and len(batch[field]) > idx:
                            result_entry[field] = batch[field][idx]
                        elif hasattr(dataset, 'metadata') and field in dataset.metadata.columns:
                            # fallback: get from dataset metadata
                            val = dataset.metadata.loc[dataset.metadata['id'] == item_id, field]
                            result_entry[field] = val.values[0] if not val.empty else None
                    
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

    # --- For BLIP/LLaVA: parse generated_text and add predicted_label before saving CSV ---
    if args.model_type == 'blip' and 'generated_text' in results_df.columns:
        logger.info("BLIP run: Parsing generated_text to extract Yes/No predictions for CSV.")
        blip_parser = BLIPAnswerParser()
        parsed_results = [blip_parser.extract_prediction(gt) for gt in results_df['generated_text']]
        results_df['predicted_label'] = [r[0] for r in parsed_results]
        results_df['confidence'] = [r[1] for r in parsed_results]
        results_df['parsing_explanation'] = [r[2] for r in parsed_results]
    elif args.model_type == 'llava' and 'generated_text' in results_df.columns:
        logger.info("LLaVA run: Parsing generated_text to extract Yes/No predictions for CSV.")
        llava_parser = LLaVAAnswerParser()
        parsed_results = [llava_parser.extract_prediction(gt) for gt in results_df['generated_text']]
        results_df['predicted_label'] = [r[0] for r in parsed_results]
        results_df['confidence'] = [r[1] for r in parsed_results]
        results_df['parsing_explanation'] = [r[2] for r in parsed_results]

    # Save all raw outputs (now always includes predicted_label for BLIP)
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
        # For BLIP, parse generated_text to extract Yes/No predictions
        logger.info("BLIP run: Parsing generated_text to extract Yes/No predictions.")
        
        # Initialize BLIP parser
        blip_parser = BLIPAnswerParser()
        
        # Parse all generated texts
        parsed_results = []
        for idx, row in results_df.iterrows():
            generated_text = row.get('generated_text', '')
            predicted_label, confidence, explanation = blip_parser.extract_prediction(generated_text)
            parsed_results.append({
                'predicted_label': predicted_label,
                'confidence': confidence,
                'parsing_explanation': explanation
            })
        
        # Add parsed results to dataframe
        results_df['predicted_label'] = [r['predicted_label'] for r in parsed_results]
        results_df['confidence'] = [r['confidence'] for r in parsed_results]
        results_df['parsing_explanation'] = [r['parsing_explanation'] for r in parsed_results]
        
        # Filter out uncertain predictions for evaluation
        certain_predictions = results_df[results_df['predicted_label'].notna()]
        uncertain_count = len(results_df) - len(certain_predictions)
        
        if len(certain_predictions) > 0:
            logger.info(f"BLIP: {len(certain_predictions)} certain predictions, {uncertain_count} uncertain predictions")
            pred_labels_col = 'predicted_label'
        else:
            logger.warning("BLIP: No certain predictions found. Skipping quantitative evaluation.")
            pred_labels_col = None
        
        evaluation_report_path = os.path.join(args.output_dir, "reports", f"{args.model_type}_{args.blip_model_name.replace('/', '_')}_evaluation_report.txt")
        figures_dir = os.path.join(args.output_dir, "figures", f"{args.model_type}_{args.blip_model_name.replace('/', '_')}")
    elif args.model_type == 'bert' and 'predicted_label' in results_df.columns: # Bert directly outputs class
        pred_labels_col = 'predicted_label'
        logger.info("BERT run: Using 'predicted_label' from model output.")
        evaluation_report_path = os.path.join(args.output_dir, "reports", f"{args.model_type}_{args.bert_model_name.replace('/', '_')}_evaluation_report.txt")
        figures_dir = os.path.join(args.output_dir, "figures", f"{args.model_type}_{args.bert_model_name.replace('/', '_')}")
    elif args.model_type == 'llava' and 'generated_text' in results_df.columns: # Evaluation for LLaVA
        # For LLaVA, parse generated_text to extract Yes/No predictions
        logger.info("LLaVA run: Parsing generated_text to extract Yes/No predictions.")
        
        # Initialize LLaVA parser
        llava_parser = LLaVAAnswerParser()
        
        # Parse all generated texts
        parsed_results = []
        for idx, row in results_df.iterrows():
            generated_text = row.get('generated_text', '')
            predicted_label, confidence, explanation = llava_parser.extract_prediction(generated_text)
            parsed_results.append({
                'predicted_label': predicted_label,
                'confidence': confidence,
                'parsing_explanation': explanation
            })
        
        # Add parsed results to dataframe
        results_df['predicted_label'] = [r['predicted_label'] for r in parsed_results]
        results_df['confidence'] = [r['confidence'] for r in parsed_results]
        results_df['parsing_explanation'] = [r['parsing_explanation'] for r in parsed_results]
        
        # Filter out uncertain predictions for evaluation
        certain_predictions = results_df[results_df['predicted_label'].notna()]
        uncertain_count = len(results_df) - len(certain_predictions)
        
        if len(certain_predictions) > 0:
            logger.info(f"LLaVA: {len(certain_predictions)} certain predictions, {uncertain_count} uncertain predictions")
            pred_labels_col = 'predicted_label'
        else:
            logger.warning("LLaVA: No certain predictions found. Skipping quantitative evaluation.")
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
        # Metrics summary table is already saved by evaluate_model_outputs
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
        # Compute and save qualitative stats and summary table
        qualitative_stats = compute_qualitative_stats(results_df, 'generated_text')
        metrics = {}  # No quantitative metrics without predicted labels
        save_metrics_table(metrics, qualitative_stats, figures_dir)
    else:
        logger.warning("Evaluation skipped: True labels or predicted labels column not found or not applicable.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run multimodal fact-checking pipeline.")
    
    # --- General arguments ---
    parser.add_argument('--data_dir', type=str, default='data', help='Directory for data, relative to project root.')
    parser.add_argument('--metadata_file', type=str, default='train_correct_pairs.csv', help='Metadata CSV file name in data_dir/processed.')
    parser.add_argument('--output_dir', type=str, default='results', help='Base directory to save experiment results.')
    parser.add_argument('--experiment_name', type=str, default=None, help='A unique name for the experiment.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for processing.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader.')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of samples to process from the dataset. -1 for all.')
    parser.add_argument('--model_type', type=str, choices=['clip', 'blip', 'bert', 'llava'], required=True, help='Type of model to use.')
    parser.add_argument('--text_column', type=str, default='clean_title', help='Column name for text in the metadata file.')
    parser.add_argument('--label_column', type=str, default='2_way_label', help='Column name for labels in the metadata file.')
    
    # --- Model-specific Arguments ---
    parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-base-patch32', help='Name of the CLIP model to use.')
    parser.add_argument('--blip_model_name', type=str, default='Salesforce/blip-image-captioning-large', help='Name of the BLIP model to use.')
    parser.add_argument('--bert_model_name', type=str, default='bert-base-uncased', help='Name of the BERT model to use.')
    parser.add_argument('--llava_model_name', type=str, default='llava-hf/llava-1.5-7b-hf', help='Name of the LLaVA model to use.')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels for BERT classification.')

    # --- Prompting and Generation Arguments ---
    parser.add_argument('--prompt_name', type=str, default='default', help='Name of the prompt from prompts.py to use for BLIP or LLaVA.')
    parser.add_argument('--use_few_shot', action='store_true', help='If set, enables few-shot prompting using examples from prompts.py.')
    parser.add_argument('--max_new_tokens_blip', type=int, default=75, help='Max new tokens for BLIP generation.')
    parser.add_argument('--max_new_tokens_llava', type=int, default=100, help='Max new tokens for LLaVA generation.')

    # Add RAG-specific arguments
    parser.add_argument("--use_rag", action="store_true", help="Enable RAG for enhanced fact-checking")
    parser.add_argument("--rag_embedding_model", type=str, default="all-MiniLM-L6-v2",
                      help="Sentence transformer model for RAG embeddings")
    parser.add_argument("--rag_top_k", type=int, default=3,
                      help="Number of documents to retrieve for RAG")
    parser.add_argument("--rag_similarity_threshold", type=float, default=0.7,
                      help="Minimum similarity score for RAG document retrieval")
    parser.add_argument("--rag_knowledge_base_path", type=str, default="data/knowledge_base",
                      help="Path to RAG knowledge base")
    parser.add_argument("--rag_initial_docs", type=str,
                      help="Path to initial knowledge base documents JSON file")

    args = parser.parse_args()
    main(args)
