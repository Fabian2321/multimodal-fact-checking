from transformers import (CLIPModel, CLIPProcessor, 
                        BlipForConditionalGeneration, BlipProcessor, BlipForImageTextRetrieval, # Example BLIP models
                        AutoImageProcessor, AutoTokenizer, AutoModelForSequenceClassification,
                        Blip2Processor, Blip2ForConditionalGeneration,
                        AutoProcessor, AutoModelForCausalLM, BertTokenizerFast, BertForSequenceClassification,
                        LlavaForConditionalGeneration, # Use this for LLaVA 1.5
                        # LlavaNextProcessor, LlavaNextForConditionalGeneration # Keep commented for now
                        ) # Added BLIP-2 classes and LLaVA
import torch
from PIL import Image
import logging

# Setup logger
logger = logging.getLogger(__name__)
# Basic config if not configured by the main script, or use setup_logger from utils
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- CLIP Model and Processor --- 
def load_clip(model_name="openai/clip-vit-base-patch32"):
    """Loads CLIP model and its processor."""
    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        logger.info(f"CLIP model and processor '{model_name}' loaded successfully.")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading CLIP model '{model_name}': {e}")
        return None, None

def process_batch_for_clip(batch, clip_processor, device="cpu"):
    """Processes a batch of data (raw text, PIL images or image tensors) using CLIPProcessor."""
    if not clip_processor:
        raise ValueError("CLIPProcessor not provided.")
    
    raw_texts = batch['text'] # List of raw text strings
    images = batch['image']    # List/Tensor of PIL Images or pre-processed image tensors

    # The CLIPProcessor expects a list of PIL Images and a list of texts.
    # If images are already tensors, we might need to handle them differently or ensure they are PIL.
    # For now, assuming `images` from DataLoader are tensors that might need to be converted back or handled.
    # Or, that the DataLoader passes PIL images if no transform is applied in Dataset.
    # Let's assume for now the FakedditDataset gives PIL images if we pass transform=None
    # or we adjust FakedditDataset to also provide raw PIL images if needed by processor.
    
    # A robust way: check if images are PIL or Tensors
    # For simplicity, this example assumes CLIPProcessor can handle raw text and PIL images directly.
    # If FakedditDataset provides transformed tensors, this part needs adjustment or the dataset needs to provide PILs.
    
    # Let's refine this: CLIPProcessor expects PIL images. 
    # Our current FakedditDataset applies DEFAULT_IMAGE_TRANSFORMS which results in tensors.
    # This means `process_batch_for_clip` might need to work with raw image paths/urls and load PILs itself,
    # OR FakedditDataset needs to be more flexible in what it returns (e.g. also return PIL image).
    
    # For now, let's assume the `batch['image']` are PIL Images. This means FakedditDataset
    # should have its `transform` parameter set to `None` when used with this CLIP path,
    # and the CLIP processor's own image transformations will be used.
    
    try:
        # `images` should be a list of PIL.Image.Image objects
        # `text` should be a list of strings
        inputs = clip_processor(text=raw_texts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()} # Move to device
        return inputs
    except Exception as e:
        logger.error(f"Error processing batch for CLIP: {e}")
        # This might happen if `images` are not PIL images as expected by CLIPProcessor
        logger.error("Ensure that the 'image' field in the batch contains PIL Images for CLIPProcessor.")
        raise

# --- BLIP Model and Processor --- 
# Example: BLIP for VQA or Image Captioning
def load_blip_conditional(model_name="Salesforce/blip-image-captioning-base"):
    """Loads a BLIP or BLIP-2 model for conditional generation and its processor."""
    try:
        if "blip2" in model_name.lower():
            logger.info(f"Detected BLIP-2 model name: '{model_name}'. Attempting to load with Blip2 classes.")
            # For BLIP-2, especially larger models, using float16 and device_map can be crucial.
            # Ensure 'accelerate' package is installed for device_map="auto".
            try:
                processor = Blip2Processor.from_pretrained(model_name)
                model = Blip2ForConditionalGeneration.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16, # Use float16 for memory efficiency
                    device_map="auto" # Automatically maps model to available devices (GPU/CPU)
                )
                logger.info(f"BLIP-2 model and processor '{model_name}' loaded successfully with device_map='auto' and float16.")
            except ImportError as ie:
                if "accelerate" in str(ie).lower():
                    logger.warning(f"'{model_name}' requires 'accelerate' for device_map. Trying without device_map.")
                    processor = Blip2Processor.from_pretrained(model_name)
                    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
                    logger.info(f"BLIP-2 model and processor '{model_name}' loaded successfully with float16 (no device_map).")
                else:
                    raise # Re-raise other import errors
            except Exception as e_blip2:
                logger.error(f"Error loading BLIP-2 model '{model_name}' even with Blip2 classes: {e_blip2}")
                logger.warning("Falling back to try with original BlipForConditionalGeneration (less likely to work for BLIP-2 names).")
                # Fallback to original Blip classes if Blip2 specific loading failed for some reason (e.g. misidentified name)
                model = BlipForConditionalGeneration.from_pretrained(model_name) 
                processor = BlipProcessor.from_pretrained(model_name)
                logger.info(f"BLIP model (using BlipForConditionalGeneration) and processor '{model_name}' loaded after BLIP-2 attempt failed.")
        else:
            logger.info(f"Detected standard BLIP model name: '{model_name}'. Loading with BlipForConditionalGeneration.")
            model = BlipForConditionalGeneration.from_pretrained(model_name)
            processor = BlipProcessor.from_pretrained(model_name)
            logger.info(f"BLIP model and processor '{model_name}' loaded successfully.")
        return model, processor
    except Exception as e:
        logger.error(f"Overall error loading BLIP/BLIP-2 model '{model_name}': {e}")
        return None, None

def process_batch_for_blip_conditional(batch, blip_processor, task="captioning", device="cpu", target_texts=None, max_length=32):
    """Processes a batch of data using BlipProcessor for tasks like captioning or VQA."""
    if not blip_processor:
        raise ValueError("BlipProcessor not provided.")

    raw_texts = batch['text'] # List of raw text strings (might be questions for VQA, or empty for captioning)
    images = batch['image']    # List/Tensor of PIL Images or pre-processed image tensors

    # BlipProcessor also typically expects PIL images.
    # Similar to CLIP, ensure FakedditDataset provides PIL images if transform=None.
    try:
        if task == "captioning":
            # For captioning, text input to processor is usually None or empty
            inputs = blip_processor(images=images, text=None, return_tensors="pt") 
        elif task == "vqa" or task == "classification_prompted": # For VQA or prompted classification
            if not raw_texts:
                raise ValueError("Text prompts (raw_texts) are required for VQA/prompted classification with BLIP.")
            # The text here would be the question or the prompt
            inputs = blip_processor(images=images, text=raw_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        elif task == "retrieval_preprocessing": # If using a retrieval model that needs text for both image and text塔
            if not raw_texts:
                raise ValueError("Text (raw_texts) are required for retrieval preprocessing with BLIP.")
            inputs = blip_processor(images=images, text=raw_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        else:
            raise ValueError(f"Unsupported BLIP task: {task}")
        
        inputs = {k: v.to(device) for k, v in inputs.items()} # Move to device
        
        # For some BLIP tasks (like training VQA or retrieval), you might need to include labels (tokenized target texts)
        if target_texts: # list of target strings
            labels = blip_processor(text=target_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)["input_ids"]
            inputs["labels"] = labels.to(device)
            
        return inputs
    except Exception as e:
        logger.error(f"Error processing batch for BLIP conditional task '{task}': {e}")
        logger.error("Ensure that the 'image' field in the batch contains PIL Images for BlipProcessor.")
        raise

# --- BERT Model and Tokenizer for Text Classification ---
def load_bert_classifier(model_name="bert-base-uncased", num_labels=2):
    """Loads a BERT model for sequence classification and its tokenizer."""
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"BERT classifier model and tokenizer '{model_name}' loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading BERT classifier model '{model_name}': {e}")
        return None, None

def process_batch_for_bert_classifier(batch, bert_tokenizer, device="cpu", max_length=128):
    """Processes a batch of text data using BertTokenizer for classification."""
    if not bert_tokenizer:
        raise ValueError("BertTokenizer not provided.")
    
    raw_texts = batch['text'] # List of raw text strings

    if not raw_texts:
        logger.warning("Warning: Empty text batch provided for BERT processing.")
        return None
        
    try:
        inputs = bert_tokenizer(raw_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()} # Move to device
        return inputs
    except Exception as e:
        logger.error(f"Error processing batch for BERT classifier: {e}")
        raise

# --- LLaVA Functions (Revised for LLaVA 1.5) ---
def load_llava(model_name_or_path, **kwargs):
    """
    Loads a LLaVA 1.5 model and processor.
    Uses LlavaForConditionalGeneration and AutoProcessor.
    """
    logger.info(f"Loading LLaVA 1.5 model and processor from: {model_name_or_path}")
    try:
        # AutoProcessor should correctly load the LlavaProcessor for LLaVA 1.5 models
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )
        logger.info(f"LLaVA 1.5 model {model_name_or_path} loaded successfully.")
        return model, processor
    except Exception as e:
        logger.error(f"Error loading LLaVA 1.5 model {model_name_or_path}: {e}")
        return None, None

def process_batch_for_llava(batch, processor, device, llava_prompt_template):
    """
    Prepares a batch for LLaVA model inference.
    The llava_prompt_template should include '{text}' for the text caption.
    The processor for LLaVA 1.5 typically handles the <image> token insertion internally or requires a specific format.
    Example template: "USER: <image>\nGiven the image and the caption '{text}', is this post misleading? Why or why not?\nASSISTANT:"
    This template structure with USER: <image>... ASSISTANT: is common for LLaVA.
    """
    texts = batch['text']
    images = batch['image'] # List of PIL Images

    # Construct prompts. The <image> token is essential and its placement is dictated by the model's training.
    # For llava-hf/llava-1.5-7b-hf, the prompt should typically start with "USER: <image>\n" followed by the question.
    prompts = [llava_prompt_template.format(text=t) for t in texts]

    try:
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        return inputs
    except Exception as e:
        logger.error(f"Error processing batch for LLaVA: {e}")
        return None

# --- Add other BLIP model types if needed (e.g., BlipForImageTextRetrieval) ---
# def load_blip_retrieval(model_name="Salesforce/blip-itm-base-coco"):
#     try:
#         model = BlipForImageTextRetrieval.from_pretrained(model_name)
#         # For retrieval, sometimes image_processor and tokenizer are loaded separately or via Auto* classes
#         image_processor = AutoImageProcessor.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         # Or a BlipProcessor if available and suitable
#         # processor = BlipProcessor.from_pretrained(model_name) 
#         logger.info(f"BLIP retrieval model '{model_name}' loaded.")
#         return model, image_processor, tokenizer # or processor
#     except Exception as e:
#         logger.error(f"Error loading BLIP retrieval model '{model_name}': {e}")
#         return None, None, None


# Example Usage (can be expanded or moved to pipeline.py)
if __name__ == '__main__':
    # This section is for basic testing of model_handler functions.
    # Assumes FakedditDataset is modified to provide PIL images when its `transform` is None.

    logger.info("--- Testing Model Handler Functions ---")

    # --- Test CLIP --- 
    logger.info("\n--- Testing CLIP Loading & Processing ---")
    clip_model, clip_processor = load_clip()
    if clip_model and clip_processor:
        # Create a dummy batch similar to what FakedditDataset (with transform=None) would output
        # Requires PIL to be installed (pip install Pillow)
        try:
            dummy_pil_image = Image.new('RGB', (224, 224), color = 'red')
            dummy_batch_clip = {
                'id': ['id1', 'id2'],
                'image': [dummy_pil_image, dummy_pil_image], # List of PIL Images
                'text': ["A photo of a cat", "A photo of a dog"],
                'label': [0, 1]
            }
            logger.info("Dummy batch for CLIP created.")
            clip_inputs = process_batch_for_clip(dummy_batch_clip, clip_processor)
            logger.info("CLIP inputs processed successfully:")
            for key, val in clip_inputs.items():
                logger.info(f"  {key}: shape {val.shape}")
            
            # Test forward pass with CLIP model (optional here, more for pipeline.py)
            # with torch.no_grad():
            #     outputs = clip_model(**clip_inputs)
            #     logger.info(f"CLIP image_embeds shape: {outputs.image_embeds.shape}")
            #     logger.info(f"CLIP text_embeds shape: {outputs.text_embeds.shape}")
            #     logger.info(f"CLIP logits_per_image shape: {outputs.logits_per_image.shape}")

        except Exception as e:
            logger.error(f"Error in CLIP test section: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error("CLIP model or processor failed to load.")

    # --- Test BLIP --- 
    logger.info("\n--- Testing BLIP Loading & Processing (Captioning Example) ---")
    # Using a smaller BLIP model for faster download if testing for the first time
    blip_model, blip_processor = load_blip_conditional(model_name="Salesforce/blip-image-captioning-base") # or Salesforce/blip-vqa-base
    if blip_model and blip_processor:
        try:
            dummy_pil_image_blip = Image.new('RGB', (224, 224), color = 'blue')
            dummy_batch_blip_caption = {
                'id': ['id3'],
                'image': [dummy_pil_image_blip], # List of PIL Images
                'text': [None], # For captioning, text can be None or empty
                'label': [0]
            }
            logger.info("Dummy batch for BLIP (captioning) created.")
            blip_inputs_caption = process_batch_for_blip_conditional(dummy_batch_blip_caption, blip_processor, task="captioning")
            logger.info("BLIP (captioning) inputs processed successfully:")
            for key, val in blip_inputs_caption.items():
                logger.info(f"  {key}: shape {val.shape}")

            # Test VQA-like processing for BLIP
            dummy_batch_blip_vqa = {
                'id': ['id4'],
                'image': [dummy_pil_image_blip],
                'text': ["What color is the image?"], # Question for VQA
                'label': [0]
            }
            logger.info("\nDummy batch for BLIP (VQA-like) created.")
            blip_inputs_vqa = process_batch_for_blip_conditional(dummy_batch_blip_vqa, blip_processor, task="vqa", max_length=20)
            logger.info("BLIP (VQA-like) inputs processed successfully:")
            for key, val in blip_inputs_vqa.items():
                logger.info(f"  {key}: shape {val.shape}")
            
            # Example of generating captions (optional here, more for pipeline.py)
            # if blip_inputs_caption:
            #     logger.info("\nGenerating caption with BLIP...")
            #     with torch.no_grad():
            #         generated_ids = blip_model.generate(**blip_inputs_caption, max_length=20)
            #         generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            #         logger.info(f"  Generated caption: {generated_text}")

        except Exception as e:
            logger.error(f"Error in BLIP test section: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error("BLIP model or processor failed to load.")

    # --- Test BERT Classifier ---
    logger.info("\n--- Testing BERT Classifier Loading & Processing ---")
    bert_model, bert_tokenizer = load_bert_classifier(num_labels=2) # Assuming 2 labels for fake/real
    if bert_model and bert_tokenizer:
        dummy_batch_bert = {
            'id': ['id_text1', 'id_text2'],
            'text': ["This is a factual statement.", "This is a misleading statement."],
            # 'image' and 'label' fields are not strictly needed for this specific processing function test
        }
        logger.info("Dummy batch for BERT created.")
        try:
            bert_inputs = process_batch_for_bert_classifier(dummy_batch_bert, bert_tokenizer)
            logger.info("BERT inputs processed successfully:")
            for key, val in bert_inputs.items():
                logger.info(f"  {key}: shape {val.shape}")
            
            # Optional: Test forward pass (more for pipeline)
            # with torch.no_grad():
            #     outputs = bert_model(**bert_inputs)
            #     logger.info(f"BERT output logits shape: {outputs.logits.shape}")
        except Exception as e:
            logger.error(f"Error in BERT test section: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error("BERT classifier model or tokenizer failed to load.")

    logger.info("\nModel handler testing finished.")
