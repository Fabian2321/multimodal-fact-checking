import pandas as pd
import os
import logging # Import logging

# Setup logger for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Configure logger for this module if not already configured by a central setup
    # This basic config will print to console.
    # For more advanced logging (e.g., to file, different formats), a central setup (e.g., in utils.py and called in main scripts) is better.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Should point to mllm/
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
IMAGE_DOWNLOAD_DIR = os.path.join(BASE_DIR, 'data', 'downloaded_fakeddit_images') # Ensure this matches FakedditDataset

# Ensure processed directory exists
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Configuration ---
# These are the default column names used in the FakedditDataset loader.
# Adjust if your CSVs use different names.
IMAGE_ID_COL = 'id'
TEXT_COL = 'clean_title'
LABEL_COL = '2_way_label' # This is key for splitting
IMAGE_URL_COL = 'image_url'
HAS_IMAGE_COL = 'hasImage'

# Define the Fakeddit original splits
splits = {
    'train': 'multimodal_train.csv',
    'validate': 'multimodal_validate.csv',
    'test': 'multimodal_test_public.csv'
}

# --- Main Processing Logic ---
def process_split(split_name, input_filename, output_dir):
    """
    Processes a single Fakeddit split (train, validate, or test).
    Filters data, separates correct/incorrect pairs, and saves them.
    """
    input_filepath = os.path.join(RAW_DATA_DIR, input_filename)
    logger.info(f"Processing {split_name} split from: {input_filepath}")

    try:
        df = pd.read_csv(input_filepath, sep='\t', engine='python', on_bad_lines='warn')
        logger.info(f"  Successfully read {len(df)} rows from {input_filename}")
    except FileNotFoundError:
        logger.error(f"  File not found: {input_filepath}. Skipping this split.")
        return
    except Exception as e:
        logger.error(f"  Could not read {input_filepath}: {e}. Skipping this split.")
        return

    # Ensure all required columns are present
    required_cols = [IMAGE_ID_COL, TEXT_COL, LABEL_COL, IMAGE_URL_COL, HAS_IMAGE_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"  Missing required columns in {input_filename}: {missing_cols}. Skipping this split.")
        return

    # 1. Filter for entries that have images
    # Convert has_image_col to boolean if it's an object type (as in FakedditDataset)
    if df[HAS_IMAGE_COL].dtype == 'object':
        df[HAS_IMAGE_COL] = df[HAS_IMAGE_COL].apply(
            lambda x: str(x).lower() == 'true' if isinstance(x, str) else bool(x)
        )
    df_with_images = df[df[HAS_IMAGE_COL] == True].copy() # Use .copy() to avoid SettingWithCopyWarning
    logger.info(f"  Found {len(df_with_images)} entries with images.")

    if df_with_images.empty:
        logger.info(f"  No entries with images found in {input_filename}. Skipping.")
        return
        
    # Ensure labels are integers (0 or 1)
    try:
        df_with_images[LABEL_COL] = pd.to_numeric(df_with_images[LABEL_COL], errors='coerce')
        df_with_images.dropna(subset=[LABEL_COL], inplace=True) # Remove rows where label couldn't be coerced
        df_with_images[LABEL_COL] = df_with_images[LABEL_COL].astype(int)
    except Exception as e:
        logger.error(f"  Could not process label column '{LABEL_COL}': {e}. Skipping this split.")
        return


    # 2. Separate into correct and incorrect pairs
    # Assuming 0 means correct/real pair, and 1 means incorrect/fake/mismatched pair
    correct_pairs_df = df_with_images[df_with_images[LABEL_COL] == 0].copy()
    incorrect_pairs_df = df_with_images[df_with_images[LABEL_COL] == 1].copy()

    logger.info(f"  Found {len(correct_pairs_df)} correct pairs (label 0).")
    logger.info(f"  Found {len(incorrect_pairs_df)} incorrect pairs (label 1).")

    # 3. Save the processed DataFrames
    correct_pairs_filename = os.path.join(output_dir, f"{split_name}_correct_pairs.csv")
    incorrect_pairs_filename = os.path.join(output_dir, f"{split_name}_incorrect_pairs.csv")

    try:
        correct_pairs_df.to_csv(correct_pairs_filename, index=False)
        logger.info(f"  Saved correct pairs to: {correct_pairs_filename}")
    except Exception as e:
        logger.error(f"  Could not save {correct_pairs_filename}: {e}")
        
    try:
        incorrect_pairs_df.to_csv(incorrect_pairs_filename, index=False)
        logger.info(f"  Saved incorrect pairs to: {incorrect_pairs_filename}")
    except Exception as e:
        logger.error(f"  Could not save {incorrect_pairs_filename}: {e}")

if __name__ == "__main__":
    # For direct script execution, keep print statements or set up a simple console handler for the logger
    print("Starting Fakeddit data preprocessing...")
    for split_key, filename in splits.items():
        process_split(split_key, filename, PROCESSED_DATA_DIR)
    print("Fakeddit data preprocessing finished.")
    print(f"Processed files are located in: {PROCESSED_DATA_DIR}")
    print(f"Ensure your FakedditDataset loader points to '{IMAGE_DOWNLOAD_DIR}' for image downloads if you use these processed CSVs.") 