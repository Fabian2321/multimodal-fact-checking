#!/usr/bin/env python3
"""
Create Clean Balanced Dataset Script

This script creates a perfectly clean and balanced dataset with exactly:
- 50 real samples (label = 0)
- 50 fake samples (label = 1)
- No corrupted or malformed rows
- Valid required fields

The script handles CSV formatting issues, corrupted rows, and ensures
perfect balance for reliable experimental results.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_row(row):
    """
    Validate a single row to ensure it has clean, valid data.
    
    Args:
        row: pandas Series representing a row
        
    Returns:
        bool: True if row is valid, False otherwise
    """
    try:
        # Check if 2_way_label exists and is exactly 0 or 1
        if '2_way_label' not in row.index:
            return False
            
        label = row['2_way_label']
        
        # Convert to string and clean
        label_str = str(label).strip()
        
        # Must be exactly '0' or '1'
        if label_str not in ['0', '1']:
            return False
            
        # Required fields validation
        required_fields = ['author', 'clean_title', 'title', 'image_url', 'subreddit']
        
        for field in required_fields:
            if field not in row.index:
                return False
                
            value = row[field]
            if pd.isna(value) or str(value).strip() == '':
                return False
                
        # Image URL validation - must be a valid URL
        image_url = str(row['image_url']).strip()
        if not image_url.startswith(('http://', 'https://')):
            return False
            
        # Title validation - must have reasonable length
        title = str(row['title']).strip()
        if len(title) < 5 or len(title) > 500:
            return False
            
        # Author validation - must not be empty
        author = str(row['author']).strip()
        if len(author) == 0:
            return False
            
        return True
        
    except Exception as e:
        logger.debug(f"Row validation error: {e}")
        return False

def load_and_clean_csv(file_path, label_value, max_samples=None):
    """
    Load and clean CSV file, extracting only valid rows with specified label.
    
    Args:
        file_path: Path to CSV file
        label_value: Expected label value (0 or 1)
        max_samples: Maximum number of samples to extract (None for all)
        
    Returns:
        list: List of valid rows as dictionaries
    """
    logger.info(f"Loading {file_path} for label {label_value}")
    
    valid_rows = []
    
    try:
        # Read CSV in chunks to handle large files
        chunk_size = 10000
        
        for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
            logger.info(f"Processing chunk with {len(chunk)} rows")
            
            for idx, row in chunk.iterrows():
                # Validate the row
                if not validate_row(row):
                    continue
                    
                # Check if label matches expected value
                if str(row['2_way_label']).strip() == str(label_value):
                    # Convert row to dict and add to valid rows
                    row_dict = row.to_dict()
                    valid_rows.append(row_dict)
                    
                    # Stop if we have enough samples
                    if max_samples and len(valid_rows) >= max_samples:
                        logger.info(f"Reached maximum samples ({max_samples}) for label {label_value}")
                        break
                        
            # Stop if we have enough samples
            if max_samples and len(valid_rows) >= max_samples:
                break
                
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []
        
    logger.info(f"Found {len(valid_rows)} valid rows with label {label_value}")
    return valid_rows

def create_balanced_dataset(correct_file, incorrect_file, output_file, samples_per_class=50):
    """
    Create a perfectly balanced dataset with exactly samples_per_class of each label.
    
    Args:
        correct_file: Path to correct pairs CSV (label 0)
        incorrect_file: Path to incorrect pairs CSV (label 1)
        output_file: Path to output balanced CSV
        samples_per_class: Number of samples per class (default 50)
    """
    logger.info(f"Creating balanced dataset with {samples_per_class} samples per class")
    
    # Load valid samples from both files
    # Get more samples than needed to account for potential duplicates
    real_samples = load_and_clean_csv(correct_file, 0, max_samples=samples_per_class * 3)
    fake_samples = load_and_clean_csv(incorrect_file, 1, max_samples=samples_per_class * 3)
    
    logger.info(f"Loaded {len(real_samples)} real samples and {len(fake_samples)} fake samples")
    
    # Check if we have enough samples
    if len(real_samples) < samples_per_class:
        logger.error(f"Not enough real samples. Need {samples_per_class}, have {len(real_samples)}")
        return False
        
    if len(fake_samples) < samples_per_class:
        logger.error(f"Not enough fake samples. Need {samples_per_class}, have {len(fake_samples)}")
        return False
    
    # Remove duplicates based on image_url and title
    def remove_duplicates(samples):
        seen = set()
        unique_samples = []
        
        for sample in samples:
            key = (str(sample.get('image_url', '')), str(sample.get('title', '')))
            if key not in seen:
                seen.add(key)
                unique_samples.append(sample)
                
        return unique_samples
    
    real_samples = remove_duplicates(real_samples)
    fake_samples = remove_duplicates(fake_samples)
    
    logger.info(f"After deduplication: {len(real_samples)} real, {len(fake_samples)} fake")
    
    # Check again if we have enough samples
    if len(real_samples) < samples_per_class:
        logger.error(f"Not enough unique real samples. Need {samples_per_class}, have {len(real_samples)}")
        return False
        
    if len(fake_samples) < samples_per_class:
        logger.error(f"Not enough unique fake samples. Need {samples_per_class}, have {len(fake_samples)}")
        return False
    
    # Take exactly samples_per_class from each
    real_samples = real_samples[:samples_per_class]
    fake_samples = fake_samples[:samples_per_class]
    
    # Combine and shuffle
    all_samples = real_samples + fake_samples
    np.random.shuffle(all_samples)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_samples)
    
    # Verify the final dataset
    label_counts = df['2_way_label'].value_counts()
    logger.info(f"Final dataset label distribution: {label_counts.to_dict()}")
    
    # Additional validation
    if len(df) != samples_per_class * 2:
        logger.error(f"Final dataset size incorrect. Expected {samples_per_class * 2}, got {len(df)}")
        return False
        
    if label_counts.get(0, 0) != samples_per_class or label_counts.get(1, 0) != samples_per_class:
        logger.error("Final dataset is not perfectly balanced")
        return False
    
    # Save the balanced dataset
    df.to_csv(output_file, index=False)
    logger.info(f"Saved balanced dataset to {output_file}")
    
    # Final verification
    logger.info("=== FINAL VERIFICATION ===")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Real samples (0): {label_counts.get(0, 0)}")
    logger.info(f"Fake samples (1): {label_counts.get(1, 0)}")
    logger.info(f"Perfect balance: {label_counts.get(0, 0) == label_counts.get(1, 0) == samples_per_class}")
    
    return True

def main():
    """Main function to create the balanced dataset."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # File paths
    data_dir = Path("data/processed")
    correct_file = data_dir / "test_correct_pairs.csv"
    incorrect_file = data_dir / "test_incorrect_pairs.csv"
    output_file = data_dir / "test_balanced_pairs_clean.csv"
    
    # Check if input files exist
    if not correct_file.exists():
        logger.error(f"Correct pairs file not found: {correct_file}")
        sys.exit(1)
        
    if not incorrect_file.exists():
        logger.error(f"Incorrect pairs file not found: {incorrect_file}")
        sys.exit(1)
    
    # Create balanced dataset
    success = create_balanced_dataset(
        correct_file=correct_file,
        incorrect_file=incorrect_file,
        output_file=output_file,
        samples_per_class=50
    )
    
    if success:
        logger.info("✅ Successfully created clean balanced dataset!")
        logger.info(f"Output file: {output_file}")
        
        # Show sample of the output
        df = pd.read_csv(output_file)
        logger.info("\n=== SAMPLE OF OUTPUT ===")
        logger.info(f"First 5 rows:\n{df.head()}")
        logger.info(f"\nLabel distribution:\n{df['2_way_label'].value_counts()}")
        
    else:
        logger.error("❌ Failed to create balanced dataset")
        sys.exit(1)

if __name__ == "__main__":
    main() 