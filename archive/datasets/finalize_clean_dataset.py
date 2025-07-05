#!/usr/bin/env python3
"""
Finalize Clean Dataset Script

This script finalizes the clean dataset by:
1. Creating a backup of the old corrupted file
2. Replacing the old file with the new clean one
3. Verifying the final result
"""

import shutil
import os
from pathlib import Path
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_dataset(file_path):
    """
    Verify that the dataset is clean and properly balanced.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    try:
        df = pd.read_csv(file_path)
        
        # Check total samples
        total_samples = len(df)
        if total_samples != 100:
            logger.error(f"Expected 100 data rows, got {total_samples}")
            return False
            
        # Check label distribution
        label_counts = df['2_way_label'].value_counts()
        real_count = label_counts.get(0, 0)
        fake_count = label_counts.get(1, 0)
        
        if real_count != 50 or fake_count != 50:
            logger.error(f"Expected 50 real and 50 fake samples, got {real_count} real and {fake_count} fake")
            return False
            
        # Check for any invalid labels
        invalid_labels = df[~df['2_way_label'].isin([0, 1])]
        if len(invalid_labels) > 0:
            logger.error(f"Found {len(invalid_labels)} rows with invalid labels")
            return False
            
        # Check for missing required fields
        required_fields = ['author', 'clean_title', 'title', 'image_url', 'subreddit']
        for field in required_fields:
            missing_count = df[field].isna().sum()
            if missing_count > 0:
                logger.error(f"Found {missing_count} rows with missing {field}")
                return False
                
        logger.info("✅ Dataset verification passed!")
        logger.info(f"Total samples: {total_samples - 1}")  # Exclude header
        logger.info(f"Real samples (0): {real_count}")
        logger.info(f"Fake samples (1): {fake_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying dataset: {e}")
        return False

def main():
    """Main function to finalize the clean dataset."""
    
    data_dir = Path("data/processed")
    
    # File paths
    old_file = data_dir / "test_balanced_pairs.csv"
    new_file = data_dir / "test_balanced_pairs_clean.csv"
    backup_file = data_dir / "test_balanced_pairs_backup.csv"
    
    logger.info("=== FINALIZING CLEAN DATASET ===")
    
    # Check if new clean file exists
    if not new_file.exists():
        logger.error(f"Clean dataset file not found: {new_file}")
        return False
        
    # Verify the new clean dataset
    logger.info("Verifying clean dataset...")
    if not verify_dataset(new_file):
        logger.error("Clean dataset verification failed!")
        return False
    
    # Create backup of old file if it exists
    if old_file.exists():
        logger.info(f"Creating backup of old file: {backup_file}")
        shutil.copy2(old_file, backup_file)
        logger.info("✅ Backup created successfully")
    else:
        logger.warning("No old file found to backup")
    
    # Replace old file with new clean file
    logger.info(f"Replacing {old_file} with clean version...")
    shutil.copy2(new_file, old_file)
    logger.info("✅ File replaced successfully")
    
    # Final verification
    logger.info("Performing final verification...")
    if verify_dataset(old_file):
        logger.info("🎉 SUCCESS: Clean dataset is now active!")
        logger.info(f"Active file: {old_file}")
        logger.info(f"Backup file: {backup_file}")
        logger.info(f"Clean source: {new_file}")
        
        # Show sample of final dataset
        df = pd.read_csv(old_file)
        logger.info("\n=== SAMPLE OF FINAL DATASET ===")
        logger.info(f"First 3 rows:\n{df.head(3)[['author', 'clean_title', '2_way_label']]}")
        
        return True
    else:
        logger.error("❌ Final verification failed!")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1) 