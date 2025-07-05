"""
Create test queries from validation data.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any
from src.utils import setup_logger

logger = setup_logger(__name__)

def create_test_queries(correct_path: str, incorrect_path: str, output_path: str):
    """Create test queries from validation data."""
    logger.info(f"Creating test queries from {correct_path} and {incorrect_path}")
    
    # Load validation data
    df_correct = pd.read_csv(correct_path)
    df_incorrect = pd.read_csv(incorrect_path)
    
    def safe_int(value, default=0):
        try:
            return int(float(value)) if pd.notna(value) else default
        except (ValueError, TypeError):
            return default
    
    def safe_float(value, default=0.0):
        try:
            return float(value) if pd.notna(value) else default
        except (ValueError, TypeError):
            return default
    
    # Create test queries
    test_queries = []
    
    # Add correct examples
    for _, row in df_correct.iterrows():
        query = {
            "text": row["clean_title"],
            "label": 1,
            "metadata": {
                "id": str(row.get("id", "")),
                "image_url": str(row.get("image_url", "")),
                "has_image": bool(row.get("hasImage", False)),
                "subreddit": str(row.get("subreddit", "")),
                "score": safe_int(row.get("score")),
                "num_comments": safe_int(row.get("num_comments")),
                "upvote_ratio": safe_float(row.get("upvote_ratio"))
            }
        }
        test_queries.append(query)
    
    # Add incorrect examples
    for _, row in df_incorrect.iterrows():
        query = {
            "text": row["clean_title"],
            "label": 0,
            "metadata": {
                "id": str(row.get("id", "")),
                "image_url": str(row.get("image_url", "")),
                "has_image": bool(row.get("hasImage", False)),
                "subreddit": str(row.get("subreddit", "")),
                "score": safe_int(row.get("score")),
                "num_comments": safe_int(row.get("num_comments")),
                "upvote_ratio": safe_float(row.get("upvote_ratio"))
            }
        }
        test_queries.append(query)
    
    # Save test queries
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(test_queries, f, indent=2)
    
    logger.info(f"Created {len(test_queries)} test queries. Saved to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create test queries from validation data")
    
    parser.add_argument("--validate_correct_path", type=str, required=True,
                      help="Path to validate_correct_pairs.csv")
    parser.add_argument("--validate_incorrect_path", type=str, required=True,
                      help="Path to validate_incorrect_pairs.csv")
    parser.add_argument("--output_path", type=str, default="data/test_queries.json",
                      help="Path to save test queries")
    
    args = parser.parse_args()
    
    create_test_queries(
        args.validate_correct_path,
        args.validate_incorrect_path,
        args.output_path
    )

if __name__ == "__main__":
    main() 