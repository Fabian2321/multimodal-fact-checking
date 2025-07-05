"""
Process Fakeddit training data for RAG integration.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any
from src.utils import setup_logger
from src.rag_handler import RAGHandler, RAGConfig

logger = setup_logger(__name__)

def process_fakeddit_data(correct_path: str, incorrect_path: str) -> List[Dict[str, Any]]:
    """Process Fakeddit data from correct and incorrect pairs into RAG-compatible format."""
    logger.info(f"Processing Fakeddit data from {correct_path} and {incorrect_path}")
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
    
    documents = []
    for _, row in df_correct.iterrows():
        doc = {
            "text": row["clean_title"],
            "label": 1,
            "source": "fakeddit",
            "metadata": {
                "id": str(row.get("id", "")),
                "image_url": str(row.get("image_url", "")),
                "has_image": bool(row.get("hasImage", False)),
                "subreddit": str(row.get("subreddit", "")),
                "score": safe_int(row.get("score")),
                "num_comments": safe_int(row.get("num_comments")),
                "upvote_ratio": safe_float(row.get("upvote_ratio")),
                "confidence": 0.9,
                "category": "training_data",
                "verification_notes": "Verified true example from training data"
            }
        }
        documents.append(doc)
    
    for _, row in df_incorrect.iterrows():
        doc = {
            "text": row["clean_title"],
            "label": 0,
            "source": "fakeddit",
            "metadata": {
                "id": str(row.get("id", "")),
                "image_url": str(row.get("image_url", "")),
                "has_image": bool(row.get("hasImage", False)),
                "subreddit": str(row.get("subreddit", "")),
                "score": safe_int(row.get("score")),
                "num_comments": safe_int(row.get("num_comments")),
                "upvote_ratio": safe_float(row.get("upvote_ratio")),
                "confidence": 0.9,
                "category": "training_data",
                "verification_notes": "Verified false example from training data"
            }
        }
        documents.append(doc)
    
    return documents

def extract_error_patterns(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract common error patterns from false examples."""
    false_examples = [doc for doc in documents if doc["label"] == 0]
    
    error_patterns = []
    pattern_categories = {
        "context_mismatch": [],
        "detail_mismatch": [],
        "temporal_mismatch": [],
        "location_mismatch": []
    }
    
    for doc in false_examples:
        text = doc["text"].lower()
        
        # Categorize error patterns
        if "out of context" in text or "misleading context" in text:
            pattern_categories["context_mismatch"].append(doc)
        elif "wrong details" in text or "incorrect details" in text:
            pattern_categories["detail_mismatch"].append(doc)
        elif "wrong time" in text or "incorrect date" in text:
            pattern_categories["temporal_mismatch"].append(doc)
        elif "wrong location" in text or "incorrect place" in text:
            pattern_categories["location_mismatch"].append(doc)
    
    # Create error pattern documents
    for category, examples in pattern_categories.items():
        if examples:
            pattern_doc = {
                "text": f"Common error pattern: {category}. Examples: {', '.join(ex['text'] for ex in examples[:3])}",
                "label": 0,
                "source": "error_patterns",
                "metadata": {
                    "category": category,
                    "example_count": len(examples),
                    "confidence": 0.9
                }
            }
            error_patterns.append(pattern_doc)
    
    return error_patterns

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process Fakeddit data for RAG integration")
    
    parser.add_argument("--train_correct_path", type=str, required=True,
                      help="Path to train_correct_pairs.csv")
    parser.add_argument("--train_incorrect_path", type=str, required=True,
                      help="Path to train_incorrect_pairs.csv")
    parser.add_argument("--output_path", type=str, default="data/knowledge_base",
                      help="Output path for processed data")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                      help="Sentence transformer model for embeddings")
    
    args = parser.parse_args()
    
    # Process Fakeddit data
    documents = process_fakeddit_data(args.train_correct_path, args.train_incorrect_path)
    
    # Extract error patterns
    error_patterns = extract_error_patterns(documents)
    
    # Combine all documents
    all_documents = documents + error_patterns
    
    # Initialize RAG handler
    rag_config = RAGConfig(
        embedding_model=args.embedding_model,
        knowledge_base_path=args.output_path
    )
    rag_handler = RAGHandler(rag_config)
    
    # Add documents to RAG system
    logger.info(f"Adding {len(all_documents)} documents to knowledge base")
    rag_handler.add_documents(all_documents)
    
    # Save document metadata
    metadata = {
        "total_documents": len(all_documents),
        "document_types": {
            "training_data": len(documents),
            "error_patterns": len(error_patterns)
        },
        "config": {
            "embedding_model": args.embedding_model,
            "knowledge_base_path": args.output_path
        }
    }
    
    metadata_path = os.path.join(args.output_path, "fakeddit_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Processing complete. Results saved to {args.output_path}")

if __name__ == "__main__":
    main() 