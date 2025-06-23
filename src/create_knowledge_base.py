"""
Script to create and populate the initial knowledge base for RAG.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any
from src.utils import setup_logger
from src.rag_handler import RAGHandler, RAGConfig

logger = setup_logger(__name__)

def load_fakeddit_data(data_path: str) -> List[Dict[str, Any]]:
    """Load and format Fakeddit data for knowledge base."""
    logger.info(f"Loading Fakeddit data from {data_path}")
    df = pd.read_csv(data_path)
    
    documents = []
    for _, row in df.iterrows():
        doc = {
            "text": row["text"],
            "label": int(row["label"]),
            "source": "fakeddit",
            "metadata": {
                "id": row["id"],
                "image_path": row.get("image_path", ""),
                "confidence": row.get("confidence", 1.0)
            }
        }
        documents.append(doc)
    
    return documents

def load_external_knowledge(external_path: str) -> List[Dict[str, Any]]:
    """Load external knowledge sources."""
    logger.info(f"Loading external knowledge from {external_path}")
    documents = []
    
    # Load fact-checking guidelines
    guidelines_path = os.path.join(external_path, "fact_checking_guidelines.json")
    if os.path.exists(guidelines_path):
        with open(guidelines_path, 'r') as f:
            guidelines = json.load(f)
            for guideline in guidelines:
                doc = {
                    "text": guideline["text"],
                    "label": 1,  # Guidelines are always true
                    "source": "guidelines",
                    "metadata": {
                        "category": guideline.get("category", "general"),
                        "importance": guideline.get("importance", "high")
                    }
                }
                documents.append(doc)
    
    # Load common misconceptions
    misconceptions_path = os.path.join(external_path, "common_misconceptions.json")
    if os.path.exists(misconceptions_path):
        with open(misconceptions_path, 'r') as f:
            misconceptions = json.load(f)
            for misconception in misconceptions:
                doc = {
                    "text": misconception["text"],
                    "label": 0,  # Misconceptions are false
                    "source": "misconceptions",
                    "metadata": {
                        "correction": misconception.get("correction", ""),
                        "category": misconception.get("category", "general")
                    }
                }
                documents.append(doc)
    
    return documents

def create_knowledge_base(
    fakeddit_path: str,
    external_path: str,
    output_path: str,
    rag_config: RAGConfig
):
    """Create and populate the knowledge base."""
    # Initialize RAG handler
    rag_handler = RAGHandler(rag_config)
    
    # Load documents from different sources
    fakeddit_docs = load_fakeddit_data(fakeddit_path)
    external_docs = load_external_knowledge(external_path)
    
    # Combine all documents
    all_docs = fakeddit_docs + external_docs
    
    # Add documents to RAG system
    logger.info(f"Adding {len(all_docs)} documents to knowledge base")
    rag_handler.add_documents(all_docs)
    
    # Save document metadata
    metadata_path = os.path.join(output_path, "knowledge_base_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump({
            "total_documents": len(all_docs),
            "sources": {
                "fakeddit": len(fakeddit_docs),
                "external": len(external_docs)
            },
            "config": {
                "embedding_model": rag_config.embedding_model,
                "index_type": rag_config.index_type,
                "top_k": rag_config.top_k,
                "similarity_threshold": rag_config.similarity_threshold
            }
        }, f, indent=2)
    
    logger.info(f"Knowledge base created successfully at {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Create initial knowledge base for RAG")
    
    parser.add_argument("--fakeddit_path", type=str, required=True,
                      help="Path to Fakeddit dataset CSV")
    parser.add_argument("--external_path", type=str, required=True,
                      help="Path to external knowledge sources")
    parser.add_argument("--output_path", type=str, default="data/knowledge_base",
                      help="Output path for knowledge base")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                      help="Sentence transformer model for embeddings")
    
    args = parser.parse_args()
    
    # Create RAG config
    rag_config = RAGConfig(
        embedding_model=args.embedding_model,
        knowledge_base_path=args.output_path
    )
    
    # Create knowledge base
    create_knowledge_base(
        args.fakeddit_path,
        args.external_path,
        args.output_path,
        rag_config
    )

if __name__ == "__main__":
    main() 