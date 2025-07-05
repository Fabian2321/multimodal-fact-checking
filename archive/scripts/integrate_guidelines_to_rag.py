#!/usr/bin/env python3
"""
Integrate External Guidelines into RAG Knowledge Base

This script integrates the hand-curated guidelines from data/external_knowledge
into the RAG knowledge base to improve fact-checking performance.
"""

import json
import os
from pathlib import Path
import logging
from src.rag_handler import RAGHandler, RAGConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_guidelines_from_json(file_path):
    """
    Load guidelines from JSON file and convert to RAG document format.
    
    Args:
        file_path: Path to JSON file containing guidelines
        
    Returns:
        list: List of documents in RAG format
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        # Handle different JSON structures
        if 'image_guidelines' in data:
            # image_specific_guidelines.json
            for guideline in data['image_guidelines']:
                doc = {
                    'text': guideline['text'],
                    'category': guideline.get('category', 'image_verification'),
                    'importance': guideline.get('importance', 'medium'),
                    'type': 'image_guideline',
                    'examples': guideline.get('examples', [])
                }
                documents.append(doc)
                
        elif 'misconceptions' in data:
            # common_misconceptions.json
            for misconception in data['misconceptions']:
                doc = {
                    'text': f"Misconception: {misconception['text']} Correction: {misconception['correction']}",
                    'category': misconception.get('category', 'general'),
                    'type': 'misconception_correction',
                    'misconception': misconception['text'],
                    'correction': misconception['correction']
                }
                documents.append(doc)
                
        elif 'guidelines' in data:
            # fact_checking_guidelines.json
            for guideline in data['guidelines']:
                doc = {
                    'text': guideline['text'],
                    'category': guideline.get('category', 'fact_checking'),
                    'importance': guideline.get('importance', 'medium'),
                    'type': 'fact_checking_guideline'
                }
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from {file_path}")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

def integrate_guidelines_to_rag(knowledge_base_path, external_knowledge_path):
    """
    Integrate external guidelines into the RAG knowledge base.
    
    Args:
        knowledge_base_path: Path to RAG knowledge base
        external_knowledge_path: Path to external knowledge directory
    """
    logger.info("=== INTEGRATING GUIDELINES INTO RAG KNOWLEDGE BASE ===")
    
    # Initialize RAG handler
    rag_config = RAGConfig(knowledge_base_path=knowledge_base_path)
    rag_handler = RAGHandler(rag_config)
    
    # Load all guideline files
    guideline_files = [
        'image_specific_guidelines.json',
        'common_misconceptions.json', 
        'fact_checking_guidelines.json'
    ]
    
    all_documents = []
    
    for filename in guideline_files:
        file_path = os.path.join(external_knowledge_path, filename)
        if os.path.exists(file_path):
            logger.info(f"Processing {filename}...")
            documents = load_guidelines_from_json(file_path)
            all_documents.extend(documents)
        else:
            logger.warning(f"File not found: {file_path}")
    
    if not all_documents:
        logger.error("No guidelines found to integrate!")
        return False
    
    # Add documents to RAG system
    logger.info(f"Adding {len(all_documents)} guideline documents to knowledge base...")
    rag_handler.add_documents(all_documents)
    
    # Save metadata about the integration
    metadata = {
        "guidelines_integration": {
            "total_guidelines_added": len(all_documents),
            "files_processed": guideline_files,
            "document_types": {
                "image_guidelines": len([d for d in all_documents if d.get('type') == 'image_guideline']),
                "misconception_corrections": len([d for d in all_documents if d.get('type') == 'misconception_correction']),
                "fact_checking_guidelines": len([d for d in all_documents if d.get('type') == 'fact_checking_guideline'])
            }
        }
    }
    
    metadata_path = os.path.join(knowledge_base_path, "guidelines_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Guidelines integration complete!")
    logger.info(f"Added {len(all_documents)} guideline documents")
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Test retrieval with a sample query
    logger.info("Testing retrieval with sample query...")
    test_query = "How to verify if image matches text description"
    retrieved_docs = rag_handler.retrieve(test_query)
    logger.info(f"Retrieved {len(retrieved_docs)} documents for test query")
    
    return True

def main():
    """Main function to integrate guidelines."""
    
    # Paths
    knowledge_base_path = "data/knowledge_base"
    external_knowledge_path = "data/external_knowledge"
    
    # Check if paths exist
    if not os.path.exists(knowledge_base_path):
        logger.error(f"Knowledge base path not found: {knowledge_base_path}")
        return False
        
    if not os.path.exists(external_knowledge_path):
        logger.error(f"External knowledge path not found: {external_knowledge_path}")
        return False
    
    # Integrate guidelines
    success = integrate_guidelines_to_rag(knowledge_base_path, external_knowledge_path)
    
    if success:
        logger.info("✅ Guidelines successfully integrated into RAG knowledge base!")
        logger.info("The knowledge base now contains expert guidelines for better fact-checking.")
    else:
        logger.error("❌ Failed to integrate guidelines")
    
    return success

if __name__ == "__main__":
    main() 