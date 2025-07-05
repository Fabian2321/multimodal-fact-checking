#!/usr/bin/env python3
"""
Integrate Enhanced Guidelines into RAG Knowledge Base

This script integrates the enhanced guidelines from data/external_knowledge
into the RAG knowledge base for improved fact-checking performance.
"""

import json
import os
from pathlib import Path
import logging
from src.rag_handler import RAGHandler, RAGConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_enhanced_guidelines(file_path):
    """
    Load enhanced guidelines from JSON file and convert to RAG document format.
    
    Args:
        file_path: Path to JSON file containing enhanced guidelines
        
    Returns:
        list: List of documents in RAG format
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        # Handle enhanced guidelines structure
        if 'enhanced_image_guidelines' in data:
            for guideline in data['enhanced_image_guidelines']:
                doc = {
                    'text': guideline['text'],
                    'category': guideline.get('category', 'image_verification'),
                    'importance': guideline.get('importance', 'medium'),
                    'priority': guideline.get('priority', 999),
                    'type': 'enhanced_image_guideline',
                    'examples': guideline.get('examples', []),
                    'keywords': guideline.get('keywords', [])
                }
                documents.append(doc)
                
        elif 'enhanced_misconceptions' in data:
            for misconception in data['enhanced_misconceptions']:
                doc = {
                    'text': f"ENHANCED MISCONCEPTION: {misconception['misconception']} CORRECTION: {misconception['correction']}",
                    'category': misconception.get('category', 'general'),
                    'priority': misconception.get('priority', 999),
                    'type': 'enhanced_misconception_correction',
                    'misconception': misconception['misconception'],
                    'correction': misconception['correction'],
                    'examples': misconception.get('examples', [])
                }
                documents.append(doc)
                
        elif 'enhanced_fact_checking_guidelines' in data:
            for guideline in data['enhanced_fact_checking_guidelines']:
                doc = {
                    'text': guideline['text'],
                    'category': guideline.get('category', 'fact_checking'),
                    'importance': guideline.get('importance', 'medium'),
                    'priority': guideline.get('priority', 999),
                    'type': 'enhanced_fact_checking_guideline',
                    'keywords': guideline.get('keywords', [])
                }
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} enhanced documents from {file_path}")
        return documents
        
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []

def integrate_enhanced_guidelines(knowledge_base_path, external_knowledge_path):
    """
    Integrate enhanced guidelines into the RAG knowledge base.
    
    Args:
        knowledge_base_path: Path to RAG knowledge base
        external_knowledge_path: Path to external knowledge directory
    """
    logger.info("=== INTEGRATING ENHANCED GUIDELINES INTO RAG KNOWLEDGE BASE ===")
    
    # Initialize RAG handler
    rag_config = RAGConfig(knowledge_base_path=knowledge_base_path)
    rag_handler = RAGHandler(rag_config)
    
    # Load all enhanced guideline files
    enhanced_guideline_files = [
        'enhanced_image_guidelines.json',
        'enhanced_misconceptions.json', 
        'enhanced_fact_checking_guidelines.json'
    ]
    
    all_documents = []
    
    for filename in enhanced_guideline_files:
        file_path = os.path.join(external_knowledge_path, filename)
        if os.path.exists(file_path):
            logger.info(f"Processing enhanced {filename}...")
            documents = load_enhanced_guidelines(file_path)
            all_documents.extend(documents)
        else:
            logger.warning(f"Enhanced file not found: {file_path}")
    
    if not all_documents:
        logger.error("No enhanced guidelines found to integrate!")
        return False
    
    # Sort documents by priority (lower number = higher priority)
    all_documents.sort(key=lambda x: x.get('priority', 999))
    
    # Add documents to RAG system
    logger.info(f"Adding {len(all_documents)} enhanced guideline documents to knowledge base...")
    rag_handler.add_documents(all_documents)
    
    # Save metadata about the enhanced integration
    metadata = {
        "enhanced_guidelines_integration": {
            "total_enhanced_guidelines_added": len(all_documents),
            "files_processed": enhanced_guideline_files,
            "document_types": {
                "enhanced_image_guidelines": len([d for d in all_documents if d.get('type') == 'enhanced_image_guideline']),
                "enhanced_misconception_corrections": len([d for d in all_documents if d.get('type') == 'enhanced_misconception_correction']),
                "enhanced_fact_checking_guidelines": len([d for d in all_documents if d.get('type') == 'enhanced_fact_checking_guideline'])
            },
            "priority_distribution": {
                "high_priority": len([d for d in all_documents if d.get('priority', 999) <= 5]),
                "medium_priority": len([d for d in all_documents if 5 < d.get('priority', 999) <= 10]),
                "low_priority": len([d for d in all_documents if d.get('priority', 999) > 10])
            }
        }
    }
    
    metadata_path = os.path.join(knowledge_base_path, "enhanced_guidelines_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Enhanced guidelines integration complete!")
    logger.info(f"Added {len(all_documents)} enhanced guideline documents")
    logger.info(f"Metadata saved to {metadata_path}")
    
    # Test retrieval with enhanced queries
    logger.info("Testing enhanced retrieval with sample queries...")
    test_queries = [
        "How to verify exact visual match between text and image",
        "Common misconceptions about image authenticity",
        "Systematic approach to fact-checking images"
    ]
    
    for query in test_queries:
        retrieved_docs = rag_handler.retrieve(query)
        logger.info(f"Retrieved {len(retrieved_docs)} documents for query: '{query}'")
    
    return True

def main():
    """Main function to integrate enhanced guidelines."""
    
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
    
    # Integrate enhanced guidelines
    success = integrate_enhanced_guidelines(knowledge_base_path, external_knowledge_path)
    
    if success:
        logger.info("✅ Enhanced guidelines successfully integrated into RAG knowledge base!")
        logger.info("The knowledge base now contains advanced expert guidelines for superior fact-checking.")
        logger.info("🚀 Ready for enhanced RAG experiments!")
    else:
        logger.error("❌ Failed to integrate enhanced guidelines")
    
    return success

if __name__ == "__main__":
    main() 