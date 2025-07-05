#!/usr/bin/env python3
"""
Create initial knowledge base for RAG system.
"""

import json
import os
from typing import List, Dict, Any

def create_initial_knowledge_base():
    """Create initial knowledge base documents for fact-checking."""
    
    # Knowledge base documents for fact-checking
    documents = [
        {
            "id": "fact_checking_guide_1",
            "text": "Fact-checking involves verifying the accuracy of claims by examining evidence and sources. When analyzing images and text, look for inconsistencies, check if the text accurately describes what is visible in the image, and consider whether the claim is supported by visual evidence.",
            "type": "fact_checking_guide",
            "category": "methodology"
        },
        {
            "id": "image_analysis_1", 
            "text": "When analyzing images for fact-checking, examine the visual elements carefully. Look for signs of manipulation, check if objects and people are consistent with the text description, and verify if the image context matches the claim being made.",
            "type": "image_analysis",
            "category": "methodology"
        },
        {
            "id": "text_verification_1",
            "text": "Text verification requires comparing written claims against visual evidence. Check if the text accurately describes what can be seen in the image, whether details match, and if there are any contradictions between the text and visual content.",
            "type": "text_verification", 
            "category": "methodology"
        },
        {
            "id": "manipulation_signs_1",
            "text": "Common signs of image manipulation include inconsistent lighting, shadows that don't match, objects that appear to float, unnatural edges, and elements that don't fit the overall scene. These can indicate that an image has been altered.",
            "type": "manipulation_detection",
            "category": "detection"
        },
        {
            "id": "context_analysis_1",
            "text": "Context analysis involves examining the broader context of an image and text. Consider the setting, time period, location, and whether the combination makes logical sense. Inconsistent context can indicate misinformation.",
            "type": "context_analysis",
            "category": "methodology"
        },
        {
            "id": "source_verification_1",
            "text": "Source verification is crucial for fact-checking. Check the origin of images and text, verify if they come from reliable sources, and determine if they have been taken out of context or misrepresented.",
            "type": "source_verification",
            "category": "methodology"
        },
        {
            "id": "misinformation_patterns_1",
            "text": "Common misinformation patterns include using real images with false captions, taking images out of context, combining unrelated images and text, and using manipulated images to support false claims.",
            "type": "misinformation_patterns",
            "category": "detection"
        },
        {
            "id": "verification_steps_1",
            "text": "Fact-checking steps: 1) Examine the image carefully, 2) Read the text claim, 3) Compare text to visual evidence, 4) Check for inconsistencies, 5) Verify sources if possible, 6) Determine if the claim is accurate.",
            "type": "verification_steps",
            "category": "methodology"
        },
        {
            "id": "accuracy_criteria_1",
            "text": "Accuracy criteria for image-text matching: the text must accurately describe what is visible in the image, details must match, there should be no contradictions, and the claim should be supported by visual evidence.",
            "type": "accuracy_criteria",
            "category": "evaluation"
        },
        {
            "id": "red_flags_1",
            "text": "Red flags for potential misinformation: dramatic or sensational claims, images that seem too perfect, text that doesn't match the image content, lack of source attribution, and claims that seem designed to provoke emotional reactions.",
            "type": "red_flags",
            "category": "detection"
        }
    ]
    
    return documents

def save_knowledge_base(documents: List[Dict[str, Any]], output_path: str):
    """Save knowledge base documents to JSON file."""
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save documents
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"Knowledge base saved to: {output_path}")
    print(f"Total documents: {len(documents)}")

def main():
    """Main function to create and save knowledge base."""
    
    # Create knowledge base
    documents = create_initial_knowledge_base()
    
    # Save to data directory
    output_path = "data/knowledge_base/initial_docs.json"
    save_knowledge_base(documents, output_path)
    
    # Print summary
    print("\nKnowledge base summary:")
    categories = {}
    for doc in documents:
        cat = doc.get('category', 'unknown')
        categories[cat] = categories.get(cat, 0) + 1
    
    for category, count in categories.items():
        print(f"  {category}: {count} documents")

if __name__ == "__main__":
    main() 