#!/usr/bin/env python3
"""
Example Generator and Qualitative Analysis for Multimodal Fact-Checking Project
Generates example prompts, sample outputs, and qualitative analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from PIL import Image
import random
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
from src.core.utils import setup_logger

class ExampleGenerator:
    def __init__(self, results_dir: str = "results", data_dir: str = "data"):
        self.results_dir = Path(results_dir)
        self.data_dir = Path(data_dir)
        self.examples_dir = self.results_dir / "final_experiments" / "examples"
        self.examples_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test data
        self.test_data = pd.read_csv(self.data_dir / "processed" / "test_balanced_pairs.csv")
        
        # Define interesting examples for analysis
        self.interesting_cases = [
            # Real examples that might be challenging
            "statue of poseidon",
            "north korea builds replica of souths presidential blue house for target practice",
            "a young monk in laos had pressing life questions for president obama last week obama answered them",
            
            # Fake examples that are obvious
            "jesus christ converting local teens to christianity repainted and then circa ad",
            "victory the great european crusade vichy france",
            "report reveals jesus christ have benefited from fathers influential position to gain highpowered role as lord and savior",
            
            # Ambiguous cases
            "this concerned sink with a tiny hat",
            "i found a face in my sheet music too",
            "major thermos"
        ]

    def generate_prompt_examples(self):
        """Generate example prompts for different models."""
        examples = {
            "CLIP": {
                "description": "CLIP uses similarity scoring between image and text embeddings",
                "method": "Zero-shot classification via cosine similarity",
                "output": "Similarity score (higher = more likely to be real)"
            },
            "BLIP2": {
                "prompts": {
                    "Zero-shot Forced Choice": "Does the text '{text}' accurately describe the image? Answer only with 'Yes' or 'No':",
                    "Zero-shot with Justification": "Text: {text}\nDoes the text match the image?\nPlease answer with only 'Yes.' or 'No.' on the first line, then a short justification on the next line.",
                    "Chain of Thought": "Analyze the following image and text. First, describe the key elements in the image. Second, compare these elements to the claim in the text. Finally, conclude if the text is a completely accurate description of the image.\nText: '{text}'\nAnalysis:",
                    "Few-shot": "Example 1:\nText: 'A dog sitting on a couch'\nAnswer:\nYes.\nThe image shows a dog on a couch.\n\nExample 2:\nText: 'A cat playing the piano'\nAnswer:\nNo.\nThe image does not show a cat or a piano.\n\nYour turn:\nText: '{text}'\nAnswer:"
                }
            },
            "LLaVA": {
                "prompts": {
                    "Zero-shot Chain of Thought": "USER: <image>\nAnalyze the following image and text. First, describe the key elements in the image. Second, compare these elements to the claim in the text. Finally, conclude if the text is a completely accurate description of the image.\nText: '{text}'\nASSISTANT:",
                    "Zero-shot Forced Choice": "USER: <image>\nIs the text '{text}' an 'Accurate' or 'Inaccurate' description of the provided image? Only answer with one of these two words.\nASSISTANT:",
                    "Few-shot Step-by-step": "USER: <image>\nText: 'A dog sitting on a couch'\nASSISTANT: The image shows a dog on a couch, so the text is accurate.\n\nUSER: <image>\nText: 'A cat playing the piano'\nASSISTANT: The image does not show a cat or piano, so the text is inaccurate.\n\nUSER: <image>\nText: '{text}'\nASSISTANT:"
                }
            }
        }
        
        # Save prompt examples
        with open(self.examples_dir / "prompt_examples.json", 'w') as f:
            json.dump(examples, f, indent=2)
        
        # Create markdown documentation
        with open(self.examples_dir / "prompt_examples.md", 'w') as f:
            f.write("# Prompt Examples for Multimodal Fact-Checking\n\n")
            
            for model, info in examples.items():
                f.write(f"## {model}\n\n")
                
                if model == "CLIP":
                    f.write(f"**Description**: {info['description']}\n\n")
                    f.write(f"**Method**: {info['method']}\n\n")
                    f.write(f"**Output**: {info['output']}\n\n")
                else:
                    f.write("**Available Prompts**:\n\n")
                    for prompt_name, prompt_text in info['prompts'].items():
                        f.write(f"### {prompt_name}\n")
                        f.write(f"```\n{prompt_text}\n```\n\n")
                
                f.write("---\n\n")
        
        print(f"Prompt examples saved to: {self.examples_dir}")

    def find_sample_outputs(self):
        """Find and extract sample outputs from experiment results."""
        sample_outputs = []
        
        # Look for results in all model directories
        for model_dir in self.results_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name == "final_experiments":
                continue
                
            for exp_dir in model_dir.iterdir():
                if not exp_dir.is_dir():
                    continue
                    
                results_file = exp_dir / "all_model_outputs.csv"
                if results_file.exists():
                    try:
                        df = pd.read_csv(results_file)
                        if not df.empty and 'generated_text' in df.columns:
                            # Find interesting examples
                            for _, row in df.iterrows():
                                text = row.get('text', '')
                                if any(case.lower() in text.lower() for case in self.interesting_cases):
                                    sample_outputs.append({
                                        'experiment': exp_dir.name,
                                        'model': model_dir.name,
                                        'text': text,
                                        'true_label': row.get('true_label', 'N/A'),
                                        'predicted_label': row.get('predicted_label', 'N/A'),
                                        'generated_text': row.get('generated_text', 'N/A'),
                                        'image_path': row.get('image_path', 'N/A')
                                    })
                    except Exception as e:
                        print(f"Error processing {results_file}: {e}")
        
        # Save sample outputs
        if sample_outputs:
            sample_df = pd.DataFrame(sample_outputs)
            sample_df.to_csv(self.examples_dir / "sample_outputs.csv", index=False)
            
            # Create markdown summary
            with open(self.examples_dir / "sample_outputs.md", 'w') as f:
                f.write("# Sample Model Outputs\n\n")
                
                for _, row in sample_df.iterrows():
                    f.write(f"## {row['model'].upper()} - {row['experiment']}\n\n")
                    f.write(f"**Text**: {row['text']}\n\n")
                    f.write(f"**True Label**: {row['true_label']}\n\n")
                    f.write(f"**Predicted Label**: {row['predicted_label']}\n\n")
                    f.write(f"**Generated Response**:\n```\n{row['generated_text']}\n```\n\n")
                    f.write("---\n\n")
            
            print(f"Sample outputs saved to: {self.examples_dir}")
        else:
            print("No sample outputs found!")

    def create_error_analysis(self):
        """Analyze common error patterns."""
        error_analysis = {
            "error_categories": {
                "false_positives": "Model predicts fake content as real",
                "false_negatives": "Model predicts real content as fake",
                "ambiguous_cases": "Cases where human judgment might differ",
                "prompt_failures": "Models not following prompt instructions"
            },
            "common_patterns": [
                "Over-reliance on text content vs visual evidence",
                "Difficulty with sarcasm and humor",
                "Confusion with historical vs contemporary context",
                "Sensitivity to text length and complexity"
            ]
        }
        
        # Save error analysis
        with open(self.examples_dir / "error_analysis.json", 'w') as f:
            json.dump(error_analysis, f, indent=2)
        
        print(f"Error analysis saved to: {self.examples_dir}")

    def generate_qualitative_insights(self):
        """Generate qualitative insights about model behavior."""
        insights = {
            "model_strengths": {
                "CLIP": [
                    "Fast inference",
                    "Good at detecting obvious visual-text mismatches",
                    "Consistent scoring"
                ],
                "BLIP2": [
                    "Detailed reasoning capabilities",
                    "Good at complex visual understanding",
                    "Flexible prompting"
                ],
                "LLaVA": [
                    "Strong reasoning abilities",
                    "Good at step-by-step analysis",
                    "Natural language explanations"
                ]
            },
            "model_weaknesses": {
                "CLIP": [
                    "No reasoning capabilities",
                    "Sensitive to threshold selection",
                    "Limited to similarity scoring"
                ],
                "BLIP2": [
                    "Can be inconsistent with prompts",
                    "Sometimes ignores visual evidence",
                    "Output parsing challenges"
                ],
                "LLaVA": [
                    "Slower inference",
                    "Can be verbose",
                    "Sometimes hallucinates details"
                ]
            },
            "rag_benefits": [
                "Provides external context",
                "Improves reasoning with background knowledge",
                "Helps with ambiguous cases"
            ],
            "prompt_engineering_insights": [
                "Forced-choice prompts improve consistency",
                "Few-shot examples help with task understanding",
                "Chain-of-thought prompts improve reasoning",
                "Clear instructions reduce ambiguity"
            ]
        }
        
        # Save insights
        with open(self.examples_dir / "qualitative_insights.json", 'w') as f:
            json.dump(insights, f, indent=2)
        
        # Create markdown summary
        with open(self.examples_dir / "qualitative_insights.md", 'w') as f:
            f.write("# Qualitative Insights\n\n")
            
            f.write("## Model Strengths\n\n")
            for model, strengths in insights["model_strengths"].items():
                f.write(f"### {model}\n")
                for strength in strengths:
                    f.write(f"- {strength}\n")
                f.write("\n")
            
            f.write("## Model Weaknesses\n\n")
            for model, weaknesses in insights["model_weaknesses"].items():
                f.write(f"### {model}\n")
                for weakness in weaknesses:
                    f.write(f"- {weakness}\n")
                f.write("\n")
            
            f.write("## RAG Benefits\n\n")
            for benefit in insights["rag_benefits"]:
                f.write(f"- {benefit}\n")
            f.write("\n")
            
            f.write("## Prompt Engineering Insights\n\n")
            for insight in insights["prompt_engineering_insights"]:
                f.write(f"- {insight}\n")
            f.write("\n")
        
        print(f"Qualitative insights saved to: {self.examples_dir}")

    def create_visualization_examples(self):
        """Create example visualizations for presentation."""
        # Create a sample confusion matrix visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Example Model Performance Visualizations', fontsize=16, fontweight='bold')
        
        # Sample confusion matrix
        ax1 = axes[0, 0]
        cm = np.array([[85, 15], [20, 80]])  # Example confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Predicted Fake', 'Predicted Real'],
                   yticklabels=['Actual Fake', 'Actual Real'], ax=ax1)
        ax1.set_title('Example Confusion Matrix')
        
        # Sample accuracy comparison with realistic expectations
        ax2 = axes[0, 1]
        models = ['CLIP', 'BLIP2', 'LLaVA']
        accuracies = [0.55, 0.60, 0.65]  # Realistic expectations
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax2.bar(models, accuracies, color=colors)
        ax2.set_title('Example Accuracy Comparison (Realistic)')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.2f}', ha='center', va='bottom')
        
        # Sample precision-recall curve
        ax3 = axes[1, 0]
        precision = [0.8, 0.85, 0.82, 0.78, 0.75]
        recall = [0.7, 0.75, 0.8, 0.85, 0.9]
        ax3.plot(recall, precision, 'o-', color='#FF6B6B', linewidth=2, markersize=8)
        ax3.set_xlabel('Recall')
        ax3.set_ylabel('Precision')
        ax3.set_title('Example Precision-Recall Curve')
        ax3.grid(True, alpha=0.3)
        
        # Sample RAG impact with realistic expectations
        ax4 = axes[1, 1]
        categories = ['CLIP', 'BLIP2', 'LLaVA']
        without_rag = [0.53, 0.58, 0.63]  # Realistic without RAG
        with_rag = [0.55, 0.60, 0.65]     # Realistic with RAG (small improvement)
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x - width/2, without_rag, width, label='Without RAG', color='#FF6B6B', alpha=0.7)
        ax4.bar(x + width/2, with_rag, width, label='With RAG', color='#4ECDC4', alpha=0.7)
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('RAG Impact Example')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.examples_dir / 'example_visualizations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Example visualizations saved to: {self.examples_dir}")

    def generate_all_examples(self):
        """Generate all example materials."""
        print("=== GENERATING EXAMPLES AND ANALYSIS ===")
        
        print("Generating prompt examples...")
        self.generate_prompt_examples()
        
        print("Finding sample outputs...")
        self.find_sample_outputs()
        
        print("Creating error analysis...")
        self.create_error_analysis()
        
        print("Generating qualitative insights...")
        self.generate_qualitative_insights()
        
        print("Creating example visualizations...")
        self.create_visualization_examples()
        
        print(f"\nAll examples and analysis saved to: {self.examples_dir}")
        print("Generated files:")
        print("  - prompt_examples.json/md")
        print("  - sample_outputs.csv/md")
        print("  - error_analysis.json")
        print("  - qualitative_insights.json/md")
        print("  - example_visualizations.png")

if __name__ == "__main__":
    generator = ExampleGenerator()
    generator.generate_all_examples() 