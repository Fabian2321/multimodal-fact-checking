#!/usr/bin/env python3
"""
Final Results Generator for Multimodal Fact-Checking Project
Generates comprehensive analysis, tables, and visualizations from all experiments.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinalResultsGenerator:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.final_dir = self.results_dir / "final_experiments"
        self.figures_dir = self.final_dir / "figures"
        self.tables_dir = self.final_dir / "tables"
        self.reports_dir = self.final_dir / "reports"
        
        # Create directories
        for dir_path in [self.final_dir, self.figures_dir, self.tables_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Define experiment categories
        self.experiment_categories = {
            'clip': ['clip_zs_baseline', 'clip_zs_rag'],
            'blip': ['blip_zs_forced_choice', 'blip_zs_yesno_justification', 'blip_zs_cot', 
                    'blip_fs_yesno_justification', 'blip_zs_rag', 'blip_fs_rag'],
            'llava': ['llava_zs_cot', 'llava_zs_forced_choice', 'llava_fs_cot', 
                     'llava_zs_rag', 'llava_fs_rag'],
            'bert': ['bert_baseline']
        }
        
        # Color scheme for models
        self.model_colors = {
            'CLIP': '#FF6B6B',
            'BLIP2': '#4ECDC4', 
            'LLaVA': '#45B7D1',
            'BERT': '#96CEB4'
        }
        
        # Realistic performance expectations based on actual testing
        self.expected_performance = {
            'CLIP': {'accuracy': 0.55, 'range': '50-60%', 'description': 'Fast but limited reasoning'},
            'BLIP2': {'accuracy': 0.60, 'range': '55-65%', 'description': 'Good balance of speed and reasoning'},
            'LLaVA': {'accuracy': 0.65, 'range': '60-70%', 'description': 'Best reasoning but slower'},
            'BERT': {'accuracy': 0.50, 'range': '45-55%', 'description': 'Text-only baseline'},
            'RAG_improvement': {'accuracy': 0.02, 'range': '1-3%', 'description': 'Small but consistent improvement'}
        }
        
        # Style mapping
        self.style_mapping = {
            'zs': 'Zero-shot',
            'fs': 'Few-shot',
            'rag': 'RAG',
            'baseline': 'Baseline'
        }

    def load_experiment_results(self) -> Dict[str, pd.DataFrame]:
        """Load all experiment results from CSV files."""
        results = {}
        
        for model_type, experiments in self.experiment_categories.items():
            for exp_name in experiments:
                # Look for results in model-specific directories
                possible_paths = [
                    self.results_dir / model_type / exp_name / "all_model_outputs.csv",
                    self.results_dir / model_type / f"{exp_name}/all_model_outputs.csv"
                ]
                
                for path in possible_paths:
                    if path.exists():
                        try:
                            df = pd.read_csv(path)
                            results[exp_name] = df
                            print(f"Loaded: {exp_name} ({len(df)} samples)")
                            break
                        except Exception as e:
                            print(f"Error loading {path}: {e}")
        
        return results

    def extract_metrics_from_results(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Extract metrics from all experiment results."""
        metrics_data = []
        
        for exp_name, df in results.items():
            if df.empty:
                continue
                
            # Determine model type and configuration
            model_type = None
            config = {}
            
            for model, exps in self.experiment_categories.items():
                if exp_name in exps:
                    model_type = model.upper()
                    break
            
            # Parse configuration from experiment name
            if 'zs' in exp_name:
                config['shot_type'] = 'Zero-shot'
            elif 'fs' in exp_name:
                config['shot_type'] = 'Few-shot'
            else:
                config['shot_type'] = 'Baseline'
                
            if 'rag' in exp_name:
                config['rag'] = 'Yes'
            else:
                config['rag'] = 'No'
                
            if 'forced_choice' in exp_name:
                config['prompt_type'] = 'Forced Choice'
            elif 'yesno_justification' in exp_name:
                config['prompt_type'] = 'Yes/No + Justification'
            elif 'cot' in exp_name:
                config['prompt_type'] = 'Chain of Thought'
            else:
                config['prompt_type'] = 'Default'
            
            # Calculate metrics
            if 'true_label' in df.columns and 'predicted_label' in df.columns:
                y_true = df['true_label']
                y_pred = df['predicted_label']
                
                # Basic metrics
                accuracy = (y_true == y_pred).mean()
                
                # Per-class metrics
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                
                # Confusion matrix
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                
                # Additional metrics
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                metrics_data.append({
                    'experiment': exp_name,
                    'model': model_type,
                    'shot_type': config['shot_type'],
                    'rag': config['rag'],
                    'prompt_type': config['prompt_type'],
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'specificity': specificity,
                    'sensitivity': sensitivity,
                    'true_negatives': tn,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'true_positives': tp,
                    'total_samples': len(df)
                })
        
        return pd.DataFrame(metrics_data)

    def create_comparison_table(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Create a comprehensive comparison table."""
        # Sort by model and performance
        metrics_df = metrics_df.sort_values(['model', 'accuracy'], ascending=[True, False])
        
        # Format metrics for display
        display_df = metrics_df.copy()
        for col in ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'sensitivity']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.3f}")
        
        return display_df

    def plot_model_comparison(self, metrics_df: pd.DataFrame):
        """Create model comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        ax1 = axes[0, 0]
        sns.barplot(data=metrics_df, x='model', y='accuracy', ax=ax1, palette=self.model_colors)
        ax1.set_title('Accuracy by Model')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. F1-Score comparison
        ax2 = axes[0, 1]
        sns.barplot(data=metrics_df, x='model', y='f1_score', ax=ax2, palette=self.model_colors)
        ax2.set_title('F1-Score by Model')
        ax2.set_ylabel('F1-Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Zero-shot vs Few-shot comparison
        ax3 = axes[1, 0]
        shot_comparison = metrics_df[metrics_df['shot_type'].isin(['Zero-shot', 'Few-shot'])]
        if not shot_comparison.empty:
            sns.boxplot(data=shot_comparison, x='shot_type', y='accuracy', ax=ax3)
            ax3.set_title('Zero-shot vs Few-shot Performance')
            ax3.set_ylabel('Accuracy')
        
        # 4. RAG vs No RAG comparison
        ax4 = axes[1, 1]
        rag_comparison = metrics_df[metrics_df['rag'].isin(['Yes', 'No'])]
        if not rag_comparison.empty:
            sns.boxplot(data=rag_comparison, x='rag', y='accuracy', ax=ax4)
            ax4.set_title('RAG vs No RAG Performance')
            ax4.set_ylabel('Accuracy')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_detailed_analysis(self, metrics_df: pd.DataFrame):
        """Create detailed analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Precision-Recall scatter
        ax1 = axes[0, 0]
        for model in metrics_df['model'].unique():
            model_data = metrics_df[metrics_df['model'] == model]
            ax1.scatter(model_data['precision'], model_data['recall'], 
                       label=model, s=100, alpha=0.7)
        ax1.set_xlabel('Precision')
        ax1.set_ylabel('Recall')
        ax1.set_title('Precision vs Recall')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Performance heatmap
        ax2 = axes[0, 1]
        pivot_data = metrics_df.pivot_table(
            values='accuracy', 
            index='model', 
            columns='shot_type', 
            aggfunc='mean'
        )
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('Accuracy Heatmap: Model vs Shot Type')
        
        # 3. RAG impact analysis
        ax3 = axes[1, 0]
        rag_impact = metrics_df.groupby(['model', 'rag'])['accuracy'].mean().unstack()
        rag_impact.plot(kind='bar', ax=ax3, color=['#FF6B6B', '#4ECDC4'])
        ax3.set_title('RAG Impact on Accuracy')
        ax3.set_ylabel('Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend(title='RAG')
        
        # 4. Best performing configurations
        ax4 = axes[1, 1]
        top_configs = metrics_df.nlargest(8, 'accuracy')[['experiment', 'accuracy', 'model']]
        bars = ax4.barh(range(len(top_configs)), top_configs['accuracy'])
        ax4.set_yticks(range(len(top_configs)))
        ax4.set_yticklabels([f"{row['model']}: {row['experiment']}" for _, row in top_configs.iterrows()])
        ax4.set_xlabel('Accuracy')
        ax4.set_title('Top 8 Performing Configurations')
        
        # Color bars by model
        for i, (_, row) in enumerate(top_configs.iterrows()):
            bars[i].set_color(self.model_colors.get(row['model'], '#gray'))
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_summary_statistics(self, metrics_df: pd.DataFrame) -> Dict:
        """Generate summary statistics."""
        summary = {
            'total_experiments': len(metrics_df),
            'models_tested': metrics_df['model'].nunique(),
            'best_accuracy': metrics_df['accuracy'].max(),
            'best_experiment': metrics_df.loc[metrics_df['accuracy'].idxmax(), 'experiment'],
            'average_accuracy': metrics_df['accuracy'].mean(),
            'accuracy_std': metrics_df['accuracy'].std(),
            'model_performance': metrics_df.groupby('model')['accuracy'].agg(['mean', 'std', 'max']).to_dict(),
            'rag_impact': {
                'with_rag': metrics_df[metrics_df['rag'] == 'Yes']['accuracy'].mean(),
                'without_rag': metrics_df[metrics_df['rag'] == 'No']['accuracy'].mean()
            },
            'shot_type_performance': metrics_df.groupby('shot_type')['accuracy'].mean().to_dict()
        }
        
        return summary

    def create_final_report(self, metrics_df: pd.DataFrame, summary: Dict):
        """Create a comprehensive final report."""
        report_path = self.reports_dir / 'final_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Multimodal Fact-Checking Project - Final Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Experiments**: {summary['total_experiments']}\n")
            f.write(f"- **Models Tested**: {summary['models_tested']}\n")
            f.write(f"- **Best Accuracy**: {summary['best_accuracy']:.3f} ({summary['best_experiment']})\n")
            f.write(f"- **Average Accuracy**: {summary['average_accuracy']:.3f} (±{summary['accuracy_std']:.3f})\n\n")
            
            f.write("## Realistic Performance Context\n\n")
            f.write("**Note**: Multimodal fact-checking is a very challenging task. Expected performance ranges:\n\n")
            for model, exp in self.expected_performance.items():
                if model != 'RAG_improvement':
                    f.write(f"- **{model}**: {exp['range']} - {exp['description']}\n")
            f.write(f"- **RAG Impact**: {self.expected_performance['RAG_improvement']['range']} improvement\n\n")
            f.write("Results in the 50-70% range indicate realistic evaluation of current model capabilities.\n\n")
            
            f.write("## Model Performance Summary\n\n")
            f.write("| Model | Mean Accuracy | Std Dev | Best Accuracy |\n")
            f.write("|-------|---------------|---------|---------------|\n")
            for model, stats in summary['model_performance'].items():
                f.write(f"| {model} | {stats['mean']:.3f} | {stats['std']:.3f} | {stats['max']:.3f} |\n")
            f.write("\n")
            
            f.write("## RAG Impact Analysis\n\n")
            f.write(f"- **With RAG**: {summary['rag_impact']['with_rag']:.3f}\n")
            f.write(f"- **Without RAG**: {summary['rag_impact']['without_rag']:.3f}\n")
            f.write(f"- **Improvement**: {summary['rag_impact']['with_rag'] - summary['rag_impact']['without_rag']:.3f}\n\n")
            
            f.write("## Shot Type Performance\n\n")
            for shot_type, acc in summary['shot_type_performance'].items():
                f.write(f"- **{shot_type}**: {acc:.3f}\n")
            f.write("\n")
            
            f.write("## Top 5 Performing Configurations\n\n")
            top_5 = metrics_df.nlargest(5, 'accuracy')
            f.write("| Rank | Experiment | Model | Accuracy | Configuration |\n")
            f.write("|------|------------|-------|----------|---------------|\n")
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                config = f"{row['shot_type']}, {row['rag']} RAG"
                f.write(f"| {i} | {row['experiment']} | {row['model']} | {row['accuracy']:.3f} | {config} |\n")
            f.write("\n")
            
            f.write("## Key Findings\n\n")
            f.write("1. **Model Comparison**: [Analysis of relative performance]\n")
            f.write("2. **RAG Effectiveness**: [Impact of retrieval-augmented generation]\n")
            f.write("3. **Prompt Engineering**: [Effect of different prompting strategies]\n")
            f.write("4. **Zero-shot vs Few-shot**: [Learning curve analysis]\n")
            f.write("5. **Error Analysis**: [Common failure patterns]\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("1. **Best Configuration**: [Optimal setup for production]\n")
            f.write("2. **Improvement Areas**: [Identified weaknesses]\n")
            f.write("3. **Future Work**: [Suggested next steps]\n\n")
        
        print(f"Final report saved to: {report_path}")

    def save_metrics_table(self, metrics_df: pd.DataFrame):
        """Save metrics table to CSV."""
        table_path = self.tables_dir / 'comprehensive_metrics.csv'
        metrics_df.to_csv(table_path, index=False)
        print(f"Metrics table saved to: {table_path}")

    def generate_all_results(self):
        """Generate all final results and visualizations."""
        print("=== GENERATING FINAL RESULTS ===")
        
        # Load experiment results
        print("Loading experiment results...")
        results = self.load_experiment_results()
        
        if not results:
            print("No experiment results found!")
            return
        
        # Extract metrics
        print("Extracting metrics...")
        metrics_df = self.extract_metrics_from_results(results)
        
        if metrics_df.empty:
            print("No metrics could be extracted!")
            return
        
        # Generate summary statistics
        print("Generating summary statistics...")
        summary = self.generate_summary_statistics(metrics_df)
        
        # Create visualizations
        print("Creating visualizations...")
        self.plot_model_comparison(metrics_df)
        self.plot_detailed_analysis(metrics_df)
        
        # Save results
        print("Saving results...")
        self.save_metrics_table(metrics_df)
        self.create_final_report(metrics_df, summary)
        
        # Print summary with realistic expectations
        print("\n=== FINAL RESULTS SUMMARY ===")
        print(f"Best Accuracy: {summary['best_accuracy']:.3f} ({summary['best_experiment']})")
        print(f"Average Accuracy: {summary['average_accuracy']:.3f}")
        print(f"RAG Improvement: {summary['rag_impact']['with_rag'] - summary['rag_impact']['without_rag']:.3f}")
        
        # Compare with realistic expectations
        print("\n=== COMPARISON WITH REALISTIC EXPECTATIONS ===")
        for model, exp in self.expected_performance.items():
            if model != 'RAG_improvement':
                print(f"{model}: Expected {exp['range']} - {exp['description']}")
        print(f"RAG Impact: Expected {self.expected_performance['RAG_improvement']['range']} improvement")
        
        print(f"\nResults saved to: {self.final_dir}")
        print("Files generated:")
        print(f"  - {self.figures_dir}/model_comparison.png")
        print(f"  - {self.figures_dir}/detailed_analysis.png")
        print(f"  - {self.tables_dir}/comprehensive_metrics.csv")
        print(f"  - {self.reports_dir}/final_report.md")

if __name__ == "__main__":
    generator = FinalResultsGenerator()
    generator.generate_all_results() 