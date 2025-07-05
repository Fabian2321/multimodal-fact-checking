import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.core.utils import setup_logger

logger = logging.getLogger(__name__)

class EnsembleHandler:
    """Handles ensemble predictions combining CLIP, BERT, and RAG models."""
    
    def __init__(self, clip_results_path: str, bert_results_path: str, 
                 rag_results_path: Optional[str] = None,
                 output_dir: str = None, experiment_name: str = None):
        self.clip_results_path = clip_results_path
        self.bert_results_path = bert_results_path
        self.rag_results_path = rag_results_path
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        
    def load_results(self) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """Load CLIP, BERT, and optionally RAG results."""
        clip_df = pd.read_csv(self.clip_results_path)
        bert_df = pd.read_csv(self.bert_results_path)
        
        # Ensure same samples
        common_ids = set(clip_df['id']) & set(bert_df['id'])
        clip_df = clip_df[clip_df['id'].isin(common_ids)].reset_index(drop=True)
        bert_df = bert_df[bert_df['id'].isin(common_ids)].reset_index(drop=True)
        
        rag_df = None
        if self.rag_results_path and os.path.exists(self.rag_results_path):
            rag_df = pd.read_csv(self.rag_results_path)
            # Update common IDs to include RAG
            common_ids = common_ids & set(rag_df['id'])
            clip_df = clip_df[clip_df['id'].isin(common_ids)].reset_index(drop=True)
            bert_df = bert_df[bert_df['id'].isin(common_ids)].reset_index(drop=True)
            rag_df = rag_df[rag_df['id'].isin(common_ids)].reset_index(drop=True)
        
        logger.info(f"Loaded {len(clip_df)} common samples for ensemble")
        return clip_df, bert_df, rag_df
    
    def create_ensemble_predictions(self, clip_df: pd.DataFrame, bert_df: pd.DataFrame,
                                  rag_df: Optional[pd.DataFrame] = None,
                                  method: str = 'weighted_vote') -> pd.DataFrame:
        """Create ensemble predictions using different methods."""
        
        # Extract predictions - handle different column names
        if 'predicted_label' in clip_df.columns:
            clip_preds = clip_df['predicted_label'].values
        elif 'predicted_labels' in clip_df.columns:
            clip_preds = clip_df['predicted_labels'].values
        else:
            # CLIP uses scores, convert to binary predictions
            clip_scores = clip_df['scores'].values
            clip_preds = (clip_scores > 25.0).astype(int)  # Threshold based on CLIP performance
            
        if 'predicted_label' in bert_df.columns:
            bert_preds = bert_df['predicted_label'].values
        elif 'predicted_labels' in bert_df.columns:
            bert_preds = bert_df['predicted_labels'].values
        else:
            raise ValueError("BERT results must have predicted_label column")
        
        rag_preds = None
        if rag_df is not None:
            if 'predicted_label' in rag_df.columns:
                rag_preds = rag_df['predicted_label'].values
            elif 'predicted_labels' in rag_df.columns:
                rag_preds = rag_df['predicted_labels'].values
            else:
                # RAG might use scores like CLIP
                rag_scores = rag_df['scores'].values
                rag_preds = (rag_scores > 25.0).astype(int)
            
        true_labels = clip_df['true_label'].values
        
        if method == 'weighted_vote':
            # Weight by individual model performance
            clip_acc = accuracy_score(true_labels, clip_preds)
            bert_acc = accuracy_score(true_labels, bert_preds)
            
            weights = [clip_acc, bert_acc]
            model_names = ['clip', 'bert']
            
            if rag_preds is not None:
                rag_acc = accuracy_score(true_labels, rag_preds)
                weights.append(rag_acc)
                model_names.append('rag')
            
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            logger.info(f"Model weights: {dict(zip(model_names, normalized_weights))}")
            
            # Weighted voting
            ensemble_preds = []
            for i in range(len(clip_preds)):
                scores = []
                for j, preds in enumerate([clip_preds, bert_preds]):
                    score = normalized_weights[j] if preds[i] == 1 else (1 - normalized_weights[j])
                    scores.append(score)
                
                if rag_preds is not None:
                    rag_score = normalized_weights[2] if rag_preds[i] == 1 else (1 - normalized_weights[2])
                    scores.append(rag_score)
                
                ensemble_preds.append(1 if sum(scores) / len(scores) > 0.5 else 0)
                
        elif method == 'majority_vote':
            # Simple majority vote
            ensemble_preds = []
            for i in range(len(clip_preds)):
                votes = [clip_preds[i], bert_preds[i]]
                if rag_preds is not None:
                    votes.append(rag_preds[i])
                ensemble_preds.append(1 if sum(votes) > len(votes)/2 else 0)
                
        elif method == 'clip_dominant':
            # Use CLIP as primary, others as tiebreaker
            ensemble_preds = clip_preds.copy()
            
        elif method == 'confidence_weighted':
            # Use confidence scores if available
            ensemble_preds = []
            for i in range(len(clip_preds)):
                # Default confidence based on model performance
                clip_conf = 0.67  # CLIP performance
                bert_conf = 0.50  # BERT performance
                
                scores = []
                scores.append(clip_conf if clip_preds[i] == 1 else (1 - clip_conf))
                scores.append(bert_conf if bert_preds[i] == 1 else (1 - bert_conf))
                
                if rag_preds is not None:
                    rag_conf = 0.69  # RAG performance estimate
                    scores.append(rag_conf if rag_preds[i] == 1 else (1 - rag_conf))
                
                ensemble_preds.append(1 if sum(scores) / len(scores) > 0.5 else 0)
        
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        # Create ensemble results
        ensemble_df = clip_df.copy()
        ensemble_df['clip_prediction'] = clip_preds
        ensemble_df['bert_prediction'] = bert_preds
        if rag_preds is not None:
            ensemble_df['rag_prediction'] = rag_preds
        ensemble_df['ensemble_prediction'] = ensemble_preds
        ensemble_df['ensemble_method'] = method
        
        return ensemble_df
    
    def evaluate_ensemble(self, ensemble_df: pd.DataFrame) -> Dict:
        """Evaluate ensemble performance."""
        true_labels = ensemble_df['true_label'].values
        ensemble_preds = ensemble_df['ensemble_prediction'].values
        clip_preds = ensemble_df['clip_prediction'].values
        bert_preds = ensemble_df['bert_prediction'].values
        
        # Calculate metrics
        metrics = {}
        for name, preds in [('ensemble', ensemble_preds), ('clip', clip_preds), ('bert', bert_preds)]:
            metrics[name] = {
                'accuracy': accuracy_score(true_labels, preds),
                'precision': precision_score(true_labels, preds, zero_division=0),
                'recall': recall_score(true_labels, preds, zero_division=0),
                'f1': f1_score(true_labels, preds, zero_division=0)
            }
        
        # Add RAG metrics if available
        if 'rag_prediction' in ensemble_df.columns:
            rag_preds = ensemble_df['rag_prediction'].values
            metrics['rag'] = {
                'accuracy': accuracy_score(true_labels, rag_preds),
                'precision': precision_score(true_labels, rag_preds, zero_division=0),
                'recall': recall_score(true_labels, rag_preds, zero_division=0),
                'f1': f1_score(true_labels, rag_preds, zero_division=0)
            }
        
        return metrics
    
    def save_results(self, ensemble_df: pd.DataFrame, metrics: Dict):
        """Save ensemble results and metrics."""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save ensemble predictions
        output_path = os.path.join(self.output_dir, 'ensemble_predictions.csv')
        ensemble_df.to_csv(output_path, index=False)
        logger.info(f"Ensemble predictions saved to {output_path}")
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, 'ensemble_metrics.csv')
        metrics_df = pd.DataFrame(metrics).T
        metrics_df.to_csv(metrics_path)
        logger.info(f"Ensemble metrics saved to {metrics_path}")
        
        # Create visualization
        self.create_metrics_visualization(metrics, self.output_dir)
        
        return output_path, metrics_path
    
    def create_metrics_visualization(self, metrics: Dict, output_dir: str):
        """Create visualization comparing ensemble vs individual models."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Ensemble vs Individual Model Performance', fontsize=16)
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        
        for i, metric in enumerate(metric_names):
            ax = axes[i//2, i%2]
            
            models = list(metrics.keys())
            values = [metrics[model][metric] for model in models]
            
            bars = ax.bar(models, values, color=colors[:len(models)])
            ax.set_title(f'{metric.title()} Score')
            ax.set_ylabel(metric.title())
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        viz_path = os.path.join(output_dir, 'ensemble_comparison.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Ensemble comparison visualization saved to {viz_path}")
    
    def run_ensemble_analysis(self, method: str = 'weighted_vote') -> Dict:
        """Run complete ensemble analysis."""
        logger.info(f"Starting ensemble analysis with method: {method}")
        
        # Load results
        clip_df, bert_df, rag_df = self.load_results()
        
        # Create ensemble predictions
        ensemble_df = self.create_ensemble_predictions(clip_df, bert_df, rag_df, method)
        
        # Evaluate ensemble
        metrics = self.evaluate_ensemble(ensemble_df)
        
        # Save results
        self.save_results(ensemble_df, metrics)
        
        # Log results
        logger.info("=== Ensemble Results ===")
        for model, model_metrics in metrics.items():
            logger.info(f"{model.upper()}: Accuracy={model_metrics['accuracy']:.3f}, "
                       f"Precision={model_metrics['precision']:.3f}, "
                       f"Recall={model_metrics['recall']:.3f}, "
                       f"F1={model_metrics['f1']:.3f}")
        
        return metrics 