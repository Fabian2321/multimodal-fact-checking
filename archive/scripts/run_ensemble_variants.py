#!/usr/bin/env python3
"""
Script to run various ensemble combinations to test if we can improve over CLIP alone.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.ensemble_handler import EnsembleHandler
from src.evaluation import calculate_metrics

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_results(file_path):
    """Load model results from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} samples from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

def create_flexible_ensemble(clip_df, other_df, method='weighted_vote', clip_weight=0.7, other_weight=0.3):
    """Create ensemble predictions between CLIP and another model."""
    
    # Extract predictions
    if 'predicted_label' in clip_df.columns:
        clip_preds = clip_df['predicted_label'].values
    elif 'predicted_labels' in clip_df.columns:
        clip_preds = clip_df['predicted_labels'].values
    else:
        # CLIP uses scores, convert to binary predictions
        clip_scores = clip_df['scores'].values
        clip_preds = (clip_scores > 25.0).astype(int)
        
    if 'predicted_label' in other_df.columns:
        other_preds = other_df['predicted_label'].values
    elif 'predicted_labels' in other_df.columns:
        other_preds = other_df['predicted_labels'].values
    else:
        raise ValueError("Other model results must have predicted_label column")
        
    true_labels = clip_df['true_label'].values
    
    if method == 'weighted_vote':
        # Weighted voting with specified weights
        ensemble_preds = []
        for i in range(len(clip_preds)):
            clip_score = clip_weight if clip_preds[i] == 1 else (1 - clip_weight)
            other_score = other_weight if other_preds[i] == 1 else (1 - other_weight)
            
            ensemble_preds.append(1 if (clip_score + other_score) / 2 > 0.5 else 0)
            
    elif method == 'majority_vote':
        # Simple majority vote
        ensemble_preds = []
        for i in range(len(clip_preds)):
            votes = [clip_preds[i], other_preds[i]]
            ensemble_preds.append(1 if sum(votes) > len(votes)/2 else 0)
            
    elif method == 'clip_dominant':
        # Use CLIP as primary
        ensemble_preds = clip_preds.copy()
        
    else:
        raise ValueError(f"Unknown ensemble method: {method}")
    
    # Create ensemble results
    ensemble_df = clip_df.copy()
    ensemble_df['clip_prediction'] = clip_preds
    ensemble_df['other_prediction'] = other_preds
    ensemble_df['ensemble_prediction'] = ensemble_preds
    ensemble_df['ensemble_method'] = method
    
    return ensemble_df

def run_ensemble_experiment(clip_file, other_file, other_name, method, weights, output_dir):
    """Run a single ensemble experiment."""
    try:
        # Load data
        clip_df = load_results(clip_file)
        other_df = load_results(other_file)
        
        if clip_df is None or other_df is None:
            return None
        
        # Ensure same samples
        common_ids = set(clip_df['id']) & set(other_df['id'])
        clip_df = clip_df[clip_df['id'].isin(common_ids)].reset_index(drop=True)
        other_df = other_df[other_df['id'].isin(common_ids)].reset_index(drop=True)
        
        logger.info(f"Using {len(clip_df)} common samples for ensemble")
        
        # Create ensemble
        if method == 'weighted_vote':
            ensemble_df = create_flexible_ensemble(clip_df, other_df, method, weights[0], weights[1])
        else:
            ensemble_df = create_flexible_ensemble(clip_df, other_df, method)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, "ensemble_results.csv")
        ensemble_df.to_csv(results_file, index=False)
        
        # Calculate metrics
        y_true = ensemble_df['true_label'].values
        y_pred = ensemble_df['ensemble_prediction'].values
        
        # Filter out NaN predictions (robust for BLIP)
        valid_mask = (~pd.isna(y_pred)) & (~pd.isna(y_true))
        if valid_mask.sum() > 0:
            y_true_valid = y_true[valid_mask]
            y_pred_valid = y_pred[valid_mask]
            
            metrics = calculate_metrics(y_true_valid, y_pred_valid, average='binary')
            # Fallback for f1/f1_score key
            f1_val = metrics.get('f1', metrics.get('f1_score', None))
            logger.info(f"Ensemble Results for CLIP + {other_name} ({method}):")
            logger.info(f"  Accuracy: {metrics.get('accuracy', -1):.4f}")
            logger.info(f"  Precision: {metrics.get('precision', -1):.4f}")
            logger.info(f"  Recall: {metrics.get('recall', -1):.4f}")
            logger.info(f"  F1: {f1_val if f1_val is not None else 'N/A'}")
            logger.info(f"  Valid predictions: {valid_mask.sum()}/{len(y_true)}")
            
            # For summary, always use 'f1' key
            metrics['f1'] = f1_val
            return metrics
        else:
            logger.error(f"No valid predictions for CLIP + {other_name}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to run ensemble experiment CLIP + {other_name}: {e}")
        return None

def main():
    # Define the best available results files
    results_files = {
        'clip_baseline': 'results/clip/clip_zs_baseline/all_model_outputs.csv',
        'clip_rag': 'results/clip/clip_zs_rag/all_model_outputs.csv',
        'bert_baseline': 'results/bert/bert_baseline/all_model_outputs.csv',
        'bert_rag': 'results/bert/bert_rag/all_model_outputs.csv',
        'blip_baseline': 'results/blip/blip_zs_direct_answer/all_model_outputs.csv',
    }
    
    # Check which files exist
    available_models = {}
    for name, path in results_files.items():
        if os.path.exists(path):
            available_models[name] = path
            logger.info(f"Found results for {name}: {path}")
        else:
            logger.warning(f"Missing results for {name}: {path}")
    
    if not available_models:
        logger.error("No result files found!")
        return
    
    # Define ensemble experiments to run
    experiments = []
    
    # Test CLIP baseline + other models
    if 'clip_baseline' in available_models:
        clip_file = available_models['clip_baseline']
        
        # CLIP + BERT baseline
        if 'bert_baseline' in available_models:
            experiments.append({
                'name': 'clip_bert_baseline_weighted',
                'clip_file': clip_file,
                'other_file': available_models['bert_baseline'],
                'other_name': 'BERT_baseline',
                'method': 'weighted_vote',
                'weights': (0.7, 0.3)
            })
            
            experiments.append({
                'name': 'clip_bert_baseline_majority',
                'clip_file': clip_file,
                'other_file': available_models['bert_baseline'],
                'other_name': 'BERT_baseline',
                'method': 'majority_vote',
                'weights': None
            })
            
            experiments.append({
                'name': 'clip_dominant_bert',
                'clip_file': clip_file,
                'other_file': available_models['bert_baseline'],
                'other_name': 'BERT_baseline',
                'method': 'weighted_vote',
                'weights': (0.8, 0.2)
            })
        
        # CLIP + BLIP baseline
        if 'blip_baseline' in available_models:
            experiments.append({
                'name': 'clip_blip_baseline_weighted',
                'clip_file': clip_file,
                'other_file': available_models['blip_baseline'],
                'other_name': 'BLIP_baseline',
                'method': 'weighted_vote',
                'weights': (0.7, 0.3)
            })
            
            experiments.append({
                'name': 'clip_dominant_blip',
                'clip_file': clip_file,
                'other_file': available_models['blip_baseline'],
                'other_name': 'BLIP_baseline',
                'method': 'weighted_vote',
                'weights': (0.8, 0.2)
            })
    
    # Test CLIP RAG + BERT RAG
    if 'clip_rag' in available_models and 'bert_rag' in available_models:
        experiments.append({
            'name': 'clip_bert_rag_weighted',
            'clip_file': available_models['clip_rag'],
            'other_file': available_models['bert_rag'],
            'other_name': 'BERT_RAG',
            'method': 'weighted_vote',
            'weights': (0.6, 0.4)
        })
    
    logger.info(f"Will test {len(experiments)} ensemble configurations")
    
    # Run all ensemble experiments
    all_results = {}
    
    for exp in experiments:
        logger.info(f"\n--- Running ensemble: {exp['name']} ---")
        output_dir = f"results/ensemble/{exp['name']}"
        
        metrics = run_ensemble_experiment(
            exp['clip_file'], 
            exp['other_file'], 
            exp['other_name'],
            exp['method'], 
            exp['weights'], 
            output_dir
        )
        
        if metrics:
            all_results[exp['name']] = metrics
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("ENSEMBLE EXPERIMENT SUMMARY")
    logger.info("="*60)
    
    for name, metrics in all_results.items():
        logger.info(f"{name:30s} | Acc: {metrics.get('accuracy', -1):.4f} | Prec: {metrics.get('precision', -1):.4f} | Rec: {metrics.get('recall', -1):.4f} | F1: {metrics.get('f1', 'N/A')}")
    
    # Find best ensemble
    if all_results:
        best_ensemble = max(all_results.items(), key=lambda x: x[1]['accuracy'])
        logger.info(f"\nBest ensemble: {best_ensemble[0]} with accuracy {best_ensemble[1]['accuracy']:.4f}")
    
    logger.info("Ensemble experiments completed!")

if __name__ == "__main__":
    main() 