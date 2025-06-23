#!/usr/bin/env python3
"""
Script to run parameter optimization for a specific embedding model chunk.
"""
import os
import json
import time
import psutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from src.optimize_rag_params import optimize_parameters, load_test_queries

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('optimization_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024  # Convert to GB

def save_checkpoint(results_dir, current_params, best_params, best_score, completed_combinations, model_name):
    """Save current optimization state."""
    checkpoint = {
        'current_params': current_params,
        'best_params': best_params,
        'best_score': best_score,
        'completed_combinations': completed_combinations,
        'timestamp': datetime.now().isoformat(),
        'model_name': model_name
    }
    
    checkpoint_path = os.path.join(results_dir, f'optimization_checkpoint_{model_name}.json')
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

def load_checkpoint(results_dir, model_name):
    """Load previous optimization state if exists."""
    checkpoint_path = os.path.join(results_dir, f'optimization_checkpoint_{model_name}.json')
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None

def main():
    parser = argparse.ArgumentParser(description="Run parameter optimization for a specific embedding model")
    
    parser.add_argument("--test_queries_path", type=str, required=True,
                      help="Path to test queries JSON file")
    parser.add_argument("--knowledge_base_path", type=str, required=True,
                      help="Path to knowledge base")
    parser.add_argument("--output_path", type=str, default="data/optimization_results",
                      help="Path to save optimization results")
    parser.add_argument("--memory_threshold", type=float, default=0.8,
                      help="Memory threshold (0-1) at which to pause optimization")
    parser.add_argument("--pause_duration", type=int, default=600,
                      help="Duration in seconds to pause when memory threshold is reached")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name of the embedding model to optimize")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Define parameter grid for the specific model
    param_grid = {
        "embedding_model": [args.model_name],
        "top_k": [3, 5, 7, 10],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8],
        "rerank_top_n": [5, 10, 15],
        "rerank_threshold": [0.5, 0.6, 0.7]
    }
    
    # Load test queries
    logger.info(f"Loading test queries for model {args.model_name}...")
    test_queries = load_test_queries(args.test_queries_path)
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(args.output_path, args.model_name)
    if checkpoint:
        logger.info(f"Found checkpoint for {args.model_name}, resuming from previous state")
        best_params = checkpoint['best_params']
        best_score = checkpoint['best_score']
        completed_combinations = set(checkpoint['completed_combinations'])
    else:
        logger.info(f"No checkpoint found for {args.model_name}, starting fresh")
        best_params = None
        best_score = float('-inf')
        completed_combinations = set()
    
    # Generate all parameter combinations
    from sklearn.model_selection import ParameterGrid
    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    
    logger.info(f"Starting parameter optimization for {args.model_name} with {total_combinations} combinations")
    logger.info(f"Memory threshold set to {args.memory_threshold * 100}%")
    
    try:
        for i, params in enumerate(param_combinations, 1):
            # Skip already completed combinations
            if str(params) in completed_combinations:
                logger.info(f"Skipping already completed combination {i}/{total_combinations}")
                continue
            
            # Check memory usage
            memory_usage = get_memory_usage()
            if memory_usage > args.memory_threshold:
                logger.warning(f"Memory usage ({memory_usage:.2f}GB) above threshold, pausing for {args.pause_duration}s")
                time.sleep(args.pause_duration)
            
            logger.info(f"Testing combination {i}/{total_combinations}: {params}")
            
            # Initialize RAG handler with current parameters
            from rag_handler import RAGHandler, RAGConfig
            rag_config = RAGConfig(
                knowledge_base_path=args.knowledge_base_path,
                **params
            )
            rag_handler = RAGHandler(rag_config)
            
            # Evaluate performance
            from evaluate_rag import evaluate_rag_performance
            metrics = evaluate_rag_performance(rag_handler, test_queries)
            score = metrics["f1_score"]
            
            if score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best parameters found! Score: {score:.4f}")
                logger.info(f"Parameters: {params}")
            
            # Mark combination as completed
            completed_combinations.add(str(params))
            
            # Save checkpoint after EVERY combination for safety
            save_checkpoint(args.output_path, params, best_params, best_score, list(completed_combinations), args.model_name)
            
            # Log progress
            logger.info(f"Progress: {i}/{total_combinations} combinations completed")
            logger.info(f"Current memory usage: {get_memory_usage():.2f}GB")
    
    except Exception as e:
        logger.error(f"Error during optimization: {e}", exc_info=True)
        # Save checkpoint on error
        save_checkpoint(args.output_path, params, best_params, best_score, list(completed_combinations), args.model_name)
        raise
    
    # Save final results
    results = {
        "model_name": args.model_name,
        "best_parameters": best_params,
        "best_score": best_score,
        "parameter_grid": param_grid,
        "completed_combinations": len(completed_combinations),
        "total_combinations": total_combinations
    }
    
    results_path = os.path.join(args.output_path, f"optimization_results_{args.model_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Optimization complete for {args.model_name}. Results saved to {results_path}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score: {best_score:.4f}")

if __name__ == "__main__":
    main() 