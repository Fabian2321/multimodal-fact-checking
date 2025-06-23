"""
Optimize RAG parameters using grid search.
"""
import os
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import ParameterGrid
from src.utils import setup_logger
from src.rag_handler import RAGHandler, RAGConfig
from src.evaluate_rag import evaluate_rag_performance

logger = setup_logger(__name__)

def load_test_queries(query_path: str) -> List[Dict[str, Any]]:
    """Load test queries from JSON file."""
    with open(query_path, 'r') as f:
        return json.load(f)

def optimize_parameters(
    test_queries: List[Dict[str, Any]],
    knowledge_base_path: str,
    param_grid: Dict[str, List[Any]]
) -> Tuple[Dict[str, Any], float]:
    """Perform grid search to find optimal parameters."""
    best_score = -np.inf
    best_params = None
    
    # Generate all parameter combinations
    param_combinations = list(ParameterGrid(param_grid))
    total_combinations = len(param_combinations)
    
    logger.info(f"Starting parameter optimization with {total_combinations} combinations")
    
    for i, params in enumerate(param_combinations, 1):
        logger.info(f"Testing combination {i}/{total_combinations}: {params}")
        
        # Initialize RAG handler with current parameters
        rag_config = RAGConfig(
            knowledge_base_path=knowledge_base_path,
            **params
        )
        rag_handler = RAGHandler(rag_config)
        
        # Evaluate performance
        metrics = evaluate_rag_performance(rag_handler, test_queries)
        score = metrics["f1_score"]  # Using F1 score as optimization metric
        
        if score > best_score:
            best_score = score
            best_params = params
            logger.info(f"New best parameters found! Score: {score:.4f}")
            logger.info(f"Parameters: {params}")
    
    return best_params, best_score

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimize RAG parameters")
    
    parser.add_argument("--test_queries_path", type=str, required=True,
                      help="Path to test queries JSON file")
    parser.add_argument("--knowledge_base_path", type=str, required=True,
                      help="Path to knowledge base")
    parser.add_argument("--output_path", type=str, default="data/optimization_results",
                      help="Path to save optimization results")
    
    args = parser.parse_args()
    
    # Define parameter grid
    param_grid = {
        "embedding_model": [
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "multi-qa-mpnet-base-dot-v1"
        ],
        "top_k": [3, 5, 7, 10],
        "similarity_threshold": [0.5, 0.6, 0.7, 0.8],
        "rerank_top_n": [5, 10, 15],
        "rerank_threshold": [0.5, 0.6, 0.7]
    }
    
    # Load test queries
    test_queries = load_test_queries(args.test_queries_path)
    
    # Perform optimization
    best_params, best_score = optimize_parameters(
        test_queries,
        args.knowledge_base_path,
        param_grid
    )
    
    # Save results
    os.makedirs(args.output_path, exist_ok=True)
    results = {
        "best_parameters": best_params,
        "best_score": best_score,
        "parameter_grid": param_grid
    }
    
    results_path = os.path.join(args.output_path, "optimization_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Optimization complete. Results saved to {results_path}")
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best score: {best_score:.4f}")

if __name__ == "__main__":
    main() 