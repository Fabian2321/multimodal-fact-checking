"""
Script to evaluate RAG performance and optimize parameters.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.core.utils import setup_logger
from src.models.rag_handler import RAGHandler, RAGConfig

logger = setup_logger(__name__)

def evaluate_retrieval(
    rag_handler: RAGHandler,
    test_queries: List[Dict[str, Any]],
    metrics: List[str] = ["precision", "recall", "f1"]
) -> Dict[str, float]:
    """Evaluate retrieval performance."""
    results = {metric: [] for metric in metrics}
    
    for query in test_queries:
        # Get retrieved documents
        retrieved_docs = rag_handler.retrieve(query["text"])
        
        # Calculate metrics
        if "precision" in metrics:
            relevant_retrieved = sum(1 for doc in retrieved_docs 
                                   if doc["document"]["label"] == query["label"])
            precision = relevant_retrieved / len(retrieved_docs) if retrieved_docs else 0
            results["precision"].append(precision)
        
        if "recall" in metrics:
            total_relevant = sum(1 for doc in rag_handler.documents 
                               if doc["label"] == query["label"])
            relevant_retrieved = sum(1 for doc in retrieved_docs 
                                   if doc["document"]["label"] == query["label"])
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
            results["recall"].append(recall)
        
        if "f1" in metrics:
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            results["f1"].append(f1)
    
    # Calculate average metrics
    return {metric: np.mean(values) for metric, values in results.items()}

def optimize_parameters(
    rag_handler: RAGHandler,
    test_queries: List[Dict[str, Any]],
    param_grid: Dict[str, List[Any]]
) -> Dict[str, Any]:
    """Optimize RAG parameters using grid search."""
    best_score = 0
    best_params = {}
    
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v)) 
                         for v in np.array(np.meshgrid(*param_grid.values())).T.reshape(-1, len(param_grid))]
    
    for params in param_combinations:
        # Update RAG handler with new parameters
        rag_handler.config.top_k = params["top_k"]
        rag_handler.config.similarity_threshold = params["similarity_threshold"]
        
        # Evaluate with current parameters
        scores = evaluate_retrieval(rag_handler, test_queries)
        current_score = scores["f1"]  # Use F1 score as optimization metric
        
        if current_score > best_score:
            best_score = current_score
            best_params = params
    
    return best_params

def evaluate_rag_performance(rag_handler: RAGHandler, test_queries: List[Dict[str, Any]]) -> Dict[str, float]:
    """Evaluate RAG system performance on test queries."""
    total_queries = len(test_queries)
    correct_predictions = 0
    total_relevant_docs = 0
    total_retrieved_docs = 0
    
    for query in test_queries:
        # Get ground truth
        expected_label = query["label"]
        query_text = query["text"]
        
        # Get RAG results
        results = rag_handler.query(query_text)
        
        # Count relevant documents
        relevant_docs = sum(1 for doc in results if doc["label"] == expected_label)
        total_relevant_docs += relevant_docs
        
        # Count retrieved documents
        total_retrieved_docs += len(results)
        
        # Check if majority of retrieved documents match expected label
        if len(results) > 0:
            majority_label = max(set(doc["label"] for doc in results), 
                               key=lambda x: sum(1 for doc in results if doc["label"] == x))
            if majority_label == expected_label:
                correct_predictions += 1
    
    # Calculate metrics
    precision = total_relevant_docs / total_retrieved_docs if total_retrieved_docs > 0 else 0
    recall = total_relevant_docs / total_queries if total_queries > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = correct_predictions / total_queries if total_queries > 0 else 0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "total_queries": total_queries,
        "correct_predictions": correct_predictions,
        "total_relevant_docs": total_relevant_docs,
        "total_retrieved_docs": total_retrieved_docs
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate and optimize RAG performance")
    
    parser.add_argument("--test_queries_path", type=str, required=True,
                      help="Path to test queries JSON file")
    parser.add_argument("--knowledge_base_path", type=str, required=True,
                      help="Path to RAG knowledge base")
    parser.add_argument("--output_path", type=str, default="results/rag_evaluation",
                      help="Output path for evaluation results")
    parser.add_argument("--optimize", action="store_true",
                      help="Run parameter optimization")
    
    args = parser.parse_args()
    
    # Load test queries
    with open(args.test_queries_path, 'r') as f:
        test_queries = json.load(f)
    
    # Initialize RAG handler
    rag_config = RAGConfig(knowledge_base_path=args.knowledge_base_path)
    rag_handler = RAGHandler(rag_config)
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Evaluate current performance
    logger.info("Evaluating current RAG performance")
    current_metrics = evaluate_retrieval(rag_handler, test_queries)
    
    # Save current metrics
    with open(os.path.join(args.output_path, "current_metrics.json"), 'w') as f:
        json.dump(current_metrics, f, indent=2)
    
    # Optimize parameters if requested
    if args.optimize:
        logger.info("Optimizing RAG parameters")
        param_grid = {
            "top_k": [1, 3, 5, 7, 10],
            "similarity_threshold": [0.5, 0.6, 0.7, 0.8, 0.9]
        }
        
        best_params = optimize_parameters(rag_handler, test_queries, param_grid)
        
        # Evaluate with best parameters
        rag_handler.config.top_k = best_params["top_k"]
        rag_handler.config.similarity_threshold = best_params["similarity_threshold"]
        best_metrics = evaluate_retrieval(rag_handler, test_queries)
        
        # Save optimization results
        optimization_results = {
            "best_parameters": best_params,
            "best_metrics": best_metrics
        }
        with open(os.path.join(args.output_path, "optimization_results.json"), 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best metrics: {best_metrics}")

    # Evaluate RAG system performance
    system_metrics = evaluate_rag_performance(rag_handler, test_queries)
    
    # Save system metrics
    with open(os.path.join(args.output_path, "system_metrics.json"), 'w') as f:
        json.dump(system_metrics, f, indent=2)

if __name__ == "__main__":
    main() 