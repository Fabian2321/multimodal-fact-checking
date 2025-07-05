#!/bin/bash

# Ensemble Experiments Script
# Combines CLIP and BERT predictions with different methods

set -e

# Configuration
DATA_DIR="data"
METADATA_FILE="test_balanced_pairs.csv"
NUM_SAMPLES=200
BATCH_SIZE=8
NUM_WORKERS=2
OUTPUT_DIR="results"

echo "=== ENSEMBLE EXPERIMENTS ==="
echo "Running CLIP + BERT ensemble experiments..."

# Function to run ensemble analysis
run_ensemble_analysis() {
    local clip_experiment=$1
    local bert_experiment=$2
    local ensemble_name=$3
    local method=$4
    
    echo "Running ensemble: $ensemble_name (method: $method)"
    
    # Paths to results
    clip_results="results/clip/$clip_experiment/all_model_outputs.csv"
    bert_results="results/bert/$bert_experiment/all_model_outputs.csv"
    ensemble_output="results/ensemble/$ensemble_name"
    
    # Check if both results exist
    if [ ! -f "$clip_results" ]; then
        echo "ERROR: CLIP results not found: $clip_results"
        return 1
    fi
    
    if [ ! -f "$bert_results" ]; then
        echo "ERROR: BERT results not found: $bert_results"
        return 1
    fi
    
    # Run ensemble analysis
    PYTHONPATH=. python -c "
from src.ensemble_handler import EnsembleHandler
import logging

logging.basicConfig(level=logging.INFO)
handler = EnsembleHandler('$clip_results', '$bert_results', '$ensemble_output', '$ensemble_name')
metrics = handler.run_ensemble_analysis('$method')
print(f'Ensemble {ensemble_name} completed successfully!')
"
    
    echo "Completed: $ensemble_name"
    echo "---"
}

# 1. CLIP Baseline + BERT Baseline Ensemble
echo "=== 1. CLIP Baseline + BERT Baseline Ensemble ==="
run_ensemble_analysis "clip_zs_baseline" "bert_baseline" "clip_bert_baseline" "weighted_vote"

# 2. CLIP Baseline + BERT RAG Ensemble  
echo "=== 2. CLIP Baseline + BERT RAG Ensemble ==="
run_ensemble_analysis "clip_zs_baseline" "bert_rag" "clip_bert_rag" "weighted_vote"

# 3. CLIP RAG + BERT Baseline Ensemble
echo "=== 3. CLIP RAG + BERT Baseline Ensemble ==="
run_ensemble_analysis "clip_zs_rag" "bert_baseline" "clip_rag_bert_baseline" "weighted_vote"

# 4. CLIP RAG + BERT RAG Ensemble
echo "=== 4. CLIP RAG + BERT RAG Ensemble ==="
run_ensemble_analysis "clip_zs_rag" "bert_rag" "clip_rag_bert_rag" "weighted_vote"

# 5. Additional ensemble methods for best combination
echo "=== 5. Different Ensemble Methods ==="
run_ensemble_analysis "clip_zs_baseline" "bert_rag" "clip_bert_rag_majority" "majority_vote"
run_ensemble_analysis "clip_zs_baseline" "bert_rag" "clip_bert_rag_clip_dominant" "clip_dominant"

echo "=== ENSEMBLE EXPERIMENTS COMPLETED ==="
echo "Results saved in results/ensemble/"
echo ""
echo "Summary of ensemble experiments:"
echo "1. clip_bert_baseline: CLIP Baseline + BERT Baseline"
echo "2. clip_bert_rag: CLIP Baseline + BERT RAG"
echo "3. clip_rag_bert_baseline: CLIP RAG + BERT Baseline"
echo "4. clip_rag_bert_rag: CLIP RAG + BERT RAG"
echo "5. clip_bert_rag_majority: Majority vote method"
echo "6. clip_bert_rag_clip_dominant: CLIP dominant method" 