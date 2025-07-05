#!/bin/bash

# CLIP + BERT Ensemble Experiments Script
# Tests ensemble methods combining CLIP and BERT for text-image matching

set -e

echo "🚀 Starting CLIP + BERT Ensemble Experiments"
echo "============================================="

# Configuration
DATA_DIR="data"
METADATA_FILE="test_balanced_pairs_clean.csv"
TEXT_COL="clean_title"
LABEL_COL="2_way_label"
BATCH_SIZE=32
NUM_SAMPLES=100
OUTPUT_DIR="results"
CLIP_MODEL="openai/clip-vit-base-patch32"
BERT_MODEL="bert-base-uncased"

# Create output directory
mkdir -p "${OUTPUT_DIR}/ensemble"

echo "📊 Step 1: Running CLIP Zero-Shot Baseline"
python -m src.pipeline \
    --model_type clip \
    --clip_model_name ${CLIP_MODEL} \
    --prompt_name clip_similarity_fact_check \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name clip_ensemble_baseline \
    --num_workers 2

echo "📊 Step 2: Running BERT Text-Only Baseline"
python -m src.pipeline \
    --model_type bert \
    --bert_model_name ${BERT_MODEL} \
    --prompt_name bert_fact_check \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name bert_ensemble_baseline \
    --num_workers 2

echo "🔗 Step 3: Creating CLIP + BERT Ensemble"
python -c "
import sys
sys.path.append('src')
from ensemble_handler import EnsembleHandler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize ensemble handler
ensemble_handler = EnsembleHandler(
    clip_results_path='${OUTPUT_DIR}/clip/clip_ensemble_baseline/all_model_outputs.csv',
    bert_results_path='${OUTPUT_DIR}/bert/bert_ensemble_baseline/all_model_outputs.csv',
    output_dir='${OUTPUT_DIR}/ensemble/clip_bert_ensemble',
    experiment_name='clip_bert_ensemble'
)

# Test different ensemble methods
methods = ['weighted_vote', 'majority_vote', 'clip_dominant']

for method in methods:
    print(f'\\n🎯 Testing ensemble method: {method}')
    metrics = ensemble_handler.run_ensemble_analysis(method)
    
    # Print best results
    ensemble_acc = metrics['ensemble']['accuracy']
    clip_acc = metrics['clip']['accuracy']
    bert_acc = metrics['bert']['accuracy']
    
    print(f'📈 Results for {method}:')
    print(f'   Ensemble: {ensemble_acc:.3f}')
    print(f'   CLIP:     {clip_acc:.3f}')
    print(f'   BERT:     {bert_acc:.3f}')
    print(f'   Improvement: {ensemble_acc - max(clip_acc, bert_acc):.3f}')
"

echo "✅ CLIP + BERT Ensemble experiments completed!"
echo "📁 Results saved in: ${OUTPUT_DIR}/ensemble/"

echo ""
echo "🔍 Summary of Ensemble Methods:"
echo "1. weighted_vote: Weighted by individual model performance"
echo "2. majority_vote: Simple majority voting"
echo "3. clip_dominant: CLIP as primary, BERT as tiebreaker"

echo ""
echo "📈 Next steps:"
echo "1. Analyze which ensemble method performs best"
echo "2. Fine-tune ensemble weights"
echo "3. Test with RAG-enhanced CLIP"
echo "4. Consider adding LLaVA to the ensemble" 