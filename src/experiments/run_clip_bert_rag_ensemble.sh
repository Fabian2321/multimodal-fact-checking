#!/bin/bash

# CLIP + BERT + RAG Ensemble Experiments Script
# Tests ensemble methods combining CLIP, BERT, and RAG for text-image matching

set -e

echo "🚀 Starting CLIP + BERT + RAG Ensemble Experiments"
echo "=================================================="

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
    --experiment_name clip_ensemble_3model \
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
    --experiment_name bert_ensemble_3model \
    --num_workers 2

echo "📊 Step 3: Running CLIP with RAG"
python -m src.pipeline \
    --model_type clip \
    --clip_model_name ${CLIP_MODEL} \
    --prompt_name clip_rag_fact_check \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name rag_ensemble_3model \
    --use_rag \
    --rag_embedding_model sentence-transformers/all-MiniLM-L6-v2 \
    --rag_top_k 3 \
    --rag_similarity_threshold 0.7 \
    --rag_knowledge_base_path src/data/knowledge_base \
    --rag_initial_docs src/data/knowledge_base/documents.json \
    --num_workers 2

echo "🔗 Step 4: Creating CLIP + BERT + RAG Ensemble"
python -c "
import sys
sys.path.append('src')
from ensemble_handler import EnsembleHandler
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize ensemble handler
ensemble_handler = EnsembleHandler(
    clip_results_path='${OUTPUT_DIR}/clip/clip_ensemble_3model/all_model_outputs.csv',
    bert_results_path='${OUTPUT_DIR}/bert/bert_ensemble_3model/all_model_outputs.csv',
    rag_results_path='${OUTPUT_DIR}/clip/rag_ensemble_3model/all_model_outputs.csv',
    output_dir='${OUTPUT_DIR}/ensemble/clip_bert_rag_ensemble',
    experiment_name='clip_bert_rag_ensemble'
)

# Test different ensemble methods
methods = ['weighted_vote', 'majority_vote', 'clip_dominant', 'confidence_weighted']

for method in methods:
    print(f'\\n🎯 Testing ensemble method: {method}')
    metrics = ensemble_handler.run_ensemble_analysis(method)
    
    # Print best results
    ensemble_acc = metrics['ensemble']['accuracy']
    clip_acc = metrics['clip']['accuracy']
    bert_acc = metrics['bert']['accuracy']
    rag_acc = metrics['rag']['accuracy']
    
    print(f'📈 Results for {method}:')
    print(f'   Ensemble: {ensemble_acc:.3f}')
    print(f'   CLIP:     {clip_acc:.3f}')
    print(f'   BERT:     {bert_acc:.3f}')
    print(f'   RAG:      {rag_acc:.3f}')
    
    # Calculate improvement over best individual model
    best_individual = max(clip_acc, bert_acc, rag_acc)
    improvement = ensemble_acc - best_individual
    print(f'   Improvement: {improvement:.3f} (over best individual: {best_individual:.3f})')
"

echo "✅ CLIP + BERT + RAG Ensemble experiments completed!"
echo "📁 Results saved in: ${OUTPUT_DIR}/ensemble/"

echo ""
echo "🔍 Summary of Ensemble Methods:"
echo "1. weighted_vote: Weighted by individual model performance"
echo "2. majority_vote: Simple majority voting (3 models)"
echo "3. clip_dominant: CLIP as primary, others as tiebreaker"
echo "4. confidence_weighted: Fixed confidence weights based on expected performance"

echo ""
echo "📈 Expected Benefits:"
echo "- CLIP: Visual-linguistic similarity"
echo "- BERT: Text-based semantics and context"
echo "- RAG: Fact-checking knowledge and guidelines"
echo "- Ensemble: Combines complementary strengths"

echo ""
echo "🚀 Next steps:"
echo "1. Analyze which ensemble method performs best"
echo "2. Fine-tune ensemble weights"
echo "3. Test with different RAG parameters"
echo "4. Consider adding LLaVA for 4-model ensemble" 