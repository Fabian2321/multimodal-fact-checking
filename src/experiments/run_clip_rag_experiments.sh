#!/bin/bash

# CLIP RAG Experiments Script
# Tests CLIP with RAG-enhanced prompts for text-image matching

set -e

echo "🚀 Starting CLIP RAG Experiments"
echo "=================================="

# Configuration
MODEL_TYPE="clip"
CLIP_MODEL="openai/clip-vit-base-patch32"
DATA_DIR="data"
METADATA_FILE="test_balanced_pairs_clean.csv"
TEXT_COL="clean_title"
LABEL_COL="2_way_label"
BATCH_SIZE=32
NUM_SAMPLES=100
OUTPUT_DIR="results"
RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K=3
RAG_SIMILARITY_THRESHOLD=0.7
RAG_KB_PATH="src/data/knowledge_base"
RAG_INITIAL_DOCS="src/data/knowledge_base/documents.json"

# Create output directory
mkdir -p "${OUTPUT_DIR}/${MODEL_TYPE}"

echo "📊 Running CLIP Zero-Shot Baseline (no RAG)"
python -m src.pipeline \
    --model_type ${MODEL_TYPE} \
    --clip_model_name ${CLIP_MODEL} \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name "clip_zeroshot_baseline" \
    --num_workers 4

echo "🔍 Running CLIP with RAG Enhanced Prompt"
python -m src.pipeline \
    --model_type ${MODEL_TYPE} \
    --clip_model_name ${CLIP_MODEL} \
    --prompt_name "clip_rag_enhanced" \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name "clip_rag_enhanced" \
    --use_rag \
    --rag_embedding_model ${RAG_EMBEDDING_MODEL} \
    --rag_top_k ${RAG_TOP_K} \
    --rag_similarity_threshold ${RAG_SIMILARITY_THRESHOLD} \
    --rag_knowledge_base_path ${RAG_KB_PATH} \
    --rag_initial_docs ${RAG_INITIAL_DOCS} \
    --num_workers 4

echo "✅ Running CLIP with RAG Fact-Check Prompt"
python -m src.pipeline \
    --model_type ${MODEL_TYPE} \
    --clip_model_name ${CLIP_MODEL} \
    --prompt_name "clip_rag_fact_check" \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name "clip_rag_fact_check" \
    --use_rag \
    --rag_embedding_model ${RAG_EMBEDDING_MODEL} \
    --rag_top_k ${RAG_TOP_K} \
    --rag_similarity_threshold ${RAG_SIMILARITY_THRESHOLD} \
    --rag_knowledge_base_path ${RAG_KB_PATH} \
    --rag_initial_docs ${RAG_INITIAL_DOCS} \
    --num_workers 4

echo "📈 Running CLIP with RAG + Metadata"
python -m src.pipeline \
    --model_type ${MODEL_TYPE} \
    --clip_model_name ${CLIP_MODEL} \
    --prompt_name "clip_rag_metadata" \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name "clip_rag_metadata" \
    --use_rag \
    --rag_embedding_model ${RAG_EMBEDDING_MODEL} \
    --rag_top_k ${RAG_TOP_K} \
    --rag_similarity_threshold ${RAG_SIMILARITY_THRESHOLD} \
    --rag_knowledge_base_path ${RAG_KB_PATH} \
    --rag_initial_docs ${RAG_INITIAL_DOCS} \
    --num_workers 4

echo "🎯 Running CLIP with Standard Fact-Check Prompt (no RAG)"
python -m src.pipeline \
    --model_type ${MODEL_TYPE} \
    --clip_model_name ${CLIP_MODEL} \
    --prompt_name "clip_similarity_fact_check" \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name "clip_fact_check_baseline" \
    --num_workers 4

echo "✅ All CLIP RAG experiments completed!"
echo "📁 Results saved in: ${OUTPUT_DIR}/${MODEL_TYPE}/"
echo ""
echo "🔍 Next steps:"
echo "1. Compare results between baseline and RAG-enhanced versions"
echo "2. Analyze which RAG prompt performs best"
echo "3. Consider ensemble methods with BLIP2 and LLaVA"
echo "4. Fine-tune RAG parameters (top_k, similarity threshold)" 