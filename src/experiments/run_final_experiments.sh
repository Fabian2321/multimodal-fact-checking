#!/bin/bash

# Final Experiments Script for Multimodal Fact-Checking Project
# This script runs all necessary experiments to generate comprehensive results

set -e  # Exit on any error

echo "=== STARTING FINAL EXPERIMENTS FOR MULTIMODAL FACT-CHECKING PROJECT ==="
echo "Date: $(date)"
echo ""

# Create results directory structure
mkdir -p results/final_experiments
mkdir -p results/final_experiments/figures
mkdir -p results/final_experiments/tables
mkdir -p results/final_experiments/reports

# Configuration
DATA_DIR="data"
METADATA_FILE="test_balanced_pairs.csv"
NUM_SAMPLES=200  # Use all balanced test samples
BATCH_SIZE=8
NUM_WORKERS=2

echo "=== EXPERIMENT CONFIGURATION ==="
echo "Dataset: $METADATA_FILE"
echo "Samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo ""

# Function to run experiment and log
run_experiment() {
    local model_type=$1
    local experiment_name=$2
    local additional_args=$3
    
    echo "Running: $model_type - $experiment_name"
    echo "Command: python src/pipeline.py --model_type $model_type --experiment_name $experiment_name --data_dir $DATA_DIR --metadata_file $METADATA_FILE --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS $additional_args"
    
    python src/pipeline.py \
        --model_type $model_type \
        --experiment_name $experiment_name \
        --data_dir $DATA_DIR \
        --metadata_file $METADATA_FILE \
        --num_samples $NUM_SAMPLES \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS \
        $additional_args
    
    echo "Completed: $model_type - $experiment_name"
    echo "---"
}

echo "=== 1. CLIP EXPERIMENTS ==="

# CLIP Zero-shot
run_experiment "clip" "clip_zs_baseline" "--clip_model_name openai/clip-vit-base-patch32"

# CLIP with RAG
run_experiment "clip" "clip_zs_rag" "--clip_model_name openai/clip-vit-base-patch32 --use_rag --rag_knowledge_base_path data/knowledge_base"

echo "=== 2. BLIP EXPERIMENTS ==="

# BLIP Zero-shot with different prompts
run_experiment "blip" "blip_zs_forced_choice" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name zs_forced_choice"
run_experiment "blip" "blip_zs_yesno_justification" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name zs_yesno_justification"
run_experiment "blip" "blip_zs_cot" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name zs_cot"

# BLIP Few-shot
run_experiment "blip" "blip_fs_yesno_justification" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name fs_yesno_justification --use_few_shot"

# BLIP with RAG
run_experiment "blip" "blip_zs_rag" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name zs_yesno_justification --use_rag --rag_knowledge_base_path data/knowledge_base"
run_experiment "blip" "blip_fs_rag" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name fs_yesno_justification --use_few_shot --use_rag --rag_knowledge_base_path data/knowledge_base"

echo "=== 3. LLaVA EXPERIMENTS ==="

# LLaVA Zero-shot
run_experiment "llava" "llava_zs_cot" "--llava_model_name llava-hf/llava-1.5-7b-hf --prompt_name zs_cot"
run_experiment "llava" "llava_zs_forced_choice" "--llava_model_name llava-hf/llava-1.5-7b-hf --prompt_name zs_forced_choice"

# LLaVA Few-shot
run_experiment "llava" "llava_fs_cot" "--llava_model_name llava-hf/llava-1.5-7b-hf --prompt_name fs_step_by_step --use_few_shot"

# LLaVA with RAG
run_experiment "llava" "llava_zs_rag" "--llava_model_name llava-hf/llava-1.5-7b-hf --prompt_name zs_cot --use_rag --rag_knowledge_base_path data/knowledge_base"
run_experiment "llava" "llava_fs_rag" "--llava_model_name llava-hf/llava-1.5-7b-hf --prompt_name fs_step_by_step --use_few_shot --use_rag --rag_knowledge_base_path data/knowledge_base"

echo "=== 4. BERT BASELINE ==="

# BERT text-only baseline
run_experiment "bert" "bert_baseline" "--bert_model_name bert-base-uncased"

echo "=== ALL EXPERIMENTS COMPLETED ==="
echo "Results saved in results/final_experiments/"
echo "Date: $(date)" 