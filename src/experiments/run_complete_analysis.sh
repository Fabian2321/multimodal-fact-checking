#!/bin/bash

# Complete Analysis Script for Multimodal Fact-Checking Project
# This script runs all experiments and generates comprehensive results

set -e  # Exit on any error

echo "=========================================="
echo "MULTIMODAL FACT-CHECKING PROJECT"
echo "COMPLETE ANALYSIS SCRIPT"
echo "=========================================="
echo "Date: $(date)"
echo ""

# Configuration
DATA_DIR="data"
METADATA_FILE="test_balanced_pairs.csv"
NUM_SAMPLES=200  # Use all balanced test samples
BATCH_SIZE=8
NUM_WORKERS=2

echo "=== CONFIGURATION ==="
echo "Dataset: $METADATA_FILE"
echo "Samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo ""

# Create results directory structure
mkdir -p results/final_experiments
mkdir -p results/final_experiments/figures
mkdir -p results/final_experiments/tables
mkdir -p results/final_experiments/reports
mkdir -p results/final_experiments/examples

# Function to run experiment and log
run_experiment() {
    local model_type=$1
    local experiment_name=$2
    local additional_args=$3
    
    echo "Running: $model_type - $experiment_name"
    echo "Command: python src/pipeline.py --model_type $model_type --experiment_name $experiment_name --data_dir $DATA_DIR --metadata_file $METADATA_FILE --num_samples $NUM_SAMPLES --batch_size $BATCH_SIZE --num_workers $NUM_WORKERS $additional_args"
    
    PYTHONPATH=. python src/pipeline.py \
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

echo "=========================================="
echo "PHASE 1: RUNNING EXPERIMENTS"
echo "=========================================="

echo "=== 1. CLIP EXPERIMENTS ==="

# CLIP Zero-shot
run_experiment "clip" "clip_zs_baseline" "--clip_model_name openai/clip-vit-base-patch32"

# CLIP with RAG
run_experiment "clip" "clip_zs_rag" "--clip_model_name openai/clip-vit-base-patch32 --use_rag --rag_knowledge_base_path data/knowledge_base"

echo "=== 2. BLIP EXPERIMENTS ==="

# BLIP Zero-shot with different prompts
run_experiment "blip" "blip_zs_direct_answer" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name zs_direct_answer --max_new_tokens_blip 150"
run_experiment "blip" "blip_zs_forced_choice" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name zs_forced_choice --max_new_tokens_blip 150"
run_experiment "blip" "blip_zs_yesno_justification" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name zs_yesno_justification --max_new_tokens_blip 150"

# BLIP Few-shot
run_experiment "blip" "blip_fs_yesno_justification" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name fs_yesno_justification --use_few_shot --max_new_tokens_blip 150"

# BLIP with RAG
run_experiment "blip" "blip_zs_rag" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name zs_yesno_justification --use_rag --rag_knowledge_base_path data/knowledge_base --max_new_tokens_blip 150"
run_experiment "blip" "blip_fs_rag" "--blip_model_name Salesforce/blip2-opt-2.7b --prompt_name fs_yesno_justification --use_few_shot --use_rag --rag_knowledge_base_path data/knowledge_base --max_new_tokens_blip 150"

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

echo "=========================================="
echo "PHASE 2: GENERATING RESULTS"
echo "=========================================="

echo "Generating comprehensive results analysis..."
PYTHONPATH=. python src/generate_final_results.py

echo "Generating examples and qualitative analysis..."
PYTHONPATH=. python src/generate_examples_and_analysis.py

echo "=========================================="
echo "PHASE 3: CREATING SUMMARY"
echo "=========================================="

# Create a summary file
SUMMARY_FILE="results/final_experiments/EXPERIMENT_SUMMARY.md"

cat > "$SUMMARY_FILE" << 'EOF'
# Multimodal Fact-Checking Project - Experiment Summary

## Overview
This document summarizes all experiments conducted for the multimodal fact-checking project.

## Experiments Completed

### CLIP Experiments
- **clip_zs_baseline**: CLIP zero-shot classification
- **clip_zs_rag**: CLIP with RAG enhancement

### BLIP Experiments
- **blip_zs_forced_choice**: BLIP2 zero-shot with forced choice prompt
- **blip_zs_yesno_justification**: BLIP2 zero-shot with yes/no + justification
- **blip_zs_cot**: BLIP2 zero-shot with chain-of-thought reasoning
- **blip_fs_yesno_justification**: BLIP2 few-shot with examples
- **blip_zs_rag**: BLIP2 zero-shot with RAG
- **blip_fs_rag**: BLIP2 few-shot with RAG

### LLaVA Experiments
- **llava_zs_cot**: LLaVA zero-shot with chain-of-thought
- **llava_zs_forced_choice**: LLaVA zero-shot with forced choice
- **llava_fs_cot**: LLaVA few-shot with chain-of-thought
- **llava_zs_rag**: LLaVA zero-shot with RAG
- **llava_fs_rag**: LLaVA few-shot with RAG

### Baseline Experiments
- **bert_baseline**: BERT text-only baseline

## Configuration
- **Dataset**: test_balanced_pairs.csv (200 samples)
- **Batch Size**: 8
- **Workers**: 2

## Results Location
All results are saved in `results/final_experiments/`:
- `figures/`: Visualizations and plots
- `tables/`: CSV files with metrics
- `reports/`: Detailed analysis reports
- `examples/`: Sample outputs and qualitative analysis

## Key Files Generated
1. `comprehensive_metrics.csv` - All experiment metrics
2. `final_report.md` - Comprehensive analysis report
3. `model_comparison.png` - Model performance comparison
4. `detailed_analysis.png` - Detailed performance analysis
5. `sample_outputs.csv` - Example model outputs
6. `prompt_examples.md` - Example prompts used
7. `qualitative_insights.md` - Qualitative analysis

## Next Steps
1. Review the final report for key findings
2. Examine visualizations for insights
3. Analyze sample outputs for qualitative understanding
4. Use results for presentation and paper

EOF

echo "=========================================="
echo "ANALYSIS COMPLETE!"
echo "=========================================="
echo "Date: $(date)"
echo ""
echo "Results saved in: results/final_experiments/"
echo "Summary file: $SUMMARY_FILE"
echo ""
echo "Key files generated:"
echo "  ✓ Comprehensive metrics table"
echo "  ✓ Model comparison visualizations"
echo "  ✓ Detailed analysis plots"
echo "  ✓ Final report with insights"
echo "  ✓ Example prompts and outputs"
echo "  ✓ Qualitative analysis"
echo ""
echo "Ready for presentation and paper!" 