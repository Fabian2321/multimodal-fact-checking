#!/bin/bash

# CLIP + RAG + Optimized Threshold Script
# Tests CLIP with RAG and optimized threshold (26.5) for maximum performance

set -e

echo "🚀 Testing CLIP + RAG + Optimized Threshold (26.5)"
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
OPTIMIZED_THRESHOLD=26.5

# RAG Configuration
RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K=3
RAG_SIMILARITY_THRESHOLD=0.7
RAG_KB_PATH="src/data/knowledge_base"
RAG_INITIAL_DOCS="src/data/knowledge_base/documents.json"

# Create output directory
mkdir -p "${OUTPUT_DIR}/clip_rag_optimized"

echo "📊 Step 1: CLIP + RAG Enhanced Prompt with Optimized Threshold"
python -m src.pipeline \
    --model_type clip \
    --clip_model_name ${CLIP_MODEL} \
    --prompt_name clip_rag_enhanced \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name clip_rag_enhanced_optimized \
    --use_rag \
    --rag_embedding_model ${RAG_EMBEDDING_MODEL} \
    --rag_top_k ${RAG_TOP_K} \
    --rag_similarity_threshold ${RAG_SIMILARITY_THRESHOLD} \
    --rag_knowledge_base_path ${RAG_KB_PATH} \
    --rag_initial_docs ${RAG_INITIAL_DOCS} \
    --num_workers 2

echo "📊 Step 2: CLIP + RAG Fact-Check Prompt with Optimized Threshold"
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
    --experiment_name clip_rag_fact_check_optimized \
    --use_rag \
    --rag_embedding_model ${RAG_EMBEDDING_MODEL} \
    --rag_top_k ${RAG_TOP_K} \
    --rag_similarity_threshold ${RAG_SIMILARITY_THRESHOLD} \
    --rag_knowledge_base_path ${RAG_KB_PATH} \
    --rag_initial_docs ${RAG_INITIAL_DOCS} \
    --num_workers 2

echo "📊 Step 3: CLIP + RAG Metadata Prompt with Optimized Threshold"
python -m src.pipeline \
    --model_type clip \
    --clip_model_name ${CLIP_MODEL} \
    --prompt_name clip_rag_metadata \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name clip_rag_metadata_optimized \
    --use_rag \
    --rag_embedding_model ${RAG_EMBEDDING_MODEL} \
    --rag_top_k ${RAG_TOP_K} \
    --rag_similarity_threshold ${RAG_SIMILARITY_THRESHOLD} \
    --rag_knowledge_base_path ${RAG_KB_PATH} \
    --rag_initial_docs ${RAG_INITIAL_DOCS} \
    --num_workers 2

echo "🔍 Step 4: Analyzing results with optimized threshold (${OPTIMIZED_THRESHOLD})"
python -c "
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
optimized_threshold = ${OPTIMIZED_THRESHOLD}
output_dir = '${OUTPUT_DIR}/clip_rag_optimized'
os.makedirs(output_dir, exist_ok=True)

# Define experiments to analyze
experiments = [
    'clip_rag_enhanced_optimized',
    'clip_rag_fact_check_optimized', 
    'clip_rag_metadata_optimized'
]

results_summary = []

for exp in experiments:
    try:
        # Load results
        results_path = f'${OUTPUT_DIR}/clip/{exp}/all_model_outputs.csv'
        df = pd.read_csv(results_path)
        
        print(f'📈 Loaded {len(df)} samples for {exp}')
        
        # Apply optimized threshold
        true_labels = df['true_label'].values
        scores = df['scores'].values
        predictions = (scores >= optimized_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        results_summary.append({
            'experiment': exp,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_score': scores.mean(),
            'std_score': scores.std()
        })
        
        print(f'   Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
        
    except Exception as e:
        print(f'❌ Error processing {exp}: {e}')

# Create comparison DataFrame
if results_summary:
    results_df = pd.DataFrame(results_summary)
    
    print(f'\\n🏆 CLIP + RAG + Optimized Threshold Results:')
    print('=' * 60)
    for _, row in results_df.iterrows():
        print(f'{row[\"experiment\"]:30} | Acc: {row[\"accuracy\"]:.3f} | F1: {row[\"f1\"]:.3f} | Prec: {row[\"precision\"]:.3f} | Rec: {row[\"recall\"]:.3f}')
    
    # Find best performing experiment
    best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    
    print(f'\\n🎯 Best Results:')
    print(f'   Best Accuracy: {best_accuracy[\"accuracy\"]:.3f} ({best_accuracy[\"experiment\"]})')
    print(f'   Best F1-Score: {best_f1[\"f1\"]:.3f} ({best_f1[\"experiment\"]})')
    
    # Save results
    results_df.to_csv(f'{output_dir}/clip_rag_optimized_results.csv', index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CLIP + RAG + Optimized Threshold Comparison', fontsize=16)
    
    # Accuracy comparison
    axes[0,0].bar(results_df['experiment'], results_df['accuracy'], color='#2E86AB')
    axes[0,0].set_title('Accuracy Comparison')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # F1 comparison
    axes[0,1].bar(results_df['experiment'], results_df['f1'], color='#A23B72')
    axes[0,1].set_title('F1-Score Comparison')
    axes[0,1].set_ylabel('F1-Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Precision vs Recall
    axes[1,0].scatter(results_df['precision'], results_df['recall'], s=100, alpha=0.7)
    for i, exp in enumerate(results_df['experiment']):
        axes[1,0].annotate(exp.split('_')[-2], (results_df['precision'].iloc[i], results_df['recall'].iloc[i]), 
                          xytext=(5, 5), textcoords='offset points')
    axes[1,0].set_xlabel('Precision')
    axes[1,0].set_ylabel('Recall')
    axes[1,0].set_title('Precision vs Recall')
    axes[1,0].grid(True, alpha=0.3)
    
    # Score distribution
    axes[1,1].boxplot([results_df['mean_score']], labels=['Mean Score'])
    axes[1,1].set_title('Score Distribution')
    axes[1,1].set_ylabel('CLIP Score')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/clip_rag_optimized_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\\n📁 Results saved to: {output_dir}/')
    print(f'   - clip_rag_optimized_results.csv')
    print(f'   - clip_rag_optimized_comparison.png')
    
    # Check if we reached 71% accuracy
    max_accuracy = results_df['accuracy'].max()
    if max_accuracy >= 0.71:
        print(f'\\n🎉 SUCCESS! Reached {max_accuracy:.3f} accuracy (≥71%)!')
    else:
        print(f'\\n📊 Best accuracy: {max_accuracy:.3f} (target: ≥71%)')
        print(f'   Need +{0.71 - max_accuracy:.3f} to reach target')

else:
    print('❌ No results to analyze')

# Compare with previous best results
print(f'\\n📊 Comparison with Previous Results:')
print(f'   CLIP Baseline (27.5): 68.0%')
print(f'   CLIP Optimized (26.5): 69.0%')
print(f'   CLIP + RAG (27.5): 69.0%')
print(f'   Target: 71.0%')
"

echo "✅ CLIP + RAG + Optimized Threshold Test completed!"
echo ""
echo "📈 Key Insights:"
echo "1. Combined CLIP + RAG + Optimized Threshold"
echo "2. Tested multiple RAG prompt strategies"
echo "3. Applied optimized threshold (26.5)"
echo "4. Analyzed performance improvements"
echo ""
echo "🎯 Next steps:"
echo "1. Analyze which RAG prompt works best"
echo "2. Consider ensemble with BLIP2"
echo "3. Fine-tune RAG parameters"
echo "4. Test on larger dataset" 