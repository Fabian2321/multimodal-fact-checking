#!/bin/bash

# BLIP2 Enhanced Experiments Script
# Applies CLIP success factors to BLIP2: RAG-enhanced prompts, optimized prompts, better strategies

set -e

echo "🚀 BLIP2 Enhanced Experiments - Learning from CLIP Success"
echo "=========================================================="

# Configuration
DATA_DIR="data"
METADATA_FILE="test_balanced_pairs_clean.csv"
TEXT_COL="clean_title"
LABEL_COL="2_way_label"
BATCH_SIZE=16
NUM_SAMPLES=100
OUTPUT_DIR="results"
BLIP_MODEL="Salesforce/blip2-flan-t5-xl"

# RAG Configuration
RAG_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
RAG_TOP_K=3
RAG_SIMILARITY_THRESHOLD=0.7
RAG_KB_PATH="src/data/knowledge_base"
RAG_INITIAL_DOCS="src/data/knowledge_base/documents.json"

# Create output directory
mkdir -p "${OUTPUT_DIR}/blip2_enhanced"

echo "📊 Step 1: BLIP2 with RAG-Enhanced Prompt (CLIP-style)"
python -m src.pipeline \
    --model_type blip \
    --blip_model_name ${BLIP_MODEL} \
    --prompt_name blip_rag_enhanced \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name blip2_rag_enhanced \
    --use_rag \
    --rag_embedding_model ${RAG_EMBEDDING_MODEL} \
    --rag_top_k ${RAG_TOP_K} \
    --rag_similarity_threshold ${RAG_SIMILARITY_THRESHOLD} \
    --rag_knowledge_base_path ${RAG_KB_PATH} \
    --rag_initial_docs ${RAG_INITIAL_DOCS} \
    --num_workers 2

echo "📊 Step 2: BLIP2 with Optimized Zero-Shot Prompt"
python -m src.pipeline \
    --model_type blip \
    --blip_model_name ${BLIP_MODEL} \
    --prompt_name blip_optimized_zeroshot \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name blip2_optimized_zeroshot \
    --num_workers 2

echo "📊 Step 3: BLIP2 with Enhanced Few-Shot Prompt"
python -m src.pipeline \
    --model_type blip \
    --blip_model_name ${BLIP_MODEL} \
    --prompt_name blip_enhanced_fewshot \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name blip2_enhanced_fewshot \
    --use_few_shot \
    --num_workers 2

echo "📊 Step 4: BLIP2 with RAG + Few-Shot Combination"
python -m src.pipeline \
    --model_type blip \
    --blip_model_name ${BLIP_MODEL} \
    --prompt_name blip_rag_fewshot_combined \
    --data_dir ${DATA_DIR} \
    --metadata_file ${METADATA_FILE} \
    --text_column ${TEXT_COL} \
    --label_column ${LABEL_COL} \
    --batch_size ${BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --output_dir ${OUTPUT_DIR} \
    --experiment_name blip2_rag_fewshot_combined \
    --use_rag \
    --use_few_shot \
    --rag_embedding_model ${RAG_EMBEDDING_MODEL} \
    --rag_top_k ${RAG_TOP_K} \
    --rag_similarity_threshold ${RAG_SIMILARITY_THRESHOLD} \
    --rag_knowledge_base_path ${RAG_KB_PATH} \
    --rag_initial_docs ${RAG_INITIAL_DOCS} \
    --num_workers 2

echo "🔍 Step 5: Analyzing BLIP2 results and comparing with CLIP"
python -c "
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
output_dir = '${OUTPUT_DIR}/blip2_enhanced'
os.makedirs(output_dir, exist_ok=True)

# Define experiments to analyze
experiments = [
    'blip2_rag_enhanced',
    'blip2_optimized_zeroshot',
    'blip2_enhanced_fewshot',
    'blip2_rag_fewshot_combined'
]

results_summary = []

for exp in experiments:
    try:
        # Load results
        results_path = f'${OUTPUT_DIR}/blip/{exp}/all_model_outputs.csv'
        df = pd.read_csv(results_path)
        
        print(f'📈 Loaded {len(df)} samples for {exp}')
        
        # Calculate metrics (BLIP2 uses predicted_labels directly)
        true_labels = df['true_label'].values
        predictions = df['predicted_labels'].values
        
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
            'f1': f1
        })
        
        print(f'   Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
        
    except Exception as e:
        print(f'❌ Error processing {exp}: {e}')

# Create comparison DataFrame
if results_summary:
    results_df = pd.DataFrame(results_summary)
    
    print(f'\\n🏆 BLIP2 Enhanced Results:')
    print('=' * 60)
    for _, row in results_df.iterrows():
        print(f'{row[\"experiment\"]:30} | Acc: {row[\"accuracy\"]:.3f} | F1: {row[\"f1\"]:.3f} | Prec: {row[\"precision\"]:.3f} | Rec: {row[\"recall\"]:.3f}')
    
    # Find best performing experiment
    best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    
    print(f'\\n🎯 Best BLIP2 Results:')
    print(f'   Best Accuracy: {best_accuracy[\"accuracy\"]:.3f} ({best_accuracy[\"experiment\"]})')
    print(f'   Best F1-Score: {best_f1[\"f1\"]:.3f} ({best_f1[\"experiment\"]})')
    
    # Compare with CLIP results
    print(f'\\n📊 Comparison with CLIP Results:')
    print(f'   CLIP + RAG Enhanced: 74.0% Accuracy')
    print(f'   BLIP2 Best: {best_accuracy[\"accuracy\"]:.1%} Accuracy')
    print(f'   Gap: {0.74 - best_accuracy[\"accuracy\"]:.1%}')
    
    # Save results
    results_df.to_csv(f'{output_dir}/blip2_enhanced_results.csv', index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('BLIP2 Enhanced Experiments - Learning from CLIP', fontsize=16)
    
    # Accuracy comparison
    axes[0,0].bar(results_df['experiment'], results_df['accuracy'], color='#2E86AB')
    axes[0,0].axhline(y=0.74, color='red', linestyle='--', label='CLIP Best (74%)')
    axes[0,0].set_title('Accuracy Comparison')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend()
    
    # F1 comparison
    axes[0,1].bar(results_df['experiment'], results_df['f1'], color='#A23B72')
    axes[0,1].set_title('F1-Score Comparison')
    axes[0,1].set_ylabel('F1-Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Precision vs Recall
    axes[1,0].scatter(results_df['precision'], results_df['recall'], s=100, alpha=0.7)
    for i, exp in enumerate(results_df['experiment']):
        axes[1,0].annotate(exp.split('_')[-1], (results_df['precision'].iloc[i], results_df['recall'].iloc[i]), 
                          xytext=(5, 5), textcoords='offset points')
    axes[1,0].set_xlabel('Precision')
    axes[1,0].set_ylabel('Recall')
    axes[1,0].set_title('Precision vs Recall')
    axes[1,0].grid(True, alpha=0.3)
    
    # Method comparison
    methods = ['Zero-Shot', 'RAG', 'Few-Shot', 'RAG+Few-Shot']
    accuracies = results_df['accuracy'].values
    axes[1,1].bar(methods, accuracies, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
    axes[1,1].axhline(y=0.74, color='red', linestyle='--', label='CLIP Best (74%)')
    axes[1,1].set_title('Method Comparison')
    axes[1,1].set_ylabel('Accuracy')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/blip2_enhanced_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'\\n📁 Results saved to: {output_dir}/')
    print(f'   - blip2_enhanced_results.csv')
    print(f'   - blip2_enhanced_comparison.png')
    
    # Analysis and recommendations
    print(f'\\n🔍 Key Insights:')
    print(f'   1. Best BLIP2 method: {best_accuracy[\"experiment\"]}')
    print(f'   2. Performance gap to CLIP: {0.74 - best_accuracy[\"accuracy\"]:.1%}')
    print(f'   3. RAG impact: Compare RAG vs non-RAG methods')
    print(f'   4. Few-shot impact: Compare few-shot vs zero-shot')
    
    # Recommendations
    print(f'\\n💡 Recommendations for further BLIP2 improvement:')
    if best_accuracy['accuracy'] < 0.65:
        print(f'   - Focus on prompt engineering (current best: {best_accuracy[\"accuracy\"]:.1%})')
        print(f'   - Try different BLIP2 model variants')
        print(f'   - Optimize RAG parameters')
    elif best_accuracy['accuracy'] < 0.70:
        print(f'   - Fine-tune prompts based on best performing method')
        print(f'   - Consider ensemble with CLIP')
        print(f'   - Test on larger dataset')
    else:
        print(f'   - Excellent performance! Consider ensemble approaches')
        print(f'   - Test on larger dataset for validation')

else:
    print('❌ No results to analyze')

# Load previous BLIP2 results for comparison
print(f'\\n📊 Historical BLIP2 Performance:')
print(f'   BLIP2 Baseline: ~59% (few-shot)')
print(f'   BLIP2 Zero-shot: ~50%')
print(f'   Target: Close the gap to CLIP (74%)')
"

echo "✅ BLIP2 Enhanced Experiments completed!"
echo ""
echo "📈 Key Learnings from CLIP Success:"
echo "1. RAG-enhanced prompts work well"
echo "2. Optimized thresholds improve performance"
echo "3. Knowledge base integration helps"
echo "4. Prompt engineering is crucial"
echo ""
echo "🎯 Next steps:"
echo "1. Analyze which BLIP2 method works best"
echo "2. Fine-tune the best performing approach"
echo "3. Consider BLIP2 + CLIP ensemble"
echo "4. Test on larger dataset" 