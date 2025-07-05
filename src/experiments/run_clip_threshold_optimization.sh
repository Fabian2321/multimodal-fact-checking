#!/bin/bash

# CLIP Threshold Optimization Script
# Tests different CLIP similarity thresholds to find optimal performance

set -e

echo "🎯 Starting CLIP Threshold Optimization"
echo "======================================"

# Configuration
DATA_DIR="data"
METADATA_FILE="test_balanced_pairs_clean.csv"
TEXT_COL="clean_title"
LABEL_COL="2_way_label"
BATCH_SIZE=32
NUM_SAMPLES=100
OUTPUT_DIR="results"
CLIP_MODEL="openai/clip-vit-base-patch32"

# Create output directory
mkdir -p "${OUTPUT_DIR}/threshold_optimization"

echo "📊 Step 1: Running CLIP with raw scores (no threshold)"
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
    --experiment_name clip_threshold_optimization \
    --num_workers 2

echo "🔍 Step 2: Analyzing score distribution and testing thresholds"
python -c "
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# Load CLIP results
results_path = '${OUTPUT_DIR}/clip/clip_threshold_optimization/all_model_outputs.csv'
df = pd.read_csv(results_path)

print(f'📈 Loaded {len(df)} samples')
print(f'📊 Score statistics:')
print(f'   Min: {df[\"scores\"].min():.2f}')
print(f'   Max: {df[\"scores\"].max():.2f}')
print(f'   Mean: {df[\"scores\"].mean():.2f}')
print(f'   Std: {df[\"scores\"].std():.2f}')
print(f'   Median: {df[\"scores\"].median():.2f}')

# Analyze score distribution by true label
print(f'\\n📊 Score distribution by true label:')
for label in [0, 1]:
    label_scores = df[df['true_label'] == label]['scores']
    print(f'   Label {label}: mean={label_scores.mean():.2f}, std={label_scores.std():.2f}')

# Test different thresholds
true_labels = df['true_label'].values
scores = df['scores'].values

# Define threshold range based on score distribution
score_min, score_max = scores.min(), scores.max()
score_mean, score_std = scores.mean(), scores.std()

# Create threshold candidates
thresholds = []
# Around current threshold (27.5)
thresholds.extend([25.0, 26.0, 27.0, 27.5, 28.0, 29.0, 30.0])
# Around mean
thresholds.extend([score_mean - 2*score_std, score_mean - score_std, score_mean, score_mean + score_std, score_mean + 2*score_std])
# Percentiles
percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
for p in percentiles:
    thresholds.append(np.percentile(scores, p))
# Fine-grained around best performing areas
thresholds.extend([24.0, 24.5, 25.5, 26.5, 28.5, 29.5, 31.0, 32.0])

# Remove duplicates and sort
thresholds = sorted(list(set(thresholds)))

print(f'\\n🎯 Testing {len(thresholds)} thresholds: {thresholds[:10]}...')

# Test each threshold
results = []
for threshold in thresholds:
    # Make predictions
    predictions = (scores >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Find best threshold for each metric
best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
best_f1 = results_df.loc[results_df['f1'].idxmax()]
best_precision = results_df.loc[results_df['precision'].idxmax()]
best_recall = results_df.loc[results_df['recall'].idxmax()]

print(f'\\n🏆 Best Results:')
print(f'   Best Accuracy: {best_accuracy[\"accuracy\"]:.3f} at threshold {best_accuracy[\"threshold\"]:.1f}')
print(f'   Best F1: {best_f1[\"f1\"]:.3f} at threshold {best_f1[\"threshold\"]:.1f}')
print(f'   Best Precision: {best_precision[\"precision\"]:.3f} at threshold {best_precision[\"threshold\"]:.1f}')
print(f'   Best Recall: {best_recall[\"recall\"]:.3f} at threshold {best_recall[\"threshold\"]:.1f}')

# Save results
output_dir = '${OUTPUT_DIR}/threshold_optimization'
os.makedirs(output_dir, exist_ok=True)
results_df.to_csv(f'{output_dir}/threshold_optimization_results.csv', index=False)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('CLIP Threshold Optimization Results', fontsize=16)

# Plot metrics vs threshold
metrics = ['accuracy', 'precision', 'recall', 'f1']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    ax.plot(results_df['threshold'], results_df[metric], color=colors[i], linewidth=2)
    ax.set_title(f'{metric.title()} vs Threshold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel(metric.title())
    ax.grid(True, alpha=0.3)
    
    # Mark best point
    best_idx = results_df[metric].idxmax()
    best_threshold = results_df.loc[best_idx, 'threshold']
    best_value = results_df.loc[best_idx, metric]
    ax.scatter(best_threshold, best_value, color=colors[i], s=100, zorder=5)
    ax.annotate(f'{best_value:.3f}', (best_threshold, best_value), 
                xytext=(10, 10), textcoords='offset points', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/threshold_optimization_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# Create score distribution plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histogram of scores
ax1.hist(df[df['true_label'] == 0]['scores'], alpha=0.7, label='Real (0)', bins=20, color='blue')
ax1.hist(df[df['true_label'] == 1]['scores'], alpha=0.7, label='Fake (1)', bins=20, color='red')
ax1.axvline(x=27.5, color='black', linestyle='--', label='Current Threshold (27.5)')
ax1.axvline(x=best_accuracy['threshold'], color='green', linestyle='-', label=f'Best Accuracy ({best_accuracy[\"threshold\"]:.1f})')
ax1.set_xlabel('CLIP Similarity Score')
ax1.set_ylabel('Frequency')
ax1.set_title('Score Distribution by True Label')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Box plot
df_plot = df.copy()
df_plot['label_name'] = df_plot['true_label'].map({0: 'Real', 1: 'Fake'})
sns.boxplot(data=df_plot, x='label_name', y='scores', ax=ax2)
ax2.axhline(y=27.5, color='black', linestyle='--', label='Current Threshold (27.5)')
ax2.axhline(y=best_accuracy['threshold'], color='green', linestyle='-', label=f'Best Accuracy ({best_accuracy[\"threshold\"]:.1f})')
ax2.set_title('Score Distribution by Label')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{output_dir}/score_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print(f'\\n📁 Results saved to: {output_dir}/')
print(f'   - threshold_optimization_results.csv')
print(f'   - threshold_optimization_plot.png')
print(f'   - score_distribution_analysis.png')

# Test best threshold on full dataset
print(f'\\n🚀 Testing best threshold ({best_accuracy[\"threshold\"]:.1f}) on full dataset...')
"

echo "✅ CLIP Threshold Optimization completed!"
echo ""
echo "📈 Next steps:"
echo "1. Analyze the threshold optimization results"
echo "2. Test the best threshold on larger dataset"
echo "3. Consider adaptive thresholds based on score distribution"
echo "4. Combine with RAG for further improvement" 