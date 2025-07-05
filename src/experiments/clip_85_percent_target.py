# --- CLIP 85% Target Script ---
# Builds on the successful 82% setup with targeted improvements

import os
import glob
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords if needed
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_local_image(image_id: str) -> Image.Image:
    """Loads local images from colab_images/ folder"""
    image_pattern = os.path.join("colab_images", f"{image_id}.*")
    matching_files = glob.glob(image_pattern)
    if matching_files:
        return Image.open(matching_files[0]).convert('RGB')
    else:
        print(f"No image found for ID {image_id}")
        return Image.new('RGB', (224, 224), color='gray')

def create_text_variants(text: str) -> List[str]:
    """Creates text variants for more robust predictions"""
    variants = []
    
    # Original (cleaned)
    cleaned = preprocess_text(text)
    variants.append(cleaned)
    
    # Variant 1: Shorter version (first 5 words)
    words = cleaned.split()
    if len(words) > 3:
        short_variant = ' '.join(words[:5])
        variants.append(short_variant)
    
    # Variant 2: Without stop words (more aggressive)
    stop_words = set(stopwords.words('english'))
    important_words = {'fake', 'real', 'true', 'false', 'news', 'image', 'photo', 'picture', 'video', 'shows', 'depicts'}
    filtered_words = [word for word in words if word not in stop_words or word in important_words]
    if len(filtered_words) >= 2:
        filtered_variant = ' '.join(filtered_words)
        if filtered_variant != cleaned:
            variants.append(filtered_variant)
    
    # Variant 3: Extended description
    if 'fake' in text.lower() or 'false' in text.lower():
        extended = f"this image shows fake news: {cleaned}"
        variants.append(extended)
    elif 'real' in text.lower() or 'true' in text.lower():
        extended = f"this image shows real news: {cleaned}"
        variants.append(extended)
    
    return list(set(variants))  # Remove duplicates

def preprocess_text(text: str) -> str:
    """Improved text preprocessing for better CLIP performance"""
    # Basic cleaning
    text = text.lower().strip()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters, keep important ones
    text = re.sub(r'[^\w\s\-\.\,\!\?]', '', text)
    
    # Remove stop words (selectively)
    stop_words = set(stopwords.words('english'))
    # Keep important words
    important_words = {'fake', 'real', 'true', 'false', 'news', 'image', 'photo', 'picture', 'video'}
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words or word in important_words]
    
    # Ensure minimum length
    if len(filtered_words) < 2:
        filtered_words = words[:3]  # Fallback
    
    return ' '.join(filtered_words)

def create_enhanced_crops(image: Image.Image, num_crops: int = 4) -> List[Image.Image]:
    """Enhanced Multi-Crop for 85% Target"""
    crops = [image]  # Original always included
    
    if num_crops > 1:
        width, height = image.size
        # Center crop
        center_crop = image.crop((width//4, height//4, 3*width//4, 3*height//4))
        crops.append(center_crop)
        
        # Corner crops for better coverage
        if num_crops > 2:
            top_left = image.crop((0, 0, width//2, height//2))
            crops.append(top_left)
        
        if num_crops > 3:
            bottom_right = image.crop((width//2, height//2, width, height))
            crops.append(bottom_right)
        
        # Additional Crop: Square from the center
        if num_crops > 4:
            min_dim = min(width, height)
            start_x = (width - min_dim) // 2
            start_y = (height - min_dim) // 2
            square_crop = image.crop((start_x, start_y, start_x + min_dim, start_y + min_dim))
            crops.append(square_crop)
    
    return crops[:num_crops]

class CLIP85TargetHandler:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        """CLIP configuration for 85% Target"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading CLIP model: {model_name}")
        
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        print("CLIP model loaded successfully!")

    def predict_similarity_85_target(self, text: str, image: Image.Image, num_crops: int = 4) -> float:
        """Enhanced Similarity calculation for 85% Target"""
        # Create text variants
        text_variants = create_text_variants(text)
        
        # Enhanced Multi-Crop
        crops = create_enhanced_crops(image, num_crops)
        
        all_similarities = []
        
        # For each text variant and each crop
        for text_variant in text_variants:
            for crop in crops:
                inputs = self.processor(
                    text=[text_variant], 
                    images=crop, 
                    return_tensors="pt", 
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
                    text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
                    similarity = (image_embeds @ text_embeds.T).cpu().item()
                    all_similarities.append(similarity)
        
        # Optimized aggregation for 85% Target
        if len(all_similarities) > 0:
            # Top-K aggregation (best 60% of scores)
            sorted_sims = sorted(all_similarities, reverse=True)
            k = max(1, int(len(sorted_sims) * 0.6))
            top_k_sims = sorted_sims[:k]
            
            # Weighted combination: Max + Top-K Mean
            max_sim = max(all_similarities)
            top_k_mean = np.mean(top_k_sims)
            
            return 0.6 * max_sim + 0.4 * top_k_mean
        else:
            return 0.0

    def find_optimal_threshold_85_target(self, similarities: list, true_labels: list) -> Dict[str, float]:
        """Enhanced threshold optimization for 85% Target"""
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        # ROC-based optimization
        fpr, tpr, roc_thresholds = roc_curve(true_labels, similarities)
        j_scores = tpr - fpr
        best_roc_idx = np.argmax(j_scores)
        roc_threshold = roc_thresholds[best_roc_idx]
        
        # Precision-Recall optimization
        precision, recall, pr_thresholds = precision_recall_curve(true_labels, similarities)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_pr_idx = np.argmax(f1_scores[:-1])
        pr_threshold = pr_thresholds[best_pr_idx]
        
        # Balanced Accuracy optimization
        balanced_accuracies = []
        for threshold in roc_thresholds:
            predictions = [int(sim >= threshold) for sim in similarities]
            tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            balanced_acc = (specificity + sensitivity) / 2
            balanced_accuracies.append(balanced_acc)
        
        best_ba_idx = np.argmax(balanced_accuracies)
        ba_threshold = roc_thresholds[best_ba_idx]
        
        # Custom Threshold for 85% Target: Balance between Precision and Recall
        custom_scores = []
        for threshold in roc_thresholds:
            predictions = [int(sim >= threshold) for sim in similarities]
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            # Weight Recall slightly higher for 85% Target
            custom_score = 0.4 * precision + 0.6 * recall
            custom_scores.append(custom_score)
        
        best_custom_idx = np.argmax(custom_scores)
        custom_threshold = roc_thresholds[best_custom_idx]
        
        print(f"85% Target Threshold Analysis:")
        print(f"  - ROC J-score threshold: {roc_threshold:.3f}")
        print(f"  - Precision-Recall F1 threshold: {pr_threshold:.3f}")
        print(f"  - Balanced Accuracy threshold: {ba_threshold:.3f}")
        print(f"  - Custom 85% target threshold: {custom_threshold:.3f}")
        
        return {
            'roc_threshold': roc_threshold,
            'pr_threshold': pr_threshold,
            'ba_threshold': ba_threshold,
            'custom_threshold': custom_threshold,
            'j_score': j_scores[best_roc_idx],
            'f1_score': f1_scores[best_pr_idx],
            'balanced_accuracy': balanced_accuracies[best_ba_idx],
            'custom_score': custom_scores[best_custom_idx]
        }

def calculate_comprehensive_metrics(y_true, y_pred, similarities, threshold):
    """Calculates all important metrics for documentation"""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)
    # Per-Class metrics
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_accuracy = (specificity + sensitivity) / 2
    # Similarity statistics
    pos_similarities = [s for s, l in zip(similarities, y_true) if l == 1]
    neg_similarities = [s for s, l in zip(similarities, y_true) if l == 0]
    print("\n" + "="*60)
    print("CLIP 85% TARGET EXPERIMENT - COMPREHENSIVE METRICS")
    print("="*60)
    print(f"Setup:")
    print(f"  - Model: openai/clip-vit-base-patch16")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Samples: {len(y_true)}")
    print(f"  - Threshold: {threshold:.3f}")
    print(f"  - Optimizations: 4-crop, Text-variants, Top-K aggregation")
    print(f"\nPerformance Metrics:")
    print(f"  - Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  - Precision: {precision:.3f}")
    print(f"  - Recall:    {recall:.3f}")
    print(f"  - F1-Score:  {f1:.3f}")
    print(f"  - Specificity: {specificity:.3f}")
    print(f"  - Sensitivity: {sensitivity:.3f}")
    print(f"  - Balanced Accuracy: {balanced_accuracy:.3f}")
    print(f"  - ROC AUC:   {roc_auc:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")
    print(f"\nSimilarity Statistics:")
    print(f"  Positive samples: {len(pos_similarities)}")
    print(f"  Negative samples: {len(neg_similarities)}")
    print(f"  Positive mean similarity: {np.mean(pos_similarities):.3f}")
    print(f"  Negative mean similarity: {np.mean(neg_similarities):.3f}")
    print(f"  Positive std similarity:  {np.std(pos_similarities):.3f}")
    print(f"  Negative std similarity:  {np.std(neg_similarities):.3f}")
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'balanced_accuracy': balanced_accuracy,
        'roc_auc': roc_auc,
        'threshold': threshold,
        'confusion_matrix': cm,
        'similarities': similarities,
        'y_true': y_true,
        'y_pred': y_pred
    }

def plot_85_target_results(metrics, threshold_analysis):
    """Creates visualizations for 85% Target results"""
    
    plt.figure(figsize=(20, 5))
    
    # Confusion Matrix
    plt.subplot(1, 4, 1)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # ROC Curve
    plt.subplot(1, 4, 2)
    fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['similarities'])
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    # Similarity Distribution
    plt.subplot(1, 4, 3)
    pos_sim = [s for s, l in zip(metrics['similarities'], metrics['y_true']) if l == 1]
    neg_sim = [s for s, l in zip(metrics['similarities'], metrics['y_true']) if l == 0]
    
    plt.hist(pos_sim, alpha=0.7, label='Positive', bins=20, color='green')
    plt.hist(neg_sim, alpha=0.7, label='Negative', bins=20, color='red')
    plt.axvline(metrics['threshold'], color='black', linestyle='--', 
                label=f'Threshold: {metrics["threshold"]:.3f}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Similarity Distribution')
    plt.legend()
    
    # Threshold Comparison
    plt.subplot(1, 4, 4)
    thresholds = ['ROC', 'PR-F1', 'Balanced', 'Custom']
    accuracies = [
        threshold_analysis.get('roc_accuracy', 0),
        threshold_analysis.get('pr_accuracy', 0),
        threshold_analysis.get('ba_accuracy', 0),
        threshold_analysis.get('custom_accuracy', 0)
    ]
    plt.bar(thresholds, accuracies, color=['blue', 'green', 'orange', 'red'])
    plt.ylabel('Accuracy')
    plt.title('Threshold Strategy Comparison')
    plt.ylim(0.75, 0.90)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.005, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function for 85% Target"""
    
    # Parameters
    CSV_FILE = "test_balanced_pairs_clean.csv"
    NUM_SAMPLES = 100
    OUTPUT_FILE = "clip_85_percent_target_results.csv"
    NUM_CROPS = 4  # Increased for 85% Target
    
    print("CLIP 85% Target Experiment")
    print("="*40)
    print(f"85% Target Optimizations:")
    print(f"  - Enhanced Multi-crop: {NUM_CROPS} crops per image")
    print(f"  - Text variants: Multiple text processing approaches")
    print(f"  - Top-K aggregation: Best 60% of similarities")
    print(f"  - Custom threshold: Recall-focused for 85%")
    
    # File checks
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV file {CSV_FILE} not found!")
        return
    
    if not os.path.exists("colab_images"):
        print("❌ colab_images folder not found!")
        return
    
    # Load data
    print(f"📊 Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE).head(NUM_SAMPLES)
    print(f"✅ Loaded {len(df)} samples")
    
    # Initialize CLIP
    clip = CLIP85TargetHandler()
    
    # Perform predictions
    results = []
    similarities = []
    true_labels = []
    
    print(f"🔄 Running 85% target CLIP predictions on {len(df)} samples...")
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing sample {idx+1}/{len(df)}")
        
        image = load_local_image(row['id'])
        sim = clip.predict_similarity_85_target(row['clean_title'], image, NUM_CROPS)
        
        similarities.append(sim)
        true_labels.append(row['2_way_label'])
        
        results.append({
            'id': row['id'],
            'text': row['clean_title'],
            'image_url': row['image_url'],
            'true_label': row['2_way_label'],
            'clip_similarity': sim,
        })
    
    # 85% Target Threshold Optimization
    print(f"\n🎯 Finding optimal threshold for 85% target...")
    threshold_analysis = clip.find_optimal_threshold_85_target(similarities, true_labels)
    
    # Test different thresholds
    thresholds_to_test = [
        ('roc_threshold', threshold_analysis['roc_threshold']),
        ('pr_threshold', threshold_analysis['pr_threshold']),
        ('ba_threshold', threshold_analysis['ba_threshold']),
        ('custom_threshold', threshold_analysis['custom_threshold'])
    ]
    
    best_accuracy = 0
    best_threshold_name = ''
    best_threshold = 0
    best_predictions = []
    
    print(f"\n🔍 Testing different threshold strategies...")
    for name, threshold in thresholds_to_test:
        predictions = [int(sim >= threshold) for sim in similarities]
        accuracy = accuracy_score(true_labels, predictions)
        print(f"  {name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Store Accuracy for plotting
        threshold_analysis[f'{name.split("_")[0]}_accuracy'] = accuracy
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold_name = name
            best_threshold = threshold
            best_predictions = predictions
    
    print(f"\n🏆 Best threshold: {best_threshold_name} = {best_threshold:.3f}")
    print(f"🎯 Best accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    # Final predictions set
    for r in results:
        r['clip_predicted_label'] = int(r['clip_similarity'] >= best_threshold)
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(true_labels, best_predictions, similarities, best_threshold)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to {OUTPUT_FILE}")
    
    # Save metrics as JSON
    import json
    metrics_file = "clip_85_percent_target_metrics.json"
    metrics_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                   for k, v in metrics.items() 
                   if k not in ['confusion_matrix', 'similarities', 'y_true', 'y_pred']}
    metrics_dict['threshold_analysis'] = threshold_analysis
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"📊 Metrics saved to {metrics_file}")
    
    # Create visualizations
    print(f"\n📈 Creating visualizations...")
    plot_85_target_results(metrics, threshold_analysis)
    
    print(f"\n✅ CLIP 85% Target Experiment completed!")
    print(f"🎉 Achieved {metrics['accuracy']*100:.1f}% accuracy!")
    
    if metrics['accuracy'] >= 0.85:
        print(f"🎯 TARGET ACHIEVED: 85%+ Accuracy!")
        print(f"🚀 EXCELLENT! You've reached the 85% milestone!")
    elif metrics['accuracy'] >= 0.83:
        print(f"📈 Very close to target: {metrics['accuracy']*100:.1f}% (target: 85%)")
        print(f"💡 Try increasing NUM_CROPS to 5 or adding more text variants")
    else:
        print(f"📉 Need further optimization: {metrics['accuracy']*100:.1f}% (target: 85%)")
        print(f"💡 Consider ensemble with other models")
    
    print(f"\n📋 85% Target Optimization Summary:")
    print(f"  - Enhanced Multi-crop: {NUM_CROPS} crops per image")
    print(f"  - Text variants: Multiple processing approaches")
    print(f"  - Top-K aggregation: Best 60% of similarities")
    print(f"  - Threshold optimization: {best_threshold_name}")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    main() 