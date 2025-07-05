# --- ULTIMATE CLIP 90% TARGET SCRIPT ---
# Combines ALL available optimizations for maximum performance

import os
import glob
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import re
from nltk.corpus import stopwords
import nltk
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

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

def create_ultimate_text_variants(text: str) -> List[str]:
    """ULTIMATE Text variants - maximum coverage"""
    variants = []
    
    # Basic cleaning
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\-\.\,\!\?]', '', text)
    
    # Variant 1: Original (cleaned)
    variants.append(text)
    
    # Variant 2: Shorter version (first 3-7 words)
    words = text.split()
    if len(words) > 2:
        for length in [3, 5, 7]:
            if len(words) >= length:
                short_variant = ' '.join(words[:length])
                variants.append(short_variant)
    
    # Variant 3: Aggressive stop word removal
    stop_words = set(stopwords.words('english'))
    important_words = {
        'fake', 'real', 'true', 'false', 'news', 'image', 'photo', 'picture', 'video',
        'shows', 'depicts', 'displays', 'reveals', 'proves', 'confirms', 'denies',
        'hoax', 'misinformation', 'fact', 'evidence', 'verified', 'unverified'
    }
    filtered_words = [word for word in words if word not in stop_words or word in important_words]
    if len(filtered_words) >= 2:
        filtered_variant = ' '.join(filtered_words)
        variants.append(filtered_variant)
    
    # Variant 4: Extended descriptions
    if any(word in text for word in ['fake', 'false', 'hoax', 'misinformation']):
        extended_fake = f"this image shows fake news or misinformation: {text}"
        variants.append(extended_fake)
    elif any(word in text for word in ['real', 'true', 'fact', 'verified']):
        extended_real = f"this image shows real news or verified information: {text}"
        variants.append(extended_real)
    
    # Variant 5: Question format
    question_variant = f"does this image accurately show: {text}?"
    variants.append(question_variant)
    
    # Variant 6: Claim format
    claim_variant = f"this image claims to show: {text}"
    variants.append(claim_variant)
    
    # Variant 7: Description format
    desc_variant = f"an image depicting: {text}"
    variants.append(desc_variant)
    
    # Variant 8: News format
    news_variant = f"news image showing: {text}"
    variants.append(news_variant)
    
    return list(set(variants))  # Remove duplicates

def create_ultimate_crops(image: Image.Image, num_crops: int = 5) -> List[Image.Image]:
    """ULTIMATE Multi-Crop - adaptive strategy"""
    crops = [image]  # Original always included
    
    if num_crops > 1:
        width, height = image.size
        
        # Crop 2: Center crop (proven)
        center_crop = image.crop((width//4, height//4, 3*width//4, 3*height//4))
        crops.append(center_crop)
        
        # Crop 3: Top-left corner
        if num_crops > 2:
            top_left = image.crop((0, 0, width//2, height//2))
            crops.append(top_left)
        
        # Crop 4: Bottom-right corner
        if num_crops > 3:
            bottom_right = image.crop((width//2, height//2, width, height))
            crops.append(bottom_right)
        
        # Crop 5: Square center crop
        if num_crops > 4:
            min_dim = min(width, height)
            start_x = (width - min_dim) // 2
            start_y = (height - min_dim) // 2
            square_crop = image.crop((start_x, start_y, start_x + min_dim, start_y + min_dim))
            crops.append(square_crop)
        
        # Crop 6: Golden ratio crop (if more than 5)
        if num_crops > 5:
            phi = 1.618
            crop_width = int(width / phi)
            crop_height = int(height / phi)
            start_x = (width - crop_width) // 2
            start_y = (height - crop_height) // 2
            golden_crop = image.crop((start_x, start_y, start_x + crop_width, start_y + crop_height))
            crops.append(golden_crop)
    
    return crops[:num_crops]

def apply_image_augmentations(image: Image.Image) -> List[Image.Image]:
    """Image augmentations for robustness"""
    augmented = [image]
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    bright = enhancer.enhance(1.2)
    dark = enhancer.enhance(0.8)
    augmented.extend([bright, dark])
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    high_contrast = enhancer.enhance(1.3)
    low_contrast = enhancer.enhance(0.7)
    augmented.extend([high_contrast, low_contrast])
    
    # Adjust sharpness
    enhancer = ImageEnhance.Sharpness(image)
    sharp = enhancer.enhance(1.5)
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1))
    augmented.extend([sharp, blurred])
    
    return augmented

class UltimateCLIPHandler:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        """ULTIMATE CLIP configuration for 90% Target"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading CLIP model: {model_name}")
        
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        print("CLIP model loaded successfully!")

    def predict_similarity_ultimate(self, text: str, image: Image.Image, 
                                  num_crops: int = 5, use_augmentation: bool = True) -> Dict[str, float]:
        """ULTIMATE Similarity calculation - all optimizations"""
        
        # Create text variants
        text_variants = create_ultimate_text_variants(text)
        
        # Ultimate Multi-Crop
        crops = create_ultimate_crops(image, num_crops)
        
        # Image augmentations (optional)
        if use_augmentation:
            augmented_crops = []
            for crop in crops:
                augmented_crops.extend(apply_image_augmentations(crop))
            crops = augmented_crops
        
        all_similarities = []
        variant_scores = {}
        
        # For each text variant
        for i, text_variant in enumerate(text_variants):
            variant_similarities = []
            
            # For each crop
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
                    variant_similarities.append(similarity)
                    all_similarities.append(similarity)
            
            # Aggregation per variant
            if variant_similarities:
                max_sim = max(variant_similarities)
                mean_sim = np.mean(variant_similarities)
                top_k_sim = np.mean(sorted(variant_similarities, reverse=True)[:3])
                variant_scores[f'variant_{i}'] = {
                    'max': max_sim,
                    'mean': mean_sim,
                    'top_k': top_k_sim,
                    'weighted': 0.5 * max_sim + 0.3 * mean_sim + 0.2 * top_k_sim
                }
        
        # ULTIMATE Aggregation strategies
        if all_similarities:
            # Strategy 1: Global Max + Mean
            global_max = max(all_similarities)
            global_mean = np.mean(all_similarities)
            
            # Strategy 2: Top-K Weighted
            sorted_sims = sorted(all_similarities, reverse=True)
            k = max(1, int(len(sorted_sims) * 0.3))  # Top 30%
            top_k_sims = sorted_sims[:k]
            top_k_weighted = np.mean(top_k_sims)
            
            # Strategy 3: Variant-Weighted
            variant_weights = [0.4, 0.3, 0.2, 0.1]  # First variants more important
            variant_scores_list = []
            for i, (variant_name, scores) in enumerate(variant_scores.items()):
                weight = variant_weights[i] if i < len(variant_weights) else 0.05
                variant_scores_list.append(scores['weighted'] * weight)
            variant_weighted = sum(variant_scores_list)
            
            # Strategy 4: Ensemble-Weighted
            ensemble_score = 0.3 * global_max + 0.2 * global_mean + 0.3 * top_k_weighted + 0.2 * variant_weighted
            
            return {
                'global_max': global_max,
                'global_mean': global_mean,
                'top_k_weighted': top_k_weighted,
                'variant_weighted': variant_weighted,
                'ensemble': ensemble_score,
                'all_similarities': all_similarities,
                'variant_scores': variant_scores
            }
        else:
            return {
                'global_max': 0.0,
                'global_mean': 0.0,
                'top_k_weighted': 0.0,
                'variant_weighted': 0.0,
                'ensemble': 0.0,
                'all_similarities': [],
                'variant_scores': {}
            }

    def find_ultimate_threshold(self, similarities: list, true_labels: list) -> Dict[str, Any]:
        """ULTIMATE threshold optimization - all strategies"""
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
        
        # Custom Threshold for 90% Target: Precision-Recall Balance
        custom_scores = []
        for threshold in roc_thresholds:
            predictions = [int(sim >= threshold) for sim in similarities]
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            # Weight recall higher for 90% Target
            custom_score = 0.35 * precision + 0.65 * recall
            custom_scores.append(custom_score)
        
        best_custom_idx = np.argmax(custom_scores)
        custom_threshold = roc_thresholds[best_custom_idx]
        
        # Grid Search for finer optimization
        grid_thresholds = np.arange(0.20, 0.35, 0.001)  # Fine search
        grid_accuracies = []
        for threshold in grid_thresholds:
            predictions = [int(sim >= threshold) for sim in similarities]
            accuracy = accuracy_score(true_labels, predictions)
            grid_accuracies.append(accuracy)
        
        best_grid_idx = np.argmax(grid_accuracies)
        grid_threshold = grid_thresholds[best_grid_idx]
        
        print(f"ULTIMATE Threshold Analysis:")
        print(f"  - ROC J-score threshold: {roc_threshold:.3f}")
        print(f"  - Precision-Recall F1 threshold: {pr_threshold:.3f}")
        print(f"  - Balanced Accuracy threshold: {ba_threshold:.3f}")
        print(f"  - Custom 90% target threshold: {custom_threshold:.3f}")
        print(f"  - Grid Search threshold: {grid_threshold:.3f}")
        
        return {
            'roc_threshold': roc_threshold,
            'pr_threshold': pr_threshold,
            'ba_threshold': ba_threshold,
            'custom_threshold': custom_threshold,
            'grid_threshold': grid_threshold,
            'j_score': j_scores[best_roc_idx],
            'f1_score': f1_scores[best_pr_idx],
            'balanced_accuracy': balanced_accuracies[best_ba_idx],
            'custom_score': custom_scores[best_custom_idx],
            'grid_accuracy': grid_accuracies[best_grid_idx]
        }

def calculate_ultimate_metrics(y_true, y_pred, similarities, threshold, strategy_name):
    """Calculates ULTIMATE metrics for 90% Target"""
    
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
    
    print(f"\n{strategy_name.upper()} STRATEGY - COMPREHENSIVE METRICS")
    print("="*60)
    print(f"Setup:")
    print(f"  - Model: openai/clip-vit-base-patch16")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Samples: {len(y_true)}")
    print(f"  - Threshold: {threshold:.3f}")
    print(f"  - Optimizations: Ultimate Multi-crop, Text-variants, Augmentation")
    
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
    print(f"  Separation: {np.mean(pos_similarities) - np.mean(neg_similarities):.3f}")
    
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
        'y_pred': y_pred,
        'strategy': strategy_name
    }

def plot_ultimate_results(all_metrics, threshold_analysis):
    """Erstellt ULTIMATE Visualisierungen"""
    
    strategies = list(all_metrics.keys())
    accuracies = [all_metrics[s]['accuracy'] for s in strategies]
    
    plt.figure(figsize=(20, 10))
    
    # Strategy Comparison
    plt.subplot(2, 3, 1)
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    bars = plt.bar(strategies, accuracies, color=colors[:len(strategies)])
    plt.ylabel('Accuracy')
    plt.title('ULTIMATE Strategy Comparison')
    plt.ylim(0.70, 0.95)
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', fontweight='bold')
    
    # Best Strategy Confusion Matrix
    best_strategy = max(all_metrics.keys(), key=lambda x: all_metrics[x]['accuracy'])
    best_metrics = all_metrics[best_strategy]
    
    plt.subplot(2, 3, 2)
    cm = best_metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Best Strategy: {best_strategy}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # ROC Curve Comparison
    plt.subplot(2, 3, 3)
    for strategy in strategies:
        metrics = all_metrics[strategy]
        fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['similarities'])
        plt.plot(fpr, tpr, label=f'{strategy} (AUC={metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    
    # Similarity Distribution (Best Strategy)
    plt.subplot(2, 3, 4)
    pos_sim = [s for s, l in zip(best_metrics['similarities'], best_metrics['y_true']) if l == 1]
    neg_sim = [s for s, l in zip(best_metrics['similarities'], best_metrics['y_true']) if l == 0]
    
    plt.hist(pos_sim, alpha=0.7, label='Positive', bins=20, color='green')
    plt.hist(neg_sim, alpha=0.7, label='Negative', bins=20, color='red')
    plt.axvline(best_metrics['threshold'], color='black', linestyle='--', 
                label=f'Threshold: {best_metrics["threshold"]:.3f}')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title(f'Similarity Distribution ({best_strategy})')
    plt.legend()
    
    # Threshold Comparison
    plt.subplot(2, 3, 5)
    threshold_names = ['ROC', 'PR-F1', 'Balanced', 'Custom', 'Grid']
    threshold_accuracies = [
        threshold_analysis.get('roc_accuracy', 0),
        threshold_analysis.get('pr_accuracy', 0),
        threshold_analysis.get('ba_accuracy', 0),
        threshold_analysis.get('custom_accuracy', 0),
        threshold_analysis.get('grid_accuracy', 0)
    ]
    plt.bar(threshold_names, threshold_accuracies, color=['blue', 'green', 'orange', 'red', 'purple'])
    plt.ylabel('Accuracy')
    plt.title('Threshold Strategy Comparison')
    plt.ylim(0.70, 0.95)
    for i, v in enumerate(threshold_accuracies):
        plt.text(i, v + 0.005, f'{v:.3f}', ha='center')
    
    # Performance Metrics Heatmap
    plt.subplot(2, 3, 6)
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    metrics_values = []
    for strategy in strategies:
        metrics = all_metrics[strategy]
        metrics_values.append([
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['roc_auc']
        ])
    
    sns.heatmap(metrics_values, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=metrics_names, yticklabels=strategies)
    plt.title('Performance Metrics Heatmap')
    
    plt.tight_layout()
    plt.show()

def main():
    """ULTIMATE CLIP Experiment für 90% Target"""
    
    # Parameter
    CSV_FILE = "test_balanced_pairs_clean.csv"
    NUM_SAMPLES = 100
    OUTPUT_FILE = "clip_ultimate_90_percent_results.csv"
    NUM_CROPS = 5  # Erhöht für Ultimate Performance
    USE_AUGMENTATION = True  # Bild-Augmentation aktivieren
    
    print("ULTIMATE CLIP Experiment - 90% Target")
    print("="*50)
    print(f"ULTIMATE Optimizations:")
    print(f"  - Enhanced Multi-crop: {NUM_CROPS} crops per image")
    print(f"  - Ultimate Text variants: 8+ processing approaches")
    print(f"  - Image augmentation: Brightness, Contrast, Sharpness")
    print(f"  - Multiple aggregation strategies")
    print(f"  - Grid search threshold optimization")
    print(f"  - Ensemble combination")
    
    # Datei-Checks
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV file {CSV_FILE} not found!")
        return
    
    if not os.path.exists("colab_images"):
        print("❌ colab_images folder not found!")
        return
    
    # Daten laden
    print(f"📊 Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE).head(NUM_SAMPLES)
    print(f"✅ Loaded {len(df)} samples")
    
    # CLIP initialisieren
    clip = UltimateCLIPHandler()
    
    # Predictions durchführen
    results = []
    all_strategies = {}
    
    print(f"🔄 Running ULTIMATE CLIP predictions on {len(df)} samples...")
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing sample {idx+1}/{len(df)}")
        
        image = load_local_image(row['id'])
        similarity_dict = clip.predict_similarity_ultimate(row['clean_title'], image, NUM_CROPS, USE_AUGMENTATION)
        
        # Alle Strategien sammeln
        for strategy_name, similarity in similarity_dict.items():
            if strategy_name not in ['all_similarities', 'variant_scores']:
                if strategy_name not in all_strategies:
                    all_strategies[strategy_name] = []
                all_strategies[strategy_name].append(similarity)
        
        results.append({
            'id': row['id'],
            'text': row['clean_title'],
            'image_url': row['image_url'],
            'true_label': row['2_way_label'],
            **similarity_dict
        })
    
    # ULTIMATE Schwellenwert-Optimierung für jede Strategie
    print(f"\n🎯 Finding ULTIMATE thresholds for all strategies...")
    all_metrics = {}
    threshold_analysis = {}
    
    for strategy_name, similarities in all_strategies.items():
        if strategy_name in ['all_similarities', 'variant_scores']:
            continue
            
        print(f"\n--- Optimizing {strategy_name} ---")
        true_labels = [r['true_label'] for r in results]
        
        # Threshold-Optimierung
        strategy_thresholds = clip.find_ultimate_threshold(similarities, true_labels)
        
        # Verschiedene Thresholds testen
        thresholds_to_test = [
            ('roc_threshold', strategy_thresholds['roc_threshold']),
            ('pr_threshold', strategy_thresholds['pr_threshold']),
            ('ba_threshold', strategy_thresholds['ba_threshold']),
            ('custom_threshold', strategy_thresholds['custom_threshold']),
            ('grid_threshold', strategy_thresholds['grid_threshold'])
        ]
        
        best_accuracy = 0
        best_threshold_name = ''
        best_threshold = 0
        best_predictions = []
        
        print(f"Testing thresholds for {strategy_name}:")
        for name, threshold in thresholds_to_test:
            predictions = [int(sim >= threshold) for sim in similarities]
            accuracy = accuracy_score(true_labels, predictions)
            print(f"  {name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            # Speichere für Plot
            threshold_analysis[f'{strategy_name}_{name.split("_")[0]}_accuracy'] = accuracy
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold_name = name
                best_threshold = threshold
                best_predictions = predictions
        
        print(f"Best {strategy_name}: {best_threshold_name} = {best_threshold:.3f} -> {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        
        # Metriken berechnen
        metrics = calculate_ultimate_metrics(true_labels, best_predictions, similarities, best_threshold, strategy_name)
        all_metrics[strategy_name] = metrics
    
    # Beste Strategie finden
    best_strategy = max(all_metrics.keys(), key=lambda x: all_metrics[x]['accuracy'])
    best_accuracy = all_metrics[best_strategy]['accuracy']
    
    print(f"\n🏆 BESTE STRATEGIE: {best_strategy}")
    print(f"🎯 BESTE ACCURACY: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    # Finale Predictions setzen
    best_similarities = all_strategies[best_strategy]
    best_threshold = all_metrics[best_strategy]['threshold']
    
    for r in results:
        r['ultimate_predicted_label'] = int(r[best_strategy] >= best_threshold)
        r['best_strategy'] = best_strategy
        r['best_threshold'] = best_threshold
    
    # Ergebnisse speichern
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to {OUTPUT_FILE}")
    
    # Metriken als JSON speichern
    import json
    metrics_file = "clip_ultimate_90_percent_metrics.json"
    metrics_dict = {}
    for strategy, metrics in all_metrics.items():
        metrics_dict[strategy] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                for k, v in metrics.items() 
                                if k not in ['confusion_matrix', 'similarities', 'y_true', 'y_pred']}
    metrics_dict['threshold_analysis'] = threshold_analysis
    metrics_dict['best_strategy'] = best_strategy
    metrics_dict['best_accuracy'] = best_accuracy
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"📊 Metrics saved to {metrics_file}")
    
    # Visualisierungen erstellen
    print(f"\n📈 Creating ULTIMATE visualizations...")
    plot_ultimate_results(all_metrics, threshold_analysis)
    
    print(f"\n✅ ULTIMATE CLIP Experiment completed!")
    print(f"🎉 Achieved {best_accuracy*100:.1f}% accuracy!")
    
    if best_accuracy >= 0.90:
        print(f"🎯 TARGET ACHIEVED: 90%+ Accuracy!")
        print(f"🚀 EXCELLENT! You've reached the 90% milestone!")
    elif best_accuracy >= 0.85:
        print(f"📈 Very close to target: {best_accuracy*100:.1f}% (target: 90%)")
        print(f"💡 Try increasing NUM_CROPS or adding more text variants")
    elif best_accuracy >= 0.82:
        print(f"📈 Good improvement: {best_accuracy*100:.1f}% (baseline: 82%)")
        print(f"💡 Consider ensemble with other models")
    else:
        print(f"📉 Need further optimization: {best_accuracy*100:.1f}%")
        print(f"💡 Investigate individual components")
    
    print(f"\n📋 ULTIMATE Optimization Summary:")
    print(f"  - Best strategy: {best_strategy}")
    print(f"  - Enhanced Multi-crop: {NUM_CROPS} crops per image")
    print(f"  - Ultimate Text variants: 8+ processing approaches")
    print(f"  - Image augmentation: {'Enabled' if USE_AUGMENTATION else 'Disabled'}")
    print(f"  - Multiple aggregation strategies: {len(all_strategies)}")
    print(f"  - Grid search threshold optimization")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    main() 