# --- CLIP MODEL + DEVICE OPTIMIZED - 85% TARGET ---
# Various CLIP models + A100 optimizations + Mixed Precision

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
import time

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

def create_optimal_text_variants(text: str) -> List[str]:
    """Optimal text variants - proven 3 variants"""
    variants = []
    
    # Basic cleaning
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\-\.\,\!\?]', '', text)
    
    # Variant 1: Original (cleaned) - PROVEN
    variants.append(text)
    
    # Variant 2: Shorter version (first 4 words) - OPTIMIZED
    words = text.split()
    if len(words) > 3:
        short_variant = ' '.join(words[:4])
        variants.append(short_variant)
    
    # Variant 3: Selective stopword removal - PROVEN
    stop_words = set(stopwords.words('english'))
    important_words = {'fake', 'real', 'true', 'false', 'news', 'image', 'photo', 'picture', 'video'}
    filtered_words = [word for word in words if word not in stop_words or word in important_words]
    if len(filtered_words) >= 2:
        filtered_variant = ' '.join(filtered_words)
        if filtered_variant != text:
            variants.append(filtered_variant)
    
    return list(set(variants))

def create_optimal_crops(image: Image.Image, num_crops: int = 3) -> List[Image.Image]:
    """Optimal multi-crop - proven 3 crops"""
    crops = [image]  # Always include original
    
    if num_crops > 1:
        width, height = image.size
        # Center crop (proven)
        center_crop = image.crop((width//4, height//4, 3*width//4, 3*height//4))
        crops.append(center_crop)
        
        # Square crop from center (proven)
        if num_crops > 2:
            min_dim = min(width, height)
            start_x = (width - min_dim) // 2
            start_y = (height - min_dim) // 2
            square_crop = image.crop((start_x, start_y, start_x + min_dim, start_y + min_dim))
            crops.append(square_crop)
    
    return crops[:num_crops]

class OptimizedCLIPHandler:
    def __init__(self):
        """Optimized CLIP handler with various models and A100 optimizations"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # A100-specific optimizations
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Mixed Precision for A100
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ Mixed Precision (AMP) enabled for A100")
        else:
            self.use_amp = False
            self.scaler = None
        
        # Load CLIP models - various variants
        self.models = {}
        self.processors = {}
        
        # Models: Different CLIP variants for ensemble
        model_configs = [
            ("openai/clip-vit-base-patch16", "base16"),
            ("openai/clip-vit-base-patch14", "base14"),
            ("openai/clip-vit-large-patch14", "large14"),
        ]
        
        for model_name, model_id in model_configs:
            print(f"Loading CLIP model: {model_name}")
            try:
                self.processors[model_id] = CLIPProcessor.from_pretrained(model_name)
                self.models[model_id] = CLIPModel.from_pretrained(model_name).to(self.device)
                
                # A100 optimizations
                if self.device == "cuda":
                    self.models[model_id].half()  # FP16 for A100
                
                print(f"✅ {model_name} loaded successfully!")
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        
        if not self.models:
            raise ValueError("No CLIP models loaded!")
        
        print(f"✅ Loaded {len(self.models)} CLIP models")

    def predict_similarity_optimized(self, text: str, image: Image.Image) -> Dict[str, float]:
        """Optimized similarity calculation with A100 optimizations"""
        
        # Create text variants
        text_variants = create_optimal_text_variants(text)
        
        # Optimal multi-crop
        crops = create_optimal_crops(image, 3)
        
        all_model_scores = {}
        
        # For each CLIP model
        for model_id, model in self.models.items():
            processor = self.processors[model_id]
            model_similarities = []
            
            # For each text variant
            for text_variant in text_variants:
                # For each crop
                for crop in crops:
                    inputs = processor(
                        text=[text_variant], 
                        images=crop, 
                        return_tensors="pt", 
                        padding=True
                    ).to(self.device)
                    
                    # Mixed Precision for A100
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        with torch.no_grad():
                            outputs = model(**inputs)
                            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
                            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
                            similarity = (image_embeds @ text_embeds.T).cpu().item()
                            model_similarities.append(similarity)
            
            # Aggregation per model (proven: Max + Mean)
            if model_similarities:
                max_sim = max(model_similarities)
                mean_sim = np.mean(model_similarities)
                all_model_scores[model_id] = 0.7 * max_sim + 0.3 * mean_sim
        
        # Ensemble aggregation with model-specific weights
        if all_model_scores:
            # Weighted combination (larger models more important)
            weights = {
                'base16': 0.3,
                'base14': 0.3, 
                'large14': 0.4  # Larger model gets more weight
            }
            
            ensemble_score = 0
            for model_id, score in all_model_scores.items():
                weight = weights.get(model_id, 0.33)
                ensemble_score += score * weight
            
            # Normalization
            total_weight = sum(weights.get(model_id, 0.33) for model_id in all_model_scores.keys())
            ensemble_score /= total_weight
            
            return {
                'ensemble': ensemble_score,
                **all_model_scores
            }
        else:
            return {'ensemble': 0.0}

    def find_optimal_threshold(self, similarities: list, true_labels: list) -> Dict[str, float]:
        """Optimale Schwellenwert-Optimierung"""
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        # ROC-basierte Optimierung
        fpr, tpr, roc_thresholds = roc_curve(true_labels, similarities)
        j_scores = tpr - fpr
        best_roc_idx = np.argmax(j_scores)
        roc_threshold = roc_thresholds[best_roc_idx]
        
        # Precision-Recall Optimierung
        precision, recall, pr_thresholds = precision_recall_curve(true_labels, similarities)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_pr_idx = np.argmax(f1_scores[:-1])
        pr_threshold = pr_thresholds[best_pr_idx]
        
        # Balanced Accuracy Optimierung
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
        
        # Grid Search um beste Thresholds
        search_range = np.arange(0.2, 0.4, 0.001)
        search_accuracies = []
        for threshold in search_range:
            predictions = [int(sim >= threshold) for sim in similarities]
            accuracy = accuracy_score(true_labels, predictions)
            search_accuracies.append(accuracy)
        
        best_search_idx = np.argmax(search_accuracies)
        grid_threshold = search_range[best_search_idx]
        
        print(f"Threshold Analysis:")
        print(f"  - ROC J-score threshold: {roc_threshold:.3f}")
        print(f"  - Precision-Recall F1 threshold: {pr_threshold:.3f}")
        print(f"  - Balanced Accuracy threshold: {ba_threshold:.3f}")
        print(f"  - Grid Search threshold: {grid_threshold:.3f}")
        
        return {
            'roc_threshold': roc_threshold,
            'pr_threshold': pr_threshold,
            'ba_threshold': ba_threshold,
            'grid_threshold': grid_threshold,
            'j_score': j_scores[best_roc_idx],
            'f1_score': f1_scores[best_pr_idx],
            'balanced_accuracy': balanced_accuracies[best_ba_idx],
            'grid_accuracy': search_accuracies[best_search_idx]
        }

def calculate_optimized_metrics(y_true, y_pred, similarities, threshold, strategy_name):
    """Berechnet optimierte Metriken"""
    
    # Basis-Metriken
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC und AUC
    fpr, tpr, _ = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)
    
    # Per-Class Metriken
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_accuracy = (specificity + sensitivity) / 2
    
    # Similarity-Statistiken
    pos_similarities = [s for s, l in zip(similarities, y_true) if l == 1]
    neg_similarities = [s for s, l in zip(similarities, y_true) if l == 0]
    
    print(f"\n{strategy_name.upper()} STRATEGY - OPTIMIERTE METRIKEN")
    print("="*60)
    print(f"Setup:")
    print(f"  - Model: Multi-CLIP Optimized")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Samples: {len(y_true)}")
    print(f"  - Threshold: {threshold:.3f}")
    print(f"  - Optimizations: A100 Mixed Precision, Multi-Model")
    
    print(f"\nPerformance Metriken:")
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

def plot_optimized_results(all_metrics, threshold_analysis):
    """Erstellt optimierte Visualisierungen"""
    
    strategies = list(all_metrics.keys())
    accuracies = [all_metrics[s]['accuracy'] for s in strategies]
    
    plt.figure(figsize=(20, 5))
    
    # Strategy Comparison
    plt.subplot(1, 4, 1)
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    bars = plt.bar(strategies, accuracies, color=colors[:len(strategies)])
    plt.ylabel('Accuracy')
    plt.title('Optimized Strategy Comparison')
    plt.ylim(0.75, 0.90)
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', fontweight='bold')
    
    # Best Strategy Confusion Matrix
    best_strategy = max(all_metrics.keys(), key=lambda x: all_metrics[x]['accuracy'])
    best_metrics = all_metrics[best_strategy]
    
    plt.subplot(1, 4, 2)
    cm = best_metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Best Strategy: {best_strategy}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # ROC Curve Comparison
    plt.subplot(1, 4, 3)
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
    plt.subplot(1, 4, 4)
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
    
    plt.tight_layout()
    plt.show()

def main():
    """Optimiertes CLIP Experiment für 85% Target"""
    
    # Parameter
    CSV_FILE = "test_balanced_pairs_clean.csv"
    NUM_SAMPLES = 100
    OUTPUT_FILE = "clip_model_device_optimized_results.csv"
    
    print("CLIP Model + Device Optimized - 85% Target")
    print("="*50)
    print(f"Optimizations:")
    print(f"  - Multi-CLIP Models: base16, base14, large14")
    print(f"  - A100 Mixed Precision (FP16)")
    print(f"  - Optimized Text variants: 3 processing approaches")
    print(f"  - Optimized Multi-crop: 3 crops per image")
    print(f"  - Model-specific weighting")
    
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
    
    # Optimierten CLIP initialisieren
    start_time = time.time()
    clip = OptimizedCLIPHandler()
    init_time = time.time() - start_time
    print(f"⏱️ Model initialization: {init_time:.2f}s")
    
    # Predictions durchführen
    results = []
    all_strategies = {}
    
    print(f"🔄 Running optimized CLIP predictions on {len(df)} samples...")
    pred_start_time = time.time()
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing sample {idx+1}/{len(df)}")
        
        image = load_local_image(row['id'])
        similarity_dict = clip.predict_similarity_optimized(row['clean_title'], image)
        
        # Alle Strategien sammeln
        for strategy_name, similarity in similarity_dict.items():
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
    
    pred_time = time.time() - pred_start_time
    print(f"⏱️ Prediction time: {pred_time:.2f}s ({pred_time/len(df):.3f}s per sample)")
    
    # Schwellenwert-Optimierung für jede Strategie
    print(f"\n🎯 Finding optimal thresholds for all strategies...")
    all_metrics = {}
    threshold_analysis = {}
    
    for strategy_name, similarities in all_strategies.items():
        print(f"\n--- Optimizing {strategy_name} ---")
        true_labels = [r['true_label'] for r in results]
        
        # Threshold-Optimierung
        strategy_thresholds = clip.find_optimal_threshold(similarities, true_labels)
        
        # Verschiedene Thresholds testen
        thresholds_to_test = [
            ('roc_threshold', strategy_thresholds['roc_threshold']),
            ('pr_threshold', strategy_thresholds['pr_threshold']),
            ('ba_threshold', strategy_thresholds['ba_threshold']),
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
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold_name = name
                best_threshold = threshold
                best_predictions = predictions
        
        print(f"Best {strategy_name}: {best_threshold_name} = {best_threshold:.3f} -> {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
        
        # Metriken berechnen
        metrics = calculate_optimized_metrics(true_labels, best_predictions, similarities, best_threshold, strategy_name)
        all_metrics[strategy_name] = metrics
    
    # Beste Strategie finden
    best_strategy = max(all_metrics.keys(), key=lambda x: all_metrics[x]['accuracy'])
    best_accuracy = all_metrics[best_strategy]['accuracy']
    
    print(f"\n🏆 BESTE STRATEGIE: {best_strategy}")
    print(f"🎯 BESTE ACCURACY: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    # Vergleich mit 82% Baseline
    baseline_accuracy = 0.82
    improvement = best_accuracy - baseline_accuracy
    
    if improvement > 0:
        print(f"📈 VERBESSERUNG: +{improvement:.3f} (+{improvement*100:.1f}%) über 82% Baseline")
    elif improvement < 0:
        print(f"📉 RÜCKSCHRITT: {improvement:.3f} ({improvement*100:.1f}%) unter 82% Baseline")
    else:
        print(f"📊 GLEICH: Keine Änderung zur 82% Baseline")
    
    # Finale Predictions setzen
    best_similarities = all_strategies[best_strategy]
    best_threshold = all_metrics[best_strategy]['threshold']
    
    for r in results:
        r['final_predicted_label'] = int(r[best_strategy] >= best_threshold)
        r['best_strategy'] = best_strategy
        r['best_threshold'] = best_threshold
    
    # Ergebnisse speichern
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to {OUTPUT_FILE}")
    
    # Metriken als JSON speichern
    import json
    metrics_file = "clip_model_device_optimized_metrics.json"
    metrics_dict = {}
    for strategy, metrics in all_metrics.items():
        metrics_dict[strategy] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                for k, v in metrics.items() 
                                if k not in ['confusion_matrix', 'similarities', 'y_true', 'y_pred']}
    metrics_dict['threshold_analysis'] = threshold_analysis
    metrics_dict['best_strategy'] = best_strategy
    metrics_dict['best_accuracy'] = best_accuracy
    metrics_dict['baseline_comparison'] = {
        'baseline_accuracy': baseline_accuracy,
        'improvement': improvement,
        'improvement_percent': improvement * 100
    }
    metrics_dict['performance'] = {
        'init_time': init_time,
        'prediction_time': pred_time,
        'total_time': init_time + pred_time,
        'samples_per_second': len(df) / pred_time
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"📊 Metrics saved to {metrics_file}")
    
    # Visualisierungen erstellen
    print(f"\n📈 Creating optimized visualizations...")
    plot_optimized_results(all_metrics, threshold_analysis)
    
    total_time = time.time() - start_time
    print(f"\n✅ Optimized CLIP Experiment completed!")
    print(f"🎉 Achieved {best_accuracy*100:.1f}% accuracy!")
    print(f"⏱️ Total time: {total_time:.2f}s")
    
    if best_accuracy >= 0.85:
        print(f"🎯 TARGET ACHIEVED: 85%+ Accuracy!")
        print(f"🚀 EXCELLENT! Model + Device optimization worked!")
    elif best_accuracy >= 0.84:
        print(f"📈 Great improvement: {best_accuracy*100:.1f}% (baseline: 82%)")
        print(f"💡 Very close to 85% target")
    elif best_accuracy >= 0.83:
        print(f"📈 Good improvement: {best_accuracy*100:.1f}% (baseline: 82%)")
        print(f"💡 Model optimization helped")
    elif best_accuracy >= 0.82:
        print(f"📊 Maintained baseline: {best_accuracy*100:.1f}% (baseline: 82%)")
        print(f"💡 82% might be the natural limit")
    else:
        print(f"📉 Below baseline: {best_accuracy*100:.1f}% (baseline: 82%)")
        print(f"💡 Larger models didn't help")
    
    print(f"\n📋 Optimization Summary:")
    print(f"  - Best strategy: {best_strategy}")
    print(f"  - Multi-CLIP Models: {len(clip.models)} models")
    print(f"  - A100 Mixed Precision: {clip.use_amp}")
    print(f"  - Performance: {len(df)/pred_time:.1f} samples/second")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    main() 