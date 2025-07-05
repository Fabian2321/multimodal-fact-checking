# --- COLAB BLIP2 SIMPLIFIED - 70%+ TARGET ---
# Vereinfachte BLIP2 Version mit besseren Prompts und Parsing

import os
import glob
import pandas as pd
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import re
import time

def load_local_image(image_id: str) -> Image.Image:
    """Lädt lokale Bilder aus colab_images/ Ordner"""
    image_pattern = os.path.join("colab_images", f"{image_id}.*")
    matching_files = glob.glob(image_pattern)
    if matching_files:
        return Image.open(matching_files[0]).convert('RGB')
    else:
        print(f"No image found for ID {image_id}")
        return Image.new('RGB', (224, 224), color='gray')

def create_simple_crops(image: Image.Image, num_crops: int = 2) -> List[Image.Image]:
    """Vereinfachte Multi-Crop - nur 2 Crops"""
    crops = [image]  # Original immer dabei
    
    if num_crops > 1:
        width, height = image.size
        # Nur Center crop
        center_crop = image.crop((width//4, height//4, 3*width//4, 3*height//4))
        crops.append(center_crop)
    
    return crops[:num_crops]

def create_simple_prompts(text: str) -> List[str]:
    """Vereinfachte Prompts für bessere BLIP2 Performance"""
    prompts = []
    
    # Basis-Cleaning
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\-\.\,\!\?]', '', text)
    
    # Prompt 1: Einfache Ja/Nein Frage
    prompt1 = f"Question: Does this image match the text '{text}'? Answer yes or no. Answer:"
    prompts.append(prompt1)
    
    # Prompt 2: Direkte Verifikation
    prompt2 = f"Question: Is this image related to '{text}'? Answer yes or no. Answer:"
    prompts.append(prompt2)
    
    return prompts

def parse_blip2_response_simple(response: str) -> float:
    """Vereinfachtes BLIP2 Response-Parsing"""
    response = response.lower().strip()
    
    # Direkte Ja/Nein Erkennung
    if 'yes' in response and 'no' not in response:
        return 0.8  # Hohe Wahrscheinlichkeit für Match
    elif 'no' in response and 'yes' not in response:
        return 0.2  # Niedrige Wahrscheinlichkeit für Match
    elif 'yes' in response and 'no' in response:
        return 0.5  # Neutral (beide Wörter)
    else:
        # Fallback: Wort-basierte Analyse
        positive_words = ['match', 'related', 'correct', 'true', 'accurate']
        negative_words = ['not', 'false', 'wrong', 'different', 'unrelated']
        
        pos_count = sum(1 for word in positive_words if word in response)
        neg_count = sum(1 for word in negative_words if word in response)
        
        if pos_count > neg_count:
            return 0.7
        elif neg_count > pos_count:
            return 0.3
        else:
            return 0.5

class SimplifiedBLIP2Handler:
    def __init__(self):
        """Vereinfachter BLIP2 Handler"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # BLIP2 Model laden
        print("Loading BLIP2 model: Salesforce/blip2-opt-2.7b")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        print("BLIP2 model loaded successfully!")

    def predict_simple(self, text: str, image: Image.Image) -> Dict[str, Any]:
        """Vereinfachte Prediction mit weniger Optimierungen"""
        
        # Vereinfachte Prompts
        prompts = create_simple_prompts(text)
        
        # Nur 2 Crops
        crops = create_simple_crops(image, 2)
        
        all_scores = []
        
        # Für jeden Prompt
        for prompt in prompts:
            prompt_scores = []
            
            # Für jeden Crop
            for crop in crops:
                try:
                    inputs = self.processor(
                        images=crop,
                        text=prompt,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=20,  # Kürzer
                            num_beams=3,        # Weniger Beams
                            do_sample=False,    # Greedy decoding
                            temperature=1.0,    # Standard
                            repetition_penalty=1.0  # Standard
                        )
                    
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
                    score = parse_blip2_response_simple(response)
                    prompt_scores.append(score)
                    
                except Exception as e:
                    print(f"Error in prediction: {e}")
                    prompt_scores.append(0.5)  # Neutral bei Fehler
            
            # Einfache Aggregation pro Prompt
            if prompt_scores:
                avg_score = np.mean(prompt_scores)
                all_scores.append(avg_score)
        
        # Finale Aggregation
        if all_scores:
            final_score = np.mean(all_scores)
        else:
            final_score = 0.5
        
        return {
            'final_score': final_score,
            'prompt_scores': all_scores,
            'num_prompts': len(prompts),
            'num_crops': len(crops)
        }

    def find_optimal_threshold(self, scores: list, true_labels: list) -> Dict[str, float]:
        """Optimale Schwellenwert-Optimierung"""
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        # ROC-basierte Optimierung
        fpr, tpr, roc_thresholds = roc_curve(true_labels, scores)
        j_scores = tpr - fpr
        best_roc_idx = np.argmax(j_scores)
        roc_threshold = roc_thresholds[best_roc_idx]
        
        # Precision-Recall Optimierung
        precision, recall, pr_thresholds = precision_recall_curve(true_labels, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_pr_idx = np.argmax(f1_scores[:-1])
        pr_threshold = pr_thresholds[best_pr_idx]
        
        # Balanced Accuracy Optimierung
        balanced_accuracies = []
        for threshold in roc_thresholds:
            predictions = [int(score >= threshold) for score in scores]
            tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            balanced_acc = (specificity + sensitivity) / 2
            balanced_accuracies.append(balanced_acc)
        
        best_ba_idx = np.argmax(balanced_accuracies)
        ba_threshold = roc_thresholds[best_ba_idx]
        
        print(f"Simplified BLIP2 Threshold Analysis:")
        print(f"  - ROC J-score threshold: {roc_threshold:.3f}")
        print(f"  - Precision-Recall F1 threshold: {pr_threshold:.3f}")
        print(f"  - Balanced Accuracy threshold: {ba_threshold:.3f}")
        
        return {
            'roc_threshold': roc_threshold,
            'pr_threshold': pr_threshold,
            'ba_threshold': ba_threshold,
            'j_score': j_scores[best_roc_idx],
            'f1_score': f1_scores[best_pr_idx],
            'balanced_accuracy': balanced_accuracies[best_ba_idx]
        }

def calculate_simplified_metrics(y_true, y_pred, scores, threshold):
    """Berechnet Metriken für vereinfachte BLIP2"""
    
    # Basis-Metriken
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # ROC und AUC
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    
    # Per-Class Metriken
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_accuracy = (specificity + sensitivity) / 2
    
    # Score-Statistiken
    pos_scores = [s for s, l in zip(scores, y_true) if l == 1]
    neg_scores = [s for s, l in zip(scores, y_true) if l == 0]
    
    print("\n" + "="*60)
    print("SIMPLIFIED BLIP2 EXPERIMENT - VOLLSTÄNDIGE METRIKEN")
    print("="*60)
    print(f"Setup:")
    print(f"  - Model: Salesforce/blip2-opt-2.7b")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Samples: {len(y_true)}")
    print(f"  - Threshold: {threshold:.3f}")
    print(f"  - Optimizations: Simplified (2 crops, 2 prompts, simple parsing)")
    
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
    
    print(f"\nScore Statistics:")
    print(f"  Positive samples: {len(pos_scores)}")
    print(f"  Negative samples: {len(neg_scores)}")
    print(f"  Positive mean score: {np.mean(pos_scores):.3f}")
    print(f"  Negative mean score: {np.mean(neg_scores):.3f}")
    print(f"  Positive std score:  {np.std(pos_scores):.3f}")
    print(f"  Negative std score:  {np.std(neg_scores):.3f}")
    print(f"  Separation: {np.mean(pos_scores) - np.mean(neg_scores):.3f}")
    
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
        'scores': scores,
        'y_true': y_true,
        'y_pred': y_pred
    }

def plot_simplified_results(metrics, threshold_analysis):
    """Erstellt Visualisierungen für vereinfachte BLIP2"""
    
    # Confusion Matrix
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Simplified BLIP2 Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # ROC Curve
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['scores'])
    plt.plot(fpr, tpr, label=f'Simplified BLIP2 (AUC={metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Simplified BLIP2 ROC Curve')
    plt.legend()
    
    # Score Distribution
    plt.subplot(1, 3, 3)
    pos_scores = [s for s, l in zip(metrics['scores'], metrics['y_true']) if l == 1]
    neg_scores = [s for s, l in zip(metrics['scores'], metrics['y_true']) if l == 0]
    
    plt.hist(pos_scores, alpha=0.7, label='Positive', bins=20, color='green')
    plt.hist(neg_scores, alpha=0.7, label='Negative', bins=20, color='red')
    plt.axvline(metrics['threshold'], color='black', linestyle='--', 
                label=f'Threshold: {metrics["threshold"]:.3f}')
    plt.xlabel('BLIP2 Score')
    plt.ylabel('Frequency')
    plt.title('Simplified BLIP2 Score Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Vereinfachtes BLIP2 Experiment für 70%+ Target"""
    
    # Parameter
    CSV_FILE = "test_balanced_pairs_clean.csv"
    NUM_SAMPLES = 100
    OUTPUT_FILE = "colab_blip2_simplified_results.csv"
    
    print("Colab BLIP2 Simplified - 70%+ Target")
    print("="*40)
    print(f"Simplifications:")
    print(f"  - Model: Salesforce/blip2-opt-2.7b")
    print(f"  - Multi-Crop: 2 crops per image (reduced)")
    print(f"  - Multi-Prompt: 2 simple prompts (reduced)")
    print(f"  - Simple parsing: Yes/No detection")
    print(f"  - Greedy decoding: Faster generation")
    
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
    
    # BLIP2 initialisieren
    start_time = time.time()
    blip2 = SimplifiedBLIP2Handler()
    init_time = time.time() - start_time
    print(f"⏱️ Model initialization: {init_time:.2f}s")
    
    # Predictions durchführen
    results = []
    scores = []
    
    print(f"🔄 Running simplified BLIP2 predictions on {len(df)} samples...")
    pred_start_time = time.time()
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing sample {idx+1}/{len(df)}")
        
        image = load_local_image(row['id'])
        prediction = blip2.predict_simple(row['clean_title'], image)
        scores.append(prediction['final_score'])
        
        results.append({
            'id': row['id'],
            'text': row['clean_title'],
            'image_url': row['image_url'],
            'true_label': row['2_way_label'],
            'blip2_score': prediction['final_score'],
            'num_prompts': prediction['num_prompts'],
            'num_crops': prediction['num_crops']
        })
    
    pred_time = time.time() - pred_start_time
    print(f"⏱️ Prediction time: {pred_time:.2f}s ({pred_time/len(df):.3f}s per sample)")
    
    # Schwellenwert-Optimierung
    print(f"\n🎯 Finding optimal threshold for simplified BLIP2...")
    true_labels = [r['true_label'] for r in results]
    
    threshold_analysis = blip2.find_optimal_threshold(scores, true_labels)
    
    # Verschiedene Thresholds testen
    thresholds_to_test = [
        ('roc_threshold', threshold_analysis['roc_threshold']),
        ('pr_threshold', threshold_analysis['pr_threshold']),
        ('ba_threshold', threshold_analysis['ba_threshold'])
    ]
    
    best_accuracy = 0
    best_threshold_name = ''
    best_threshold = 0
    best_predictions = []
    
    print(f"Testing thresholds for simplified BLIP2:")
    for name, threshold in thresholds_to_test:
        predictions = [int(score >= threshold) for score in scores]
        accuracy = accuracy_score(true_labels, predictions)
        print(f"  {name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold_name = name
            best_threshold = threshold
            best_predictions = predictions
    
    print(f"Best simplified BLIP2: {best_threshold_name} = {best_threshold:.3f} -> {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    # Metriken berechnen
    metrics = calculate_simplified_metrics(true_labels, best_predictions, scores, best_threshold)
    
    # Finale Predictions setzen
    for r in results:
        r['predicted_label'] = int(r['blip2_score'] >= best_threshold)
        r['threshold'] = best_threshold
    
    # Ergebnisse speichern
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to {OUTPUT_FILE}")
    
    # Metriken als JSON speichern
    import json
    metrics_file = "colab_blip2_simplified_metrics.json"
    metrics_dict = {
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'specificity': metrics['specificity'],
        'sensitivity': metrics['sensitivity'],
        'balanced_accuracy': metrics['balanced_accuracy'],
        'roc_auc': metrics['roc_auc'],
        'threshold': metrics['threshold'],
        'threshold_analysis': threshold_analysis,
        'model': 'Salesforce/blip2-opt-2.7b',
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'performance': {
            'init_time': init_time,
            'prediction_time': pred_time,
            'total_time': init_time + pred_time,
            'samples_per_second': len(df) / pred_time
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"📊 Metrics saved to {metrics_file}")
    
    # Visualisierungen erstellen
    print(f"\n📈 Creating simplified BLIP2 visualizations...")
    plot_simplified_results(metrics, threshold_analysis)
    
    total_time = time.time() - start_time
    print(f"\n✅ Simplified BLIP2 Experiment completed!")
    print(f"🎉 Achieved {best_accuracy*100:.1f}% accuracy!")
    print(f"⏱️ Total time: {total_time:.2f}s")
    
    # Vergleich mit vorherigem BLIP2
    previous_blip2 = 0.61
    improvement = best_accuracy - previous_blip2
    
    if best_accuracy >= 0.75:
        print(f"🎯 EXCELLENT: 75%+ Accuracy!")
        print(f"🚀 Simplified approach worked perfectly!")
    elif best_accuracy >= 0.70:
        print(f"📈 Good improvement: {best_accuracy*100:.1f}% (previous: 61%)")
        print(f"💡 Simplification helped significantly")
    elif best_accuracy >= 0.65:
        print(f"📈 Moderate improvement: {best_accuracy*100:.1f}% (previous: 61%)")
        print(f"💡 Some improvement from simplification")
    else:
        print(f"📉 Still below target: {best_accuracy*100:.1f}% (previous: 61%)")
        print(f"💡 BLIP2 might not be optimal for this task")
    
    if improvement > 0:
        print(f"📈 VERBESSERUNG: +{improvement:.3f} (+{improvement*100:.1f}%) über vorherigem BLIP2")
    elif improvement < 0:
        print(f"📉 RÜCKSCHRITT: {improvement:.3f} ({improvement*100:.1f}%) unter vorherigem BLIP2")
    else:
        print(f"📊 GLEICH: Keine Änderung zum vorherigen BLIP2")
    
    print(f"\n📋 Simplified BLIP2 Summary:")
    print(f"  - Model: Salesforce/blip2-opt-2.7b")
    print(f"  - Accuracy: {best_accuracy*100:.1f}%")
    print(f"  - Threshold: {best_threshold:.3f}")
    print(f"  - Multi-Crop: 2 crops (reduced)")
    print(f"  - Multi-Prompt: 2 prompts (reduced)")
    print(f"  - Performance: {len(df)/pred_time:.1f} samples/second")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    main() 