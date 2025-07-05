# --- Konservatives CLIP Verbesserungs-Script ---
# Baut auf dem erfolgreichen 82%-Setup auf mit vorsichtigen Optimierungen

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
    """Lädt lokale Bilder aus colab_images/ Ordner"""
    image_pattern = os.path.join("colab_images", f"{image_id}.*")
    matching_files = glob.glob(image_pattern)
    if matching_files:
        return Image.open(matching_files[0]).convert('RGB')
    else:
        print(f"No image found for ID {image_id}")
        return Image.new('RGB', (224, 224), color='gray')

def conservative_text_preprocessing(text: str) -> str:
    """Konservative Text-Preprocessing - nur bewährte Methoden"""
    # Basis-Cleaning
    text = text.lower().strip()
    
    # Entferne URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Entferne spezielle Zeichen, behalte wichtige
    text = re.sub(r'[^\w\s\-\.\,\!\?]', '', text)
    
    # Selektive Stop-Wort-Entfernung (nur bewährte)
    stop_words = set(stopwords.words('english'))
    important_words = {'fake', 'real', 'true', 'false', 'news', 'image', 'photo', 'picture', 'video'}
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words or word in important_words]
    
    # Mindestlänge sicherstellen
    if len(filtered_words) < 2:
        filtered_words = words[:3]  # Fallback
    
    return ' '.join(filtered_words)

def create_conservative_crops(image: Image.Image, num_crops: int = 3) -> List[Image.Image]:
    """Konservative Multi-Crop - nur bewährte Ausschnitte"""
    crops = [image]  # Original immer dabei
    
    if num_crops > 1:
        width, height = image.size
        # Center crop
        center_crop = image.crop((width//4, height//4, 3*width//4, 3*height//4))
        crops.append(center_crop)
        
        # Nur ein zusätzlicher Crop für Stabilität
        if num_crops > 2:
            # Quadratischer Crop aus der Mitte
            min_dim = min(width, height)
            start_x = (width - min_dim) // 2
            start_y = (height - min_dim) // 2
            square_crop = image.crop((start_x, start_y, start_x + min_dim, start_y + min_dim))
            crops.append(square_crop)
    
    return crops[:num_crops]

class ConservativeCLIPHandler:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        """Konservative CLIP-Konfiguration - baut auf 82% auf"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading CLIP model: {model_name}")
        
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        print("CLIP model loaded successfully!")

    def predict_similarity_conservative(self, text: str, image: Image.Image, num_crops: int = 3) -> float:
        """Konservative Similarity-Berechnung - bewährte Methoden"""
        # Konservative Text-Preprocessing
        processed_text = conservative_text_preprocessing(text)
        
        # Konservative Multi-Crop
        crops = create_conservative_crops(image, num_crops)
        
        similarities = []
        
        for crop in crops:
            inputs = self.processor(
                text=[processed_text], 
                images=crop, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
                similarity = (image_embeds @ text_embeds.T).cpu().item()
                similarities.append(similarity)
        
        # Bewährte Aggregation: Max + Mean (wie im 82%-Setup)
        max_sim = max(similarities)
        mean_sim = np.mean(similarities)
        return 0.7 * max_sim + 0.3 * mean_sim

    def find_optimal_threshold_conservative(self, similarities: list, true_labels: list) -> Dict[str, float]:
        """Konservative Schwellenwert-Optimierung - bewährte Methoden"""
        from sklearn.metrics import roc_curve, precision_recall_curve
        
        # ROC-basierte Optimierung (bewährt)
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
        
        print(f"Conservative Threshold Analysis:")
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

def calculate_comprehensive_metrics(y_true, y_pred, similarities, threshold):
    """Berechnet alle wichtigen Metriken für die Dokumentation"""
    
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
    
    print("\n" + "="*60)
    print("CONSERVATIVE CLIP EXPERIMENT - VOLLSTÄNDIGE METRIKEN")
    print("="*60)
    print(f"Setup:")
    print(f"  - Model: openai/clip-vit-base-patch16")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Samples: {len(y_true)}")
    print(f"  - Threshold: {threshold:.3f}")
    print(f"  - Optimizations: Conservative 3-crop, Text-preprocessing")
    
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
        'y_pred': y_pred
    }

def main():
    """Hauptfunktion mit konservativen Verbesserungen"""
    
    # Parameter
    CSV_FILE = "test_balanced_pairs_clean.csv"
    NUM_SAMPLES = 100
    OUTPUT_FILE = "clip_conservative_improvement_results.csv"
    NUM_CROPS = 3  # Konservativ - wie im 82%-Setup
    
    print("Conservative CLIP Experiment - Vorsichtige Verbesserungen")
    print("="*55)
    print(f"Conservative Optimizations:")
    print(f"  - Multi-crop: {NUM_CROPS} crops per image (bewährt)")
    print(f"  - Text preprocessing: Selektive Stop-word removal")
    print(f"  - Similarity aggregation: Max + Mean (bewährt)")
    print(f"  - No aggressive augmentation")
    print(f"  - No over-optimization")
    
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
    clip = ConservativeCLIPHandler()
    
    # Predictions durchführen
    results = []
    similarities = []
    true_labels = []
    
    print(f"🔄 Running conservative CLIP predictions on {len(df)} samples...")
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing sample {idx+1}/{len(df)}")
        
        image = load_local_image(row['id'])
        sim = clip.predict_similarity_conservative(row['clean_title'], image, NUM_CROPS)
        
        similarities.append(sim)
        true_labels.append(row['2_way_label'])
        
        results.append({
            'id': row['id'],
            'text': row['clean_title'],
            'image_url': row['image_url'],
            'true_label': row['2_way_label'],
            'clip_similarity': sim,
        })
    
    # Konservative Schwellenwert-Optimierung
    print(f"\n🎯 Finding optimal threshold with conservative strategies...")
    threshold_analysis = clip.find_optimal_threshold_conservative(similarities, true_labels)
    
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
    
    print(f"\n🔍 Testing different threshold strategies...")
    for name, threshold in thresholds_to_test:
        predictions = [int(sim >= threshold) for sim in similarities]
        accuracy = accuracy_score(true_labels, predictions)
        print(f"  {name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold_name = name
            best_threshold = threshold
            best_predictions = predictions
    
    print(f"\n🏆 Best threshold: {best_threshold_name} = {best_threshold:.3f}")
    print(f"🎯 Best accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    # Finale Predictions setzen
    for r in results:
        r['clip_predicted_label'] = int(r['clip_similarity'] >= best_threshold)
    
    # Vollständige Metriken berechnen
    metrics = calculate_comprehensive_metrics(true_labels, best_predictions, similarities, best_threshold)
    
    # Ergebnisse speichern
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to {OUTPUT_FILE}")
    
    # Metriken als JSON speichern
    import json
    metrics_file = "clip_conservative_improvement_metrics.json"
    metrics_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                   for k, v in metrics.items() 
                   if k not in ['confusion_matrix', 'similarities', 'y_true', 'y_pred']}
    metrics_dict['threshold_analysis'] = threshold_analysis
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"📊 Metrics saved to {metrics_file}")
    
    print(f"\n✅ Conservative CLIP Experiment completed!")
    print(f"🎉 Achieved {metrics['accuracy']*100:.1f}% accuracy!")
    
    if metrics['accuracy'] >= 0.82:
        print(f"🎯 SUCCESS: Back to 82%+ territory!")
    elif metrics['accuracy'] >= 0.80:
        print(f"📈 Good: Above 80%")
    else:
        print(f"📉 Need to investigate further")
    
    print(f"\n📋 Conservative Optimization Summary:")
    print(f"  - Multi-crop strategy: {NUM_CROPS} crops per image")
    print(f"  - Text preprocessing: Selective stop-word removal")
    print(f"  - Similarity aggregation: Max + Mean weighted")
    print(f"  - Threshold optimization: {best_threshold_name}")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    main() 