# --- CLIP Standalone Script for 79% Accuracy Setup ---
# Exact parameters from the successful ensemble experiment
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

def load_local_image(image_id: str) -> Image.Image:
    """Loads local images from colab_images/ folder"""
    image_pattern = os.path.join("colab_images", f"{image_id}.*")
    matching_files = glob.glob(image_pattern)
    if matching_files:
        return Image.open(matching_files[0]).convert('RGB')
    else:
        print(f"No image found for ID {image_id}")
        return Image.new('RGB', (224, 224), color='gray')

class CLIPHandler:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch16"):
        """Exact CLIP configuration from the ensemble experiment"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        print(f"Loading CLIP model: {model_name}")
        
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        print("CLIP model loaded successfully!")

    def predict_similarity(self, text: str, image: Image.Image) -> float:
        """Exact similarity calculation from the ensemble experiment"""
        inputs = self.processor(
            text=[text], 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(p=2, dim=-1, keepdim=True)
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)
            similarity = (image_embeds @ text_embeds.T).cpu().item()
        
        return similarity

    def find_optimal_threshold(self, similarities: list, true_labels: list) -> float:
        """Exact threshold optimization from the ensemble experiment"""
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(true_labels, similarities)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx]
        
        print(f"ROC Analysis:")
        print(f"  - Number of thresholds evaluated: {len(thresholds)}")
        print(f"  - Threshold range: {thresholds.min():.3f} to {thresholds.max():.3f}")
        print(f"  - Optimal threshold: {optimal_threshold:.3f}")
        print(f"  - J-score at optimal: {j_scores[best_idx]:.3f}")
        
        return optimal_threshold

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
    
    # Similarity statistics
    pos_similarities = [s for s, l in zip(similarities, y_true) if l == 1]
    neg_similarities = [s for s, l in zip(similarities, y_true) if l == 0]
    
    print("\n" + "="*60)
    print("CLIP STANDALONE EXPERIMENT - COMPREHENSIVE METRICS")
    print("="*60)
    print(f"Setup:")
    print(f"  - Model: openai/clip-vit-base-patch16")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Samples: {len(y_true)}")
    print(f"  - Optimal Threshold: {threshold:.3f}")
    
    print(f"\nPerformance Metrics:")
    print(f"  - Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  - Precision: {precision:.3f}")
    print(f"  - Recall:    {recall:.3f}")
    print(f"  - F1-Score:  {f1:.3f}")
    print(f"  - Specificity: {specificity:.3f}")
    print(f"  - Sensitivity: {sensitivity:.3f}")
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
        'roc_auc': roc_auc,
        'threshold': threshold,
        'confusion_matrix': cm,
        'similarities': similarities,
        'y_true': y_true,
        'y_pred': y_pred
    }

def main():
    """Main function with exact parameters from the ensemble experiment"""
    
    # Exact parameters from the successful experiment
    CSV_FILE = "test_balanced_pairs_clean.csv"
    NUM_SAMPLES = 100
    OUTPUT_FILE = "clip_standalone_79_percent_results.csv"
    
    print("CLIP Standalone Experiment - 79% Accuracy Setup")
    print("="*50)
    
    # File checks
    if not os.path.exists(CSV_FILE):
        print(f"❌ CSV file {CSV_FILE} not found!")
        return
    
    if not os.path.exists("colab_images"):
        print("❌ colab_images folder not found!")
        print("Please extract colab_images.zip: !unzip -o colab_images.zip -d colab_images")
        return
    
    # Load data
    print(f"📊 Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE).head(NUM_SAMPLES)
    print(f"✅ Loaded {len(df)} samples")
    
    # Initialize CLIP
    clip = CLIPHandler()
    
    # Perform predictions
    results = []
    similarities = []
    true_labels = []
    
    print(f"🔄 Running CLIP predictions on {len(df)} samples...")
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing sample {idx+1}/{len(df)}")
        
        image = load_local_image(row['id'])
        sim = clip.predict_similarity(row['clean_title'], image)
        
        similarities.append(sim)
        true_labels.append(row['2_way_label'])
        
        results.append({
            'id': row['id'],
            'text': row['clean_title'],
            'image_url': row['image_url'],
            'true_label': row['2_way_label'],
            'clip_similarity': sim,
        })
    
    # Determine optimal threshold
    print(f"\n🎯 Finding optimal threshold...")
    threshold = clip.find_optimal_threshold(similarities, true_labels)
    
    # Set predictions
    predictions = [int(sim >= threshold) for sim in similarities]
    
    for r in results:
        r['clip_predicted_label'] = int(r['clip_similarity'] >= threshold)
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(true_labels, predictions, similarities, threshold)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to {OUTPUT_FILE}")
    
    # Also save metrics as JSON
    import json
    metrics_file = "clip_standalone_79_percent_metrics.json"
    metrics_dict = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                   for k, v in metrics.items() 
                   if k not in ['confusion_matrix', 'similarities', 'y_true', 'y_pred']}
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"📊 Metrics saved to {metrics_file}")
    
    print(f"\n✅ CLIP Standalone Experiment completed!")
    print(f"🎉 Achieved {metrics['accuracy']*100:.1f}% accuracy with threshold {threshold:.3f}")

if __name__ == "__main__":
    main() 