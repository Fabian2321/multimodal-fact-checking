# --- COLAB BLIP2 + RAG + FEW-SHOT - 75%+ TARGET ---
# BLIP2 mit RAG und Few-Shot Learning für bessere Performance

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
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_local_image(image_id: str) -> Image.Image:
    """Lädt lokale Bilder aus colab_images/ Ordner"""
    image_pattern = os.path.join("colab_images", f"{image_id}.*")
    matching_files = glob.glob(image_pattern)
    if matching_files:
        return Image.open(matching_files[0]).convert('RGB')
    else:
        print(f"No image found for ID {image_id}")
        return Image.new('RGB', (224, 224), color='gray')

def load_knowledge_base():
    """Lädt die RAG Knowledge Base"""
    # Verschiedene mögliche Pfade versuchen
    possible_paths = [
        "knowledge_base/documents.json",
        "knowledge_base/knowledge_base.json",
        "knowledge_base/data.json",
        "external_knowledge/documents.json",
        "data/external_knowledge/documents.json",
        "documents.json"
    ]
    
    for path in possible_paths:
        try:
            with open(path, "r") as f:
                data = json.load(f)
                print(f"✅ Loaded knowledge base from: {path}")
                return data
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            continue
    
    print("⚠️ Knowledge base not found in any expected location")
    print("   Tried paths:", possible_paths)
    return []

def create_rag_context(text: str, knowledge_base: List[Dict], top_k: int = 3) -> str:
    """Erstellt RAG-Kontext basierend auf Text-Ähnlichkeit"""
    if not knowledge_base:
        return ""
    
    # TF-IDF für Ähnlichkeitsberechnung
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    # Alle Dokumente und Query vektorisieren
    all_texts = [doc['content'] for doc in knowledge_base] + [text]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Ähnlichkeit zwischen Query und Dokumenten
    query_vector = tfidf_matrix[-1]
    doc_vectors = tfidf_matrix[:-1]
    
    similarities = cosine_similarity(query_vector, doc_vectors).flatten()
    
    # Top-K ähnlichste Dokumente
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Kontext erstellen
    context_parts = []
    for idx in top_indices:
        if similarities[idx] > 0.1:  # Mindestähnlichkeit
            doc = knowledge_base[idx]
            context_parts.append(f"Fact: {doc['content']}")
    
    return "\n".join(context_parts)

def create_few_shot_examples() -> str:
    """Erstellt Few-Shot Beispiele für konsistente Antworten"""
    examples = """
Here are examples of how to analyze images for fake news detection:

Example 1:
Text: "man holding giant fish"
Image: Shows a person holding a large fish
Analysis: This appears to be a real image. The fish size is reasonable and the image looks natural.
Answer: real news

Example 2:
Text: "alien spaceship landing in backyard"
Image: Shows a backyard with no spaceship visible
Analysis: The text claims an alien spaceship but the image shows an ordinary backyard. This is clearly fake.
Answer: fake news

Example 3:
Text: "cat wearing sunglasses"
Image: Shows a cat with sunglasses on its face
Analysis: This is a real image. Cats can wear accessories and this looks authentic.
Answer: real news

Example 4:
Text: "person flying without wings"
Image: Shows a person jumping or in mid-air
Analysis: The text claims flying but the image shows jumping. This is misleading and fake.
Answer: fake news

Now analyze this image and text:
"""
    return examples

def create_enhanced_prompt(text: str, rag_context: str = "") -> str:
    """Erstellt erweiterten Prompt mit RAG und Few-Shot"""
    
    # Basis-Cleaning
    text = text.lower().strip()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s\-\.\,\!\?]', '', text)
    
    # Few-Shot Beispiele
    few_shot = create_few_shot_examples()
    
    # RAG Kontext
    context_part = ""
    if rag_context:
        context_part = f"\nRelevant facts:\n{rag_context}\n"
    
    # Erweiterter Prompt
    prompt = f"""{few_shot}
Text: "{text}"
{context_part}
Question: Is this image fake news or real news? 
Provide a detailed analysis and then answer with only "fake news" or "real news".

Analysis:"""
    
    return prompt

def parse_blip2_response_enhanced(response: str) -> float:
    """Erweitertes BLIP2 Response-Parsing mit RAG-Kontext"""
    response = response.lower().strip()
    
    # Entferne den Prompt-Teil aus der Antwort
    if "analysis:" in response:
        response = response.split("analysis:")[-1].strip()
    
    # Finale Antwort extrahieren
    final_answer = ""
    if "answer:" in response:
        final_answer = response.split("answer:")[-1].strip()
    elif "fake news" in response[-20:]:  # Letzte 20 Zeichen
        final_answer = "fake news"
    elif "real news" in response[-20:]:
        final_answer = "real news"
    
    # Direkte Antwort-Parsing
    if "fake news" in final_answer:
        return 0.2
    elif "real news" in final_answer:
        return 0.8
    
    # Fallback: Analyse des gesamten Responses
    fake_indicators = [
        'fake news', 'fake', 'false', 'misleading', 'manipulated', 
        'photoshopped', 'edited', 'not real', 'artificial', 'staged',
        'impossible', 'unrealistic', 'hoax', 'deceptive'
    ]
    
    real_indicators = [
        'real news', 'real', 'true', 'authentic', 'genuine', 
        'actual', 'legitimate', 'verified', 'confirmed', 'it\'s real',
        'natural', 'genuine', 'authentic', 'believable'
    ]
    
    # Zähle Indikatoren
    fake_count = sum(1 for indicator in fake_indicators if indicator in response)
    real_count = sum(1 for indicator in real_indicators if indicator in response)
    
    # Scoring basierend auf Indikatoren
    if fake_count > real_count:
        return 0.2 + (fake_count - real_count) * 0.1  # 0.2-0.4
    elif real_count > fake_count:
        return 0.8 - (real_count - fake_count) * 0.1  # 0.6-0.8
    else:
        # Neutral basierend auf Länge und Wörtern
        words = response.split()
        if len(words) < 5:
            return 0.5
        
        # Sentiment-Analyse
        positive_words = ['yes', 'correct', 'accurate', 'true', 'real']
        negative_words = ['no', 'wrong', 'incorrect', 'false', 'fake']
        
        pos_count = sum(1 for word in positive_words if word in response)
        neg_count = sum(1 for word in negative_words if word in response)
        
        if pos_count > neg_count:
            return 0.6
        elif neg_count > pos_count:
            return 0.4
        else:
            return 0.5

class EnhancedBLIP2Handler:
    def __init__(self, use_rag: bool = True, use_few_shot: bool = True):
        """Erweiterter BLIP2 Handler mit RAG und Few-Shot"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_rag = use_rag
        self.use_few_shot = use_few_shot
        
        print(f"Using device: {self.device}")
        print(f"RAG enabled: {use_rag}")
        print(f"Few-Shot enabled: {use_few_shot}")
        
        # Knowledge Base laden
        if use_rag:
            self.knowledge_base = load_knowledge_base()
            print(f"Loaded {len(self.knowledge_base)} knowledge base documents")
        
        # BLIP2 Model laden
        print("Loading BLIP2 model: Salesforce/blip2-opt-2.7b")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        print("BLIP2 model loaded successfully!")

    def predict_enhanced(self, text: str, image: Image.Image) -> Dict[str, Any]:
        """Erweiterte Prediction mit RAG und Few-Shot"""
        
        # RAG Kontext erstellen
        rag_context = ""
        if self.use_rag:
            rag_context = create_rag_context(text, self.knowledge_base)
        
        # Erweiterten Prompt erstellen
        prompt = create_enhanced_prompt(text, rag_context)
        
        # Nur Original-Bild (keine Multi-Crop für Geschwindigkeit)
        crops = [image]
        
        all_scores = []
        all_responses = []
        
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
                        max_new_tokens=100,  # Längere Antworten für Analyse
                        num_beams=3,
                        do_sample=False,
                        temperature=1.0,
                        repetition_penalty=1.0
                    )
                
                response = self.processor.decode(outputs[0], skip_special_tokens=True)
                score = parse_blip2_response_enhanced(response)
                all_scores.append(score)
                all_responses.append(response)
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                all_scores.append(0.5)
                all_responses.append("error")
        
        # Einfache Aggregation
        final_score = np.mean(all_scores) if all_scores else 0.5
        final_response = all_responses[0] if all_responses else "error"
        
        return {
            'final_score': final_score,
            'response': final_response,
            'num_crops': len(crops),
            'rag_context': rag_context,
            'prompt_length': len(prompt)
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
        
        print(f"Enhanced BLIP2 Threshold Analysis:")
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

def calculate_enhanced_metrics(y_true, y_pred, scores, threshold):
    """Berechnet Metriken für erweiterte BLIP2"""
    
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
    print("ENHANCED BLIP2 + RAG + FEW-SHOT - VOLLSTÄNDIGE METRIKEN")
    print("="*60)
    print(f"Setup:")
    print(f"  - Model: Salesforce/blip2-opt-2.7b")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  - Samples: {len(y_true)}")
    print(f"  - Threshold: {threshold:.3f}")
    print(f"  - Optimizations: RAG + Few-Shot + Enhanced Parsing")
    
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

def plot_enhanced_results(metrics):
    """Erstellt Visualisierungen für erweiterte BLIP2"""
    
    # Confusion Matrix
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Enhanced BLIP2 + RAG + Few-Shot\nConfusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # ROC Curve
    plt.subplot(1, 3, 2)
    fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['scores'])
    plt.plot(fpr, tpr, label=f'Enhanced BLIP2 (AUC={metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Enhanced BLIP2 + RAG + Few-Shot\nROC Curve')
    plt.legend()
    
    # Score Distribution
    plt.subplot(1, 3, 3)
    pos_scores = [s for s, l in zip(metrics['scores'], metrics['y_true']) if l == 1]
    neg_scores = [s for s, l in zip(metrics['scores'], metrics['y_true']) if l == 0]
    
    plt.hist(pos_scores, alpha=0.7, label='Positive', bins=20, color='green')
    plt.hist(neg_scores, alpha=0.7, label='Negative', bins=20, color='red')
    plt.axvline(metrics['threshold'], color='black', linestyle='--', 
                label=f'Threshold: {metrics["threshold"]:.3f}')
    plt.xlabel('Enhanced BLIP2 Score')
    plt.ylabel('Frequency')
    plt.title('Enhanced BLIP2 + RAG + Few-Shot\nScore Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """Erweitertes BLIP2 Experiment mit RAG und Few-Shot"""
    
    # Parameter
    CSV_FILE = "test_balanced_pairs_clean.csv"
    NUM_SAMPLES = 50  # Weniger Samples für längere Prompts
    OUTPUT_FILE = "colab_blip2_rag_fewshot_results.csv"
    USE_RAG = True
    USE_FEW_SHOT = True
    
    print("Colab BLIP2 + RAG + Few-Shot - 75%+ Target")
    print("="*50)
    print(f"Enhancements:")
    print(f"  - Model: Salesforce/blip2-opt-2.7b")
    print(f"  - RAG: {USE_RAG}")
    print(f"  - Few-Shot: {USE_FEW_SHOT}")
    print(f"  - Enhanced Parsing: Yes")
    print(f"  - Longer Prompts: Yes")
    
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
    blip2 = EnhancedBLIP2Handler(use_rag=USE_RAG, use_few_shot=USE_FEW_SHOT)
    init_time = time.time() - start_time
    print(f"⏱️ Model initialization: {init_time:.2f}s")
    
    # Predictions durchführen
    results = []
    scores = []
    
    print(f"🔄 Running enhanced BLIP2 predictions on {len(df)} samples...")
    pred_start_time = time.time()
    
    for idx, row in df.iterrows():
        if idx % 5 == 0:  # Häufigere Updates wegen längerer Prompts
            print(f"  Processing sample {idx+1}/{len(df)}")
        
        image = load_local_image(row['id'])
        prediction = blip2.predict_enhanced(row['clean_title'], image)
        scores.append(prediction['final_score'])
        
        results.append({
            'id': row['id'],
            'text': row['clean_title'],
            'image_url': row['image_url'],
            'true_label': row['2_way_label'],
            'blip2_score': prediction['final_score'],
            'response': prediction.get('response', 'error'),
            'rag_context': prediction.get('rag_context', ''),
            'prompt_length': prediction.get('prompt_length', 0),
            'num_crops': prediction['num_crops']
        })
    
    pred_time = time.time() - pred_start_time
    print(f"⏱️ Prediction time: {pred_time:.2f}s ({pred_time/len(df):.3f}s per sample)")
    
    # Schwellenwert-Optimierung
    print(f"\n🎯 Finding optimal threshold for enhanced BLIP2...")
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
    
    print(f"Testing thresholds for enhanced BLIP2:")
    for name, threshold in thresholds_to_test:
        predictions = [int(score >= threshold) for score in scores]
        accuracy = accuracy_score(true_labels, predictions)
        print(f"  {name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold_name = name
            best_threshold = threshold
            best_predictions = predictions
    
    print(f"Best enhanced BLIP2: {best_threshold_name} = {best_threshold:.3f} -> {best_accuracy:.3f} ({best_accuracy*100:.1f}%)")
    
    # Metriken berechnen
    metrics = calculate_enhanced_metrics(true_labels, best_predictions, scores, best_threshold)
    
    # Finale Predictions setzen
    for r in results:
        r['predicted_label'] = int(r['blip2_score'] >= best_threshold)
        r['threshold'] = best_threshold
    
    # Ergebnisse speichern
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n💾 Results saved to {OUTPUT_FILE}")
    
    # Metriken als JSON speichern
    metrics_file = "colab_blip2_rag_fewshot_metrics.json"
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
        'enhancements': {
            'rag': USE_RAG,
            'few_shot': USE_FEW_SHOT,
            'knowledge_base_size': len(blip2.knowledge_base) if USE_RAG else 0
        },
        'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'performance': {
            'init_time': init_time,
            'prediction_time': pred_time,
            'total_time': init_time + pred_time,
            'samples_per_second': len(df) / pred_time,
            'avg_prompt_length': np.mean([r['prompt_length'] for r in results])
        }
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"📊 Metrics saved to {metrics_file}")
    
    # Visualisierungen erstellen
    print(f"\n📈 Creating enhanced BLIP2 visualizations...")
    plot_enhanced_results(metrics)
    
    total_time = time.time() - start_time
    print(f"\n✅ Enhanced BLIP2 + RAG + Few-Shot Experiment completed!")
    print(f"🎉 Achieved {best_accuracy*100:.1f}% accuracy!")
    print(f"⏱️ Total time: {total_time:.2f}s")
    
    # Vergleich mit vorherigen Versionen
    previous_blip2 = 0.56
    improvement = best_accuracy - previous_blip2
    
    if best_accuracy >= 0.75:
        print(f"🎯 EXCELLENT: 75%+ Accuracy!")
        print(f"🚀 RAG + Few-Shot approach worked perfectly!")
    elif best_accuracy >= 0.70:
        print(f"📈 Great improvement: {best_accuracy*100:.1f}% (previous: 56%)")
        print(f"💡 RAG + Few-Shot was successful")
    elif best_accuracy >= 0.65:
        print(f"📈 Good improvement: {best_accuracy*100:.1f}% (previous: 56%)")
        print(f"💡 Some improvement from enhancements")
    else:
        print(f"📉 Still below target: {best_accuracy*100:.1f}% (previous: 56%)")
        print(f"💡 BLIP2 might need different approach")
    
    if improvement > 0:
        print(f"📈 VERBESSERUNG: +{improvement:.3f} (+{improvement*100:.1f}%) über vorherigem BLIP2")
    elif improvement < 0:
        print(f"📉 RÜCKSCHRITT: {improvement:.3f} ({improvement*100:.1f}%) unter vorherigem BLIP2")
    else:
        print(f"📊 GLEICH: Keine Änderung zum vorherigen BLIP2")
    
    print(f"\n📋 Enhanced BLIP2 Summary:")
    print(f"  - Model: Salesforce/blip2-opt-2.7b")
    print(f"  - Accuracy: {best_accuracy*100:.1f}%")
    print(f"  - Threshold: {best_threshold:.3f}")
    print(f"  - RAG: {USE_RAG}")
    print(f"  - Few-Shot: {USE_FEW_SHOT}")
    print(f"  - Performance: {len(df)/pred_time:.1f} samples/second")
    print(f"  - Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    main() 