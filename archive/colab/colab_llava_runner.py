#!/usr/bin/env python3
"""
LLaVA Fact-Checking Experiments für Colab
Einfacher Runner für LLaVA-Experimente
"""

import os
import shutil
import json
import subprocess
import datetime
from google.colab import drive
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🚀 LLaVA Fact-Checking Experiments")
    print("=" * 50)
    
    # 1. Google Drive mounten
    print("\n1️⃣ Google Drive mounten...")
    drive.mount('/content/drive')
    print('✅ Google Drive gemountet')
    
    # 2. Arbeitsverzeichnis einrichten
    print("\n2️⃣ Arbeitsverzeichnis einrichten...")
    DRIVE_PATH = '/content/drive/MyDrive/mllm_colab'
    WORK_DIR = '/content/mllm'
    
    os.makedirs(WORK_DIR, exist_ok=True)
    print(f'📁 Arbeitsverzeichnis: {WORK_DIR}')
    
    # Daten von Drive kopieren
    if os.path.exists(DRIVE_PATH):
        shutil.copytree(DRIVE_PATH, WORK_DIR, dirs_exist_ok=True)
        print('✅ Daten von Google Drive kopiert')
    else:
        print('❌ mllm_colab Ordner nicht in Google Drive gefunden')
        print('Bitte lade mllm_colab.tar.gz in Google Drive hoch und entpacke es')
        return
    
    # 3. Dependencies installieren
    print("\n3️⃣ Dependencies installieren...")
    deps = [
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'accelerate',
        'bitsandbytes',
        'pandas',
        'numpy',
        'tqdm',
        'Pillow',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ]
    
    for dep in deps:
        print(f'Installing {dep}...')
        subprocess.run(f'pip install {dep}', shell=True, check=True)
    
    print('✅ Alle Dependencies installiert')
    
    # 4. GPU Check
    print("\n4️⃣ GPU Status prüfen...")
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
        print(f'   Speicher: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
        device = 'cuda'
    else:
        print('⚠️  Keine GPU verfügbar')
        device = 'cpu'
    
    print(f'💻 Device: {device}')
    
    # 5. LLaVA Modell laden
    print("\n5️⃣ LLaVA Modell laden...")
    MODEL_NAME = 'llava-hf/llava-1.5-7b-hf'
    
    processor = LlavaProcessor.from_pretrained(MODEL_NAME)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map='auto',
        load_in_8bit=True
    )
    
    print('✅ LLaVA Modell geladen')
    
    # 6. Daten laden
    print("\n6️⃣ Testdaten laden...")
    data_dir = f'{WORK_DIR}/data'
    images_dir = f'{data_dir}/downloaded_fakeddit_images'
    test_file = f'{data_dir}/processed/test_balanced_pairs.csv'
    
    # Verfügbare Dateien prüfen
    if os.path.exists(images_dir):
        images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        print(f'✅ {len(images)} Bilder gefunden')
    else:
        print('❌ Bilderordner nicht gefunden')
        return
    
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        print(f'✅ {len(df)} Testbeispiele geladen')
        print('Spalten:', df.columns.tolist())
        print('\nErste 3 Zeilen:')
        print(df[['id', 'title', '2_way_label']].head(3))
    else:
        print('❌ Testdaten nicht gefunden')
        return
    
    # 7. LLaVA Vorhersage-Funktionen
    print("\n7️⃣ Vorhersage-Funktionen definieren...")
    
    def predict_fake_real(image_path, prompt="Ist dieses Bild echt oder gefälscht? Antworte nur mit 'echt' oder 'gefälscht'."):
        try:
            if not os.path.exists(image_path):
                return None, f'Bild nicht gefunden: {image_path}'
            
            image = Image.open(image_path)
            inputs = processor(prompt, image, return_tensors='pt')
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=50,
                    do_sample=False,
                    temperature=0.0
                )
            
            response = processor.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, '').strip()
            
            return response, None
            
        except Exception as e:
            return None, str(e)
    
    def parse_response(response):
        if not response:
            return None
        
        response_lower = response.lower().strip()
        
        if 'echt' in response_lower or 'real' in response_lower:
            return 1  # Echt
        elif 'gefälscht' in response_lower or 'fake' in response_lower:
            return 0  # Gefälscht
        else:
            return None  # Unklar
    
    print('✅ Vorhersage-Funktionen definiert')
    
    # 8. Experiment ausführen
    print("\n8️⃣ LLaVA Experiment ausführen...")
    
    # Experiment-Parameter
    NUM_SAMPLES = 20  # Reduziert für schnelleren Test
    
    # Daten vorbereiten
    test_df = df.head(NUM_SAMPLES).copy()
    
    results = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Verarbeite Bilder'):
        # Bildpfad erstellen
        image_id = row['id']
        image_path = f'{images_dir}/{image_id}.jpg'
        
        # Vorhersage
        response, error = predict_fake_real(image_path)
        prediction = parse_response(response)
        
        # Ergebnis speichern
        result = {
            'image_id': image_id,
            'title': row['title'],
            'true_label': row['2_way_label'],
            'prediction': prediction,
            'response': response,
            'error': error
        }
        
        results.append(result)
        
        # Fortschritt anzeigen
        if (idx + 1) % 5 == 0:
            print(f'  Verarbeitet: {idx + 1}/{len(test_df)}')
    
    print(f'✅ Experiment abgeschlossen: {len(results)} Ergebnisse')
    
    # 9. Ergebnisse analysieren
    print("\n9️⃣ Ergebnisse analysieren...")
    
    results_df = pd.DataFrame(results)
    
    # Übersicht
    print(f'Gesamt: {len(results_df)}')
    print(f'Erfolgreich: {len(results_df[results_df["error"].isna()])}')
    print(f'Klare Vorhersagen: {len(results_df[results_df["prediction"].notna()])}')
    
    # Metriken
    valid_results = results_df[results_df['prediction'].notna()]
    
    if len(valid_results) > 0:
        accuracy = accuracy_score(valid_results['true_label'], valid_results['prediction'])
        print(f'\n📈 Accuracy: {accuracy:.3f}')
        
        # Confusion Matrix
        cm = confusion_matrix(valid_results['true_label'], valid_results['prediction'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Gefälscht', 'Echt'], 
                   yticklabels=['Gefälscht', 'Echt'])
        plt.title('LLaVA Fact-Checking - Confusion Matrix')
        plt.ylabel('Wahre Labels')
        plt.xlabel('Vorhersagen')
        plt.show()
        
        # Detaillierte Ergebnisse
        print('\n🔍 Detaillierte Ergebnisse:')
        print(results_df[['image_id', 'true_label', 'prediction', 'response']].head(10))
    else:
        print('❌ Keine gültigen Vorhersagen für Metriken')
    
    # 10. Ergebnisse speichern
    print("\n🔟 Ergebnisse speichern...")
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f'{WORK_DIR}/llava_results_{timestamp}.csv'
    
    results_df.to_csv(results_file, index=False)
    print(f'✅ Ergebnisse gespeichert: {results_file}')
    
    # Zurück auf Drive kopieren
    drive_results = f'{DRIVE_PATH}/llava_results_{timestamp}.csv'
    shutil.copy(results_file, drive_results)
    print(f'✅ Ergebnisse auf Google Drive kopiert: {drive_results}')
    
    print("\n🎉 Experiment erfolgreich abgeschlossen!")
    print("\nNächste Schritte:")
    print("1. Ergebnisse in Google Drive überprüfen")
    print("2. NUM_SAMPLES erhöhen für mehr Testbeispiele")
    print("3. Prompt in predict_fake_real() anpassen")
    print("4. Mit anderen Modellen vergleichen")

if __name__ == "__main__":
    main() 