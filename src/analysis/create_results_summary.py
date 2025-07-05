#!/usr/bin/env python3
"""
Erstelle eine Zusammenfassung der wichtigsten Ergebnisse
Vor der Aufräumung des results-Ordners
"""

import pandas as pd
import json
from pathlib import Path
import glob

def create_results_summary(results_dir="results"):
    """Erstelle eine Zusammenfassung der wichtigsten Ergebnisse"""
    
    results_path = Path(results_dir)
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_experiments': 0,
        'best_models': {},
        'key_findings': [],
        'important_files': [],
        'metrics_summary': {}
    }
    
    print("📋 Erstelle Zusammenfassung der wichtigsten Ergebnisse...")
    print("=" * 60)
    
    # 1. Hauptmetriken-Datei analysieren
    main_metrics = results_path / 'metrics_comparison.csv'
    if main_metrics.exists():
        print("📊 Analysiere Hauptmetriken...")
        df = pd.read_csv(main_metrics)
        
        # Beste Modelle nach verschiedenen Metriken
        best_accuracy = df.loc[df['Accuracy'].idxmax()]
        best_f1 = df.loc[df['F1-Score'].idxmax()]
        best_auc = df.loc[df['ROC_AUC'].idxmax()]
        
        summary['best_models'] = {
            'best_accuracy': {
                'model': f"{best_accuracy['Model']} {best_accuracy['Version']}",
                'mode': best_accuracy['Mode'],
                'accuracy': best_accuracy['Accuracy'],
                'f1': best_accuracy['F1-Score']
            },
            'best_f1': {
                'model': f"{best_f1['Model']} {best_f1['Version']}",
                'mode': best_f1['Mode'],
                'accuracy': best_f1['Accuracy'],
                'f1': best_f1['F1-Score']
            },
            'best_auc': {
                'model': f"{best_auc['Model']} {best_auc['Version']}",
                'mode': best_auc['Mode'],
                'auc': best_auc['ROC_AUC']
            }
        }
        
        print(f"🏆 Beste Accuracy: {summary['best_models']['best_accuracy']['model']} ({summary['best_models']['best_accuracy']['accuracy']:.3f})")
        print(f"🏆 Beste F1-Score: {summary['best_models']['best_f1']['model']} ({summary['best_models']['best_f1']['f1']:.3f})")
        print(f"🏆 Beste AUC: {summary['best_models']['best_auc']['model']} ({summary['best_models']['best_auc']['auc']:.3f})")
        print()
        
        # RAG Verbesserungen
        rag_improvements = df[df['RAG_Improvement'] > 0]
        if not rag_improvements.empty:
            print("📈 RAG Verbesserungen:")
            for _, row in rag_improvements.iterrows():
                print(f"   {row['Model']} {row['Version']}: +{row['RAG_Improvement']:.3f}")
            print()
    
    # 2. Wichtige Experimente identifizieren
    print("🔍 Identifiziere wichtige Experimente...")
    
    important_patterns = [
        'final_*',
        'best_*', 
        'optimal_*',
        '*_final_*',
        '*_best_*',
        '*_optimal_*'
    ]
    
    important_experiments = []
    for pattern in important_patterns:
        for path in results_path.rglob(pattern):
            if path.is_dir():
                important_experiments.append(str(path.relative_to(results_path)))
    
    summary['important_experiments'] = sorted(list(set(important_experiments)))
    
    print(f"⭐ {len(summary['important_experiments'])} wichtige Experimente gefunden:")
    for exp in summary['important_experiments'][:10]:  # Zeige nur die ersten 10
        print(f"   - {exp}")
    if len(summary['important_experiments']) > 10:
        print(f"   ... und {len(summary['important_experiments']) - 10} weitere")
    print()
    
    # 3. Modell-spezifische Bestleistungen
    print("📊 Modell-spezifische Bestleistungen:")
    
    model_results = {}
    for csv_file in results_path.rglob("*/figures/*/metrics_summary.csv"):
        try:
            df = pd.read_csv(csv_file)
            if 'Accuracy' in df.columns or 'Value' in df.columns:
                # Extrahiere Modell-Name aus Pfad
                path_parts = csv_file.parts
                model_name = path_parts[-2] if len(path_parts) > 2 else "Unknown"
                
                if 'Value' in df.columns:
                    # Suche nach Accuracy in Value-Spalte
                    acc_row = df[df['Metric'] == 'Accuracy']
                    if not acc_row.empty:
                        accuracy = float(acc_row.iloc[0]['Value'])
                        if model_name not in model_results or accuracy > model_results[model_name]:
                            model_results[model_name] = accuracy
                elif 'Accuracy' in df.columns:
                    accuracy = float(df['Accuracy'].iloc[0])
                    if model_name not in model_results or accuracy > model_results[model_name]:
                        model_results[model_name] = accuracy
        except Exception as e:
            continue
    
    # Zeige Top-Modelle
    top_models = sorted(model_results.items(), key=lambda x: x[1], reverse=True)[:5]
    summary['top_models'] = dict(top_models)
    
    for model, acc in top_models:
        print(f"   {model}: {acc:.3f}")
    print()
    
    # 4. Ensemble-Ergebnisse
    print("🤝 Ensemble-Ergebnisse:")
    ensemble_files = list(results_path.rglob("ensemble/*/ensemble_metrics.csv"))
    ensemble_files.extend(list(results_path.rglob("ensemble/*/ensemble_predictions.csv")))
    
    ensemble_results = {}
    for file in ensemble_files:
        try:
            df = pd.read_csv(file)
            if 'accuracy' in df.columns:
                accuracy = float(df['accuracy'].iloc[0])
                ensemble_name = file.parent.name
                ensemble_results[ensemble_name] = accuracy
        except Exception as e:
            continue
    
    summary['ensemble_results'] = ensemble_results
    
    if ensemble_results:
        for name, acc in sorted(ensemble_results.items(), key=lambda x: x[1], reverse=True):
            print(f"   {name}: {acc:.3f}")
    else:
        print("   Keine Ensemble-Ergebnisse gefunden")
    print()
    
    # 5. Wichtige Dateien für Backup
    print("💾 Wichtige Dateien für Backup:")
    important_files = []
    
    # Hauptmetriken
    important_files.append(str(main_metrics))
    
    # Beste Experimente
    for exp in summary['important_experiments']:
        exp_path = results_path / exp
        if exp_path.exists():
            # Suche nach wichtigen Dateien im Experiment
            for pattern in ['all_model_outputs.csv', 'metrics_summary.csv', 'ensemble_*.csv']:
                for file in exp_path.rglob(pattern):
                    important_files.append(str(file))
    
    summary['important_files'] = sorted(list(set(important_files)))
    
    print(f"   {len(summary['important_files'])} wichtige Dateien identifiziert")
    print()
    
    # 6. Speichere Zusammenfassung
    summary_file = results_path / 'results_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📋 Zusammenfassung gespeichert: {summary_file}")
    
    # 7. Erstelle Backup-Liste
    backup_list_file = results_path / 'backup_file_list.txt'
    with open(backup_list_file, 'w') as f:
        f.write("# Wichtige Dateien für Backup\n")
        f.write(f"# Erstellt: {pd.Timestamp.now()}\n\n")
        for file in summary['important_files']:
            f.write(f"{file}\n")
    
    print(f"📝 Backup-Liste erstellt: {backup_list_file}")
    
    return summary

def main():
    """Hauptfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Erstelle Zusammenfassung der Ergebnisse')
    parser.add_argument('--results-dir', default='results', help='Pfad zum results-Ordner')
    
    args = parser.parse_args()
    
    summary = create_results_summary(args.results_dir)
    
    print("✅ Zusammenfassung erfolgreich erstellt!")
    print("💡 Diese Dateien sind wichtig für das Backup:")
    print("   - results_summary.json")
    print("   - backup_file_list.txt")
    print("   - metrics_comparison.csv")

if __name__ == "__main__":
    main() 