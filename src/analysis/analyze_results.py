#!/usr/bin/env python3
"""
Results-Ordner Analyse
Detaillierte Analyse vor der Aufräumung
"""

import os
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_results_structure(results_dir="results"):
    """Detaillierte Analyse der Results-Struktur"""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results-Ordner {results_dir} nicht gefunden!")
        return
    
    print("🔍 Analysiere Results-Ordner...")
    print("=" * 50)
    
    # Sammle Statistiken
    stats = {
        'total_files': 0,
        'total_dirs': 0,
        'size_mb': 0,
        'file_types': defaultdict(int),
        'model_dirs': [],
        'important_files': [],
        'large_files': [],
        'duplicate_names': defaultdict(list)
    }
    
    # Durchlaufe alle Dateien
    for root, dirs, files in os.walk(results_path):
        stats['total_dirs'] += len(dirs)
        stats['total_files'] += len(files)
        
        for file in files:
            file_path = Path(root) / file
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                stats['size_mb'] += size_mb
                
                # Dateityp
                ext = file_path.suffix.lower()
                stats['file_types'][ext] += 1
                
                # Große Dateien (>1MB)
                if size_mb > 1:
                    stats['large_files'].append({
                        'path': str(file_path),
                        'size_mb': size_mb
                    })
                
                # Wichtige Dateien
                if any(keyword in file.lower() for keyword in ['metrics', 'comparison', 'final', 'best', 'summary']):
                    stats['important_files'].append(str(file_path))
                
                # Duplikate nach Namen
                stats['duplicate_names'][file].append(str(file_path))
    
    # Modell-Ordner identifizieren
    for item in results_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            stats['model_dirs'].append(item.name)
    
    # Ausgabe
    print(f"📊 Gesamtstatistiken:")
    print(f"   Dateien: {stats['total_files']}")
    print(f"   Ordner: {stats['total_dirs']}")
    print(f"   Größe: {stats['size_mb']:.2f} MB")
    print()
    
    print(f"📁 Modell-Ordner ({len(stats['model_dirs'])}):")
    for dir_name in sorted(stats['model_dirs']):
        print(f"   - {dir_name}")
    print()
    
    print(f"📄 Dateitypen:")
    for ext, count in sorted(stats['file_types'].items(), key=lambda x: x[1], reverse=True):
        print(f"   {ext}: {count}")
    print()
    
    print(f"⭐ Wichtige Dateien ({len(stats['important_files'])}):")
    for file_path in sorted(stats['important_files']):
        rel_path = Path(file_path).relative_to(results_path)
        print(f"   - {rel_path}")
    print()
    
    print(f"💾 Große Dateien (>1MB):")
    for file_info in sorted(stats['large_files'], key=lambda x: x['size_mb'], reverse=True):
        rel_path = Path(file_info['path']).relative_to(results_path)
        print(f"   - {rel_path} ({file_info['size_mb']:.2f} MB)")
    print()
    
    # Duplikate finden
    duplicates = {name: paths for name, paths in stats['duplicate_names'].items() if len(paths) > 1}
    if duplicates:
        print(f"⚠️  Potentielle Duplikate:")
        for name, paths in list(duplicates.items())[:10]:  # Zeige nur die ersten 10
            print(f"   {name}:")
            for path in paths:
                rel_path = Path(path).relative_to(results_path)
                print(f"     - {rel_path}")
        print()
    
    # Speichere detaillierte Analyse
    analysis_file = results_path / 'detailed_analysis.json'
    with open(analysis_file, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"📋 Detaillierte Analyse gespeichert: {analysis_file}")
    
    return stats

def analyze_metrics_files(results_dir="results"):
    """Analysiere alle Metriken-Dateien"""
    
    results_path = Path(results_dir)
    metrics_files = []
    
    # Finde alle CSV-Dateien
    for csv_file in results_path.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            metrics_files.append({
                'path': str(csv_file),
                'rows': len(df),
                'columns': list(df.columns),
                'size_mb': csv_file.stat().st_size / (1024 * 1024)
            })
        except Exception as e:
            print(f"⚠️  Konnte {csv_file} nicht lesen: {e}")
    
    print(f"📈 Metriken-Dateien gefunden: {len(metrics_files)}")
    print()
    
    for file_info in sorted(metrics_files, key=lambda x: x['size_mb'], reverse=True):
        rel_path = Path(file_info['path']).relative_to(results_path)
        print(f"📊 {rel_path}")
        print(f"   Größe: {file_info['size_mb']:.2f} MB")
        print(f"   Zeilen: {file_info['rows']}")
        print(f"   Spalten: {file_info['columns']}")
        print()
    
    return metrics_files

def generate_cleanup_recommendations(stats):
    """Generiere Aufräum-Empfehlungen"""
    
    print("🎯 Aufräum-Empfehlungen:")
    print("=" * 50)
    
    # Empfehlungen basierend auf Analyse
    recommendations = []
    
    # Test-Ordner
    test_dirs = [d for d in stats['model_dirs'] if 'test' in d.lower()]
    if test_dirs:
        recommendations.append(f"📁 {len(test_dirs)} Test-Ordner ins Archive verschieben: {', '.join(test_dirs)}")
    
    # Große Dateien
    large_files = [f for f in stats['large_files'] if f['size_mb'] > 5]
    if large_files:
        recommendations.append(f"💾 {len(large_files)} sehr große Dateien (>5MB) prüfen")
    
    # Duplikate
    duplicates = {name: paths for name, paths in stats['duplicate_names'].items() if len(paths) > 1}
    if duplicates:
        recommendations.append(f"⚠️  {len(duplicates)} Dateien mit identischen Namen - Duplikate prüfen")
    
    # Wichtige Dateien bewahren
    if stats['important_files']:
        recommendations.append(f"⭐ {len(stats['important_files'])} wichtige Dateien im final_results Ordner bewahren")
    
    # Neue Struktur
    recommendations.append("📂 Neue Struktur erstellen: final_results/, experiments/, archive/, analysis/")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    print()
    print("✅ Empfehlung: Verwende das cleanup_results.py Skript mit --dry-run zuerst!")

def main():
    """Hauptfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Results-Ordner analysieren')
    parser.add_argument('--results-dir', default='results', help='Pfad zum results-Ordner')
    parser.add_argument('--metrics-only', action='store_true', help='Nur Metriken-Dateien analysieren')
    
    args = parser.parse_args()
    
    if args.metrics_only:
        analyze_metrics_files(args.results_dir)
    else:
        stats = analyze_results_structure(args.results_dir)
        analyze_metrics_files(args.results_dir)
        generate_cleanup_recommendations(stats)
    
    print("🎉 Analyse abgeschlossen!")

if __name__ == "__main__":
    main() 