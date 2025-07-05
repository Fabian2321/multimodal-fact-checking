#!/usr/bin/env python3
"""
Sicheres Backup aller wichtigen Dateien
Vor der Aufräumung des results-Ordners
"""

import shutil
import json
from pathlib import Path
from datetime import datetime

def create_safe_backup(results_dir="results"):
    """Erstelle ein sicheres Backup aller wichtigen Dateien"""
    
    results_path = Path(results_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = Path(f"results_safe_backup_{timestamp}")
    
    print(f"🛡️  Erstelle sicheres Backup: {backup_dir}")
    print("=" * 50)
    
    # Erstelle Backup-Ordner
    backup_dir.mkdir(exist_ok=True)
    
    # 1. Kopiere Zusammenfassungsdateien
    print("📋 Kopiere Zusammenfassungsdateien...")
    summary_files = [
        'results_summary.json',
        'backup_file_list.txt', 
        'metrics_comparison.csv',
        'detailed_analysis.json'
    ]
    
    for file_name in summary_files:
        source = results_path / file_name
        if source.exists():
            shutil.copy2(source, backup_dir / file_name)
            print(f"   ✅ {file_name}")
        else:
            print(f"   ⚠️  {file_name} nicht gefunden")
    
    # 2. Kopiere wichtige Experimente
    print("\n📁 Kopiere wichtige Experimente...")
    
    # Lade Zusammenfassung
    summary_file = results_path / 'results_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        important_experiments = summary.get('important_experiments', [])
        
        for exp_path in important_experiments:
            source = results_path / exp_path
            if source.exists():
                target = backup_dir / 'important_experiments' / exp_path
                target.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    shutil.copytree(source, target, dirs_exist_ok=True)
                    print(f"   ✅ {exp_path}")
                except Exception as e:
                    print(f"   ❌ {exp_path}: {e}")
            else:
                print(f"   ⚠️  {exp_path} nicht gefunden")
    
    # 3. Kopiere Ensemble-Ergebnisse
    print("\n🤝 Kopiere Ensemble-Ergebnisse...")
    ensemble_dir = results_path / 'ensemble'
    if ensemble_dir.exists():
        target = backup_dir / 'ensemble'
        shutil.copytree(ensemble_dir, target, dirs_exist_ok=True)
        print("   ✅ Ensemble-Ergebnisse kopiert")
    
    # 4. Kopiere Beste Modelle nach Modell-Typ
    print("\n🏆 Kopiere beste Modelle...")
    
    model_dirs = ['clip', 'blip', 'blip2', 'llava', 'bert']
    for model in model_dirs:
        source = results_path / model
        if source.exists():
            target = backup_dir / 'best_models' / model
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # Kopiere nur wichtige Dateien
            try:
                shutil.copytree(source, target, dirs_exist_ok=True)
                print(f"   ✅ {model}")
            except Exception as e:
                print(f"   ❌ {model}: {e}")
    
    # 5. Erstelle Backup-Metadaten
    print("\n📝 Erstelle Backup-Metadaten...")
    
    backup_metadata = {
        'backup_timestamp': timestamp,
        'original_results_dir': str(results_path.absolute()),
        'backup_contents': {
            'summary_files': len([f for f in summary_files if (results_path / f).exists()]),
            'important_experiments': len(important_experiments) if 'important_experiments' in locals() else 0,
            'ensemble_results': ensemble_dir.exists(),
            'model_dirs': len([d for d in model_dirs if (results_path / d).exists()])
        },
        'backup_size_mb': 0
    }
    
    # Berechne Backup-Größe
    total_size = 0
    for file_path in backup_dir.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    backup_metadata['backup_size_mb'] = total_size / (1024 * 1024)
    
    # Speichere Metadaten
    with open(backup_dir / 'backup_metadata.json', 'w') as f:
        json.dump(backup_metadata, f, indent=2)
    
    print(f"   ✅ Backup-Metadaten erstellt")
    print(f"   📊 Backup-Größe: {backup_metadata['backup_size_mb']:.2f} MB")
    
    # 6. Erstelle README für Backup
    readme_content = f"""# Results Backup - {timestamp}

Dieses Backup wurde vor der Aufräumung des results-Ordners erstellt.

## Inhalt:
- Zusammenfassungsdateien (results_summary.json, metrics_comparison.csv, etc.)
- Wichtige Experimente (final_*, best_*, optimal_*)
- Ensemble-Ergebnisse
- Beste Modelle nach Modell-Typ

## Wiederherstellung:
Um Dateien wiederherzustellen, kopiere sie aus diesem Backup zurück in den results-Ordner.

## Metadaten:
- Backup-Größe: {backup_metadata['backup_size_mb']:.2f} MB
- Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Original-Ordner: {results_path.absolute()}

## Wichtige Dateien:
- results_summary.json: Zusammenfassung aller Ergebnisse
- backup_file_list.txt: Liste aller wichtigen Dateien
- metrics_comparison.csv: Hauptmetriken-Vergleich
- backup_metadata.json: Backup-Metadaten
"""
    
    with open(backup_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print("   ✅ README erstellt")
    
    print(f"\n✅ Sicheres Backup erfolgreich erstellt: {backup_dir}")
    print(f"📊 Größe: {backup_metadata['backup_size_mb']:.2f} MB")
    print(f"📁 Inhalt: {backup_metadata['backup_contents']}")
    
    return backup_dir

def main():
    """Hauptfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Erstelle sicheres Backup der Ergebnisse')
    parser.add_argument('--results-dir', default='results', help='Pfad zum results-Ordner')
    
    args = parser.parse_args()
    
    backup_dir = create_safe_backup(args.results_dir)
    
    print("\n🎉 Backup abgeschlossen!")
    print("💡 Du kannst jetzt sicher mit der Aufräumung fortfahren:")
    print("   python scripts/cleanup_results.py")

if __name__ == "__main__":
    main() 