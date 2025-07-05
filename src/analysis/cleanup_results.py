#!/usr/bin/env python3
"""
Results Ordner Aufräumskript
Sicherer Aufräumprozess für den results-Ordner
"""

import os
import shutil
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results_cleanup.log'),
        logging.StreamHandler()
    ]
)

class ResultsCleaner:
    def __init__(self, results_dir="results"):
        self.results_dir = Path(results_dir)
        self.backup_dir = Path(f"results_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.cleanup_log = []
        
    def create_backup(self):
        """Erstelle ein vollständiges Backup des results-Ordners"""
        logging.info(f"Erstelle Backup: {self.backup_dir}")
        if self.results_dir.exists():
            shutil.copytree(self.results_dir, self.backup_dir)
            logging.info("Backup erfolgreich erstellt")
        else:
            logging.error("Results-Ordner nicht gefunden!")
            return False
        return True
    
    def analyze_structure(self):
        """Analysiere die aktuelle Struktur des results-Ordners"""
        logging.info("Analysiere Results-Struktur...")
        
        structure = {
            'total_files': 0,
            'total_dirs': 0,
            'size_mb': 0,
            'model_dirs': [],
            'csv_files': [],
            'json_files': [],
            'log_files': []
        }
        
        for root, dirs, files in os.walk(self.results_dir):
            structure['total_dirs'] += len(dirs)
            structure['total_files'] += len(files)
            
            for file in files:
                file_path = Path(root) / file
                if file_path.exists():
                    structure['size_mb'] += file_path.stat().st_size / (1024 * 1024)
                
                if file.endswith('.csv'):
                    structure['csv_files'].append(str(file_path))
                elif file.endswith('.json'):
                    structure['json_files'].append(str(file_path))
                elif file.endswith('.log'):
                    structure['log_files'].append(str(file_path))
        
        # Finde Modell-spezifische Ordner
        for item in self.results_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                structure['model_dirs'].append(item.name)
        
        logging.info(f"Gefunden: {structure['total_files']} Dateien, {structure['total_dirs']} Ordner, {structure['size_mb']:.2f} MB")
        return structure
    
    def identify_important_files(self):
        """Identifiziere wichtige Dateien, die nicht gelöscht werden dürfen"""
        important_files = []
        
        # Suche nach wichtigen Metriken-Dateien
        for root, dirs, files in os.walk(self.results_dir):
            for file in files:
                if any(keyword in file.lower() for keyword in ['metrics', 'comparison', 'final', 'best', 'summary']):
                    important_files.append(str(Path(root) / file))
        
        logging.info(f"Identifiziert: {len(important_files)} wichtige Dateien")
        return important_files
    
    def create_new_structure(self):
        """Erstelle die neue, saubere Struktur"""
        new_structure = {
            'final_results': ['best_models', 'final_reports'],
            'experiments': ['clip', 'blip2', 'llava', 'bert', 'ensemble'],
            'archive': ['test_runs', 'deprecated', 'old_versions'],
            'analysis': ['figures', 'reports', 'comparisons']
        }
        
        for main_dir, sub_dirs in new_structure.items():
            main_path = self.results_dir / main_dir
            main_path.mkdir(exist_ok=True)
            
            for sub_dir in sub_dirs:
                (main_path / sub_dir).mkdir(exist_ok=True)
        
        logging.info("Neue Struktur erstellt")
    
    def move_files_safely(self, source, destination):
        """Sichere Datei-Verschiebung mit Logging"""
        try:
            if source.exists():
                shutil.move(str(source), str(destination))
                self.cleanup_log.append(f"Verschoben: {source} -> {destination}")
                logging.info(f"Verschoben: {source} -> {destination}")
                return True
        except Exception as e:
            logging.error(f"Fehler beim Verschieben {source}: {e}")
            return False
    
    def organize_by_model(self):
        """Organisiere Dateien nach Modell-Typen"""
        logging.info("Organisiere Dateien nach Modellen...")
        
        # Mapping von Ordnernamen zu neuen Kategorien
        model_mapping = {
            'clip': 'experiments/clip',
            'blip': 'experiments/blip2', 
            'blip2': 'experiments/blip2',
            'blip2_enhanced': 'experiments/blip2',
            'llava': 'experiments/llava',
            'bert': 'experiments/bert',
            'ensemble': 'experiments/ensemble'
        }
        
        # Verschiebe Modell-spezifische Ordner
        for item in self.results_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if item.name in model_mapping:
                    target_dir = self.results_dir / model_mapping[item.name]
                    if target_dir.exists():
                        # Wenn Zielordner existiert, verschiebe Inhalt
                        for sub_item in item.iterdir():
                            target_path = target_dir / f"{item.name}_{sub_item.name}"
                            self.move_files_safely(sub_item, target_path)
                    else:
                        self.move_files_safely(item, target_dir)
                elif 'test' in item.name.lower():
                    # Test-Ordner ins Archive
                    target_dir = self.results_dir / 'archive' / 'test_runs'
                    self.move_files_safely(item, target_dir / item.name)
                else:
                    # Unbekannte Ordner ins Archive
                    target_dir = self.results_dir / 'archive' / 'deprecated'
                    self.move_files_safely(item, target_dir / item.name)
    
    def preserve_important_files(self):
        """Bewahre wichtige Dateien im final_results Ordner"""
        logging.info("Bewahre wichtige Dateien...")
        
        # Verschiebe metrics_comparison.csv
        metrics_file = self.results_dir / 'metrics_comparison.csv'
        if metrics_file.exists():
            target = self.results_dir / 'final_results' / 'metrics_comparison.csv'
            self.move_files_safely(metrics_file, target)
        
        # Suche nach anderen wichtigen Dateien
        important_patterns = ['*metrics*.csv', '*comparison*.csv', '*final*.csv', '*summary*.csv']
        for pattern in important_patterns:
            for file_path in self.results_dir.rglob(pattern):
                if file_path.is_file():
                    target = self.results_dir / 'final_results' / file_path.name
                    self.move_files_safely(file_path, target)
    
    def cleanup_empty_dirs(self):
        """Entferne leere Ordner"""
        logging.info("Entferne leere Ordner...")
        
        for root, dirs, files in os.walk(self.results_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        logging.info(f"Entfernt leerer Ordner: {dir_path}")
                except Exception as e:
                    logging.warning(f"Konnte leeren Ordner nicht entfernen {dir_path}: {e}")
    
    def generate_cleanup_report(self):
        """Generiere einen Bericht über die Aufräumung"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'backup_location': str(self.backup_dir),
            'actions_performed': self.cleanup_log,
            'new_structure': self.analyze_structure()
        }
        
        report_file = self.results_dir / 'cleanup_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Aufräumbericht gespeichert: {report_file}")
    
    def run_cleanup(self, dry_run=False):
        """Führe die komplette Aufräumung durch"""
        logging.info("Starte Results-Aufräumung...")
        
        if dry_run:
            logging.info("DRY RUN MODE - Keine Änderungen werden vorgenommen")
        
        # 1. Backup erstellen
        if not dry_run:
            if not self.create_backup():
                return False
        
        # 2. Aktuelle Struktur analysieren
        old_structure = self.analyze_structure()
        
        # 3. Neue Struktur erstellen
        if not dry_run:
            self.create_new_structure()
        
        # 4. Wichtige Dateien identifizieren
        important_files = self.identify_important_files()
        
        # 5. Dateien organisieren
        if not dry_run:
            self.organize_by_model()
            self.preserve_important_files()
            self.cleanup_empty_dirs()
            self.generate_cleanup_report()
        
        # 6. Neue Struktur analysieren
        new_structure = self.analyze_structure()
        
        logging.info("Aufräumung abgeschlossen!")
        logging.info(f"Dateien vorher: {old_structure['total_files']}, nachher: {new_structure['total_files']}")
        logging.info(f"Größe vorher: {old_structure['size_mb']:.2f} MB, nachher: {new_structure['size_mb']:.2f} MB")
        
        return True

def main():
    """Hauptfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Results-Ordner aufräumen')
    parser.add_argument('--dry-run', action='store_true', help='Nur simulieren, keine Änderungen vornehmen')
    parser.add_argument('--results-dir', default='results', help='Pfad zum results-Ordner')
    
    args = parser.parse_args()
    
    cleaner = ResultsCleaner(args.results_dir)
    success = cleaner.run_cleanup(dry_run=args.dry_run)
    
    if success:
        print("✅ Aufräumung erfolgreich abgeschlossen!")
        if args.dry_run:
            print("💡 Führe das Skript ohne --dry-run aus, um die Änderungen tatsächlich vorzunehmen")
    else:
        print("❌ Aufräumung fehlgeschlagen!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 