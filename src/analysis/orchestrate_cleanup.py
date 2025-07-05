#!/usr/bin/env python3
"""
Master-Skript für die Results-Ordner Aufräumung
Orchestriert den gesamten sicheren Aufräumprozess
"""

import subprocess
import sys
import time
from pathlib import Path

def run_script(script_name, args=None):
    """Führe ein Python-Skript aus"""
    cmd = [sys.executable, f"scripts/{script_name}"]
    if args:
        cmd.extend(args)
    
    print(f"🚀 Führe aus: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler beim Ausführen von {script_name}:")
        print(f"   Exit Code: {e.returncode}")
        print(f"   Stdout: {e.stdout}")
        print(f"   Stderr: {e.stderr}")
        return False

def main():
    """Hauptfunktion - Orchestriert den Aufräumprozess"""
    
    print("🎯 Results-Ordner Aufräumung - Master-Skript")
    print("=" * 60)
    print("Dieses Skript führt den kompletten sicheren Aufräumprozess durch:")
    print("1. 📊 Detaillierte Analyse des results-Ordners")
    print("2. 📋 Zusammenfassung der wichtigsten Ergebnisse")
    print("3. 🛡️  Sicheres Backup aller wichtigen Dateien")
    print("4. 🧹 Aufräumung des results-Ordners")
    print("=" * 60)
    
    # Prüfe ob results-Ordner existiert
    if not Path("results").exists():
        print("❌ Results-Ordner nicht gefunden!")
        return 1
    
    # Schritt 1: Analyse
    print("\n📊 SCHRITT 1: Detaillierte Analyse")
    if not run_script("analyze_results.py"):
        print("❌ Analyse fehlgeschlagen! Stoppe Aufräumung.")
        return 1
    
    # Schritt 2: Zusammenfassung
    print("\n📋 SCHRITT 2: Zusammenfassung erstellen")
    if not run_script("create_results_summary.py"):
        print("❌ Zusammenfassung fehlgeschlagen! Stoppe Aufräumung.")
        return 1
    
    # Schritt 3: Sicheres Backup
    print("\n🛡️  SCHRITT 3: Sicheres Backup erstellen")
    if not run_script("create_safe_backup.py"):
        print("❌ Backup fehlgeschlagen! Stoppe Aufräumung.")
        return 1
    
    # Bestätigung für Aufräumung
    print("\n" + "=" * 60)
    print("✅ Alle Sicherheitsmaßnahmen abgeschlossen!")
    print("📊 Analyse: ✅")
    print("📋 Zusammenfassung: ✅")
    print("🛡️  Backup: ✅")
    print("=" * 60)
    
    print("\n🧹 Bereit für die Aufräumung!")
    print("Das Skript wird jetzt:")
    print("1. Eine neue, saubere Struktur erstellen")
    print("2. Dateien nach Modellen organisieren")
    print("3. Test-Ordner ins Archive verschieben")
    print("4. Wichtige Dateien im final_results Ordner bewahren")
    print("5. Leere Ordner entfernen")
    
    # Kurze Pause für Benutzer
    print("\n⏳ Starte Aufräumung in 3 Sekunden...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    # Schritt 4: Aufräumung
    print("\n🧹 SCHRITT 4: Aufräumung durchführen")
    if not run_script("cleanup_results.py"):
        print("❌ Aufräumung fehlgeschlagen!")
        print("💡 Du kannst das Backup verwenden, um Dateien wiederherzustellen.")
        return 1
    
    # Finale Zusammenfassung
    print("\n" + "=" * 60)
    print("🎉 AUFRÄUMUNG ERFOLGREICH ABGESCHLOSSEN!")
    print("=" * 60)
    
    print("\n📁 Neue Struktur:")
    print("   results/")
    print("   ├── final_results/     # Wichtigste Ergebnisse")
    print("   ├── experiments/       # Alle Experimente nach Modell")
    print("   │   ├── clip/")
    print("   │   ├── blip2/")
    print("   │   ├── llava/")
    print("   │   ├── bert/")
    print("   │   └── ensemble/")
    print("   ├── archive/           # Alte/Test-Experimente")
    print("   └── analysis/          # Analysen und Visualisierungen")
    
    print("\n🛡️  Sicherheit:")
    print("   - Vollständiges Backup erstellt")
    print("   - Alle wichtigen Dateien gesichert")
    print("   - Aufräumbericht verfügbar")
    
    print("\n📋 Verfügbare Dateien:")
    print("   - results/cleanup_report.json")
    print("   - results/results_summary.json")
    print("   - results/backup_file_list.txt")
    print("   - results_safe_backup_*/ (Backup-Ordner)")
    
    print("\n✅ Aufräumung erfolgreich abgeschlossen!")
    return 0

if __name__ == "__main__":
    exit(main()) 