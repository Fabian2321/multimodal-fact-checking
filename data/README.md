# Data Directory Overview

Dieses Verzeichnis enthält alle für das Projekt relevanten Daten, Knowledge Bases und Testdaten. Die Inhalte sind logisch nach Typ und Zweck gegliedert:

## Struktur

- **datasets/**: Enthält alle Datensätze für Training, Test und Validierung
  - `processed/train/` – Trainingsdaten
  - `processed/test/` – Testdaten
  - `processed/validate/` – Validierungsdaten
  - `raw/` – Rohdaten (z.B. unbearbeitete Fakeddit-CSV-Dateien)
- **images/**: Bilddaten, z.B. Fakeddit-Images
  - `fakeddit/` – Alle zugehörigen Bilddateien
- **knowledge/**: Wissensbasis und externe Faktenquellen
  - `guidelines/` – Fact-Checking- und Bildanalyse-Guidelines (jeweils basic/enhanced)
  - `misconceptions/` – Missverständnisse (basic/enhanced)
  - `processed/` – Verarbeitete Knowledge Base (z.B. FAISS-Index, große JSONs)
- **test_queries/**: Test- und Evaluierungsdaten für RAG und Modelle
  - `rag_evaluation/` – Queries für RAG-Tests
  - `model_evaluation/` – Queries für Modell-Evaluation
- **few_shot_examples/**: Beispiele für Few-Shot-Learning und Prompt-Design
- **optimization_results/**: Ergebnisse von Optimierungs- und Grid-Search-Läufen

## Hinweise
- Große Binärdateien (z.B. FAISS-Index) sollten nicht versioniert werden (siehe .gitignore).
- Die Knowledge Base ist versioniert (basic/enhanced) und modular aufgebaut.
- Für Details zu einzelnen Unterordnern siehe ggf. dortige README.md. 