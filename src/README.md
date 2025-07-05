# src/ – Multimodal Fact-Checking Source Code

Dieses Verzeichnis enthält den vollständigen, modularen Quellcode für das Multimodal Fact-Checking Projekt.

## Struktur

```
core/        # Kernfunktionen: Pipeline, Modelle, Daten, Utils
models/      # Parser, Prompts, RAG-Handler
ensemble/    # Ensemble-Methoden
analysis/    # Analyse- und Auswertungs-Tools
experiments/ # CLI-Experiment-Skripte (direkt ausführen)
data/        # Knowledge Base (z.B. documents.json, faiss_index.bin)
logs/        # Log-Dateien
```

## Hauptmodule & Funktionen

- **core/**: Daten-Handling, Modell-Loader, Pipeline, Evaluation
- **models/**: BLIP/LLaVA Parser, Prompts, RAG-Handler
- **ensemble/**: EnsembleHandler für Modell-Kombinationen
- **analysis/**: Tools für Auswertung, Backup, Cleanup
- **experiments/**: CLI-Skripte für Experimente (nicht importieren, sondern direkt ausführen)

## Verwendung

### Als Python-Modul
```python
from src import run_pipeline, BLIPAnswerParser, RAGHandler
from src.core import FakedditDataset, setup_logger
from src.models import BLIP_PROMPTS, LLAVA_PROMPTS
```

### CLI-Skripte ausführen
```bash
python src/experiments/test_llava_mini.py
python src/analysis/cleanup_results.py
```

## Hinweise
- **Experiments** und **Analysis** enthalten CLI-Tools, die direkt ausgeführt werden.
- Alle wichtigen Funktionen sind über das Hauptpaket `src` importierbar.
- Die Knowledge Base liegt in `src/data/knowledge_base/`.

---

**Letzte Migration:** Juli 2025 – Struktur modularisiert, Altlasten entfernt, alle Funktionen getestet. 