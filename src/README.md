# src/ – Multimodal Fact-Checking Source Code

This directory contains the complete, modular source code for the Multimodal Fact-Checking project.

## Structure

```
core/        # Core functions: Pipeline, Models, Data, Utils
models/      # Parsers, Prompts, RAG-Handler
ensemble/    # Ensemble methods
analysis/    # Analysis and evaluation tools
experiments/ # CLI experiment scripts (execute directly)
data/        # Knowledge Base (e.g., documents.json, faiss_index.bin)
logs/        # Log files
```

## Main modules & functions

- **core/**: Data handling, model loader, pipeline, evaluation
- **models/**: BLIP/LLaVA parsers, prompts, RAG handler
- **ensemble/**: EnsembleHandler for model combinations
- **analysis/**: Tools for evaluation, backup, cleanup
- **experiments/**: CLI scripts for experiments (do not import, execute directly)

## Usage

### As Python module
```python
from src import run_pipeline, BLIPAnswerParser, RAGHandler
from src.core import FakedditDataset, setup_logger
from src.models import BLIP_PROMPTS, LLAVA_PROMPTS
```

### Execute CLI scripts
```bash
python src/experiments/test_llava_mini.py
python src/analysis/cleanup_results.py
```

## Notes
- **Experiments** and **Analysis** contain CLI tools that are executed directly.
- All important functions are importable via the main package `src`.
- The knowledge base is located in `src/data/knowledge_base/`.

---

**Last migration:** July 2025 – Structure modularized, legacy code removed, all functions tested. 