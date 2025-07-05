# Core Module

This directory contains the core functionality for the multimodal fact-checking pipeline.

## Key Components

- **pipeline.py**: Main pipeline for running experiments
- **model_handler.py**: Model loading and device management
- **data_loader.py**: Dataset loading and preprocessing
- **evaluation.py**: Metrics calculation and result analysis
- **utils.py**: Utility functions and helpers
- **logging_config.py**: Centralized logging configuration

## Usage

```python
from src.core import run_pipeline, FakedditDataset
from src.core.model_handler import load_clip, load_blip_conditional
from src.core.evaluation import evaluate_model_outputs
```

## Notes
- All core functions are importable from the main `src` package
- Handles device detection and memory optimization automatically
- Provides unified logging and error handling 