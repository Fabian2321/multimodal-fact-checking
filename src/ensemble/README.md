# Ensemble Module

This directory contains ensemble methods for combining multiple model predictions.

## Key Components

- **ensemble_handler.py**: Main ensemble handler for model combinations

## Usage

```python
from src.ensemble import EnsembleHandler

# Create ensemble with CLIP and BERT results
ensemble = EnsembleHandler(
    clip_results_path="results/clip/clip_zs_baseline/all_model_outputs.csv",
    bert_results_path="results/bert/bert_baseline/all_model_outputs.csv"
)

# Get ensemble predictions
ensemble_results = ensemble.get_ensemble_predictions(method="weighted_vote")
```

## Notes
- Supports weighted voting and other ensemble methods
- Combines results from different model types (CLIP, BERT, BLIP2, LLaVA)
- Provides improved robustness for fact-checking tasks 