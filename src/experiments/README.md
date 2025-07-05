# Experiments Module

This directory contains CLI scripts for running various experiments.

## Key Components

### CLIP Experiments
- **clip_optimized_80_percent.py**: CLIP with 80% accuracy target
- **clip_85_percent_target.py**: CLIP with 85% accuracy target
- **clip_ultimate_90_percent.py**: CLIP with 90% accuracy target
- **clip_aggressive_85_percent.py**: Aggressive CLIP optimization

### BLIP2 Experiments
- **colab_blip2_rag_fewshot.py**: BLIP2 with RAG and few-shot
- **colab_blip2_optimized.py**: Optimized BLIP2 configuration
- **debug_blip2_responses.py**: BLIP2 response debugging

### LLaVA Experiments
- **test_llava_mini.py**: LLaVA model testing
- **colab_llava_runner.py**: LLaVA runner for Colab

### Shell Scripts
- **run_ensemble_experiments.sh**: Ensemble experiment runner
- **run_final_experiments.sh**: Final experiment execution
- **run_clip_threshold_optimization.sh**: CLIP threshold optimization

## Usage

```bash
# Run CLIP experiment
python src/experiments/clip_optimized_80_percent.py

# Run ensemble experiments
bash src/experiments/run_ensemble_experiments.sh

# Run threshold optimization
bash src/experiments/run_clip_threshold_optimization.sh
```

## Notes
- All scripts are designed to run independently
- Results are saved to the `results/` directory
- Use shell scripts for batch experiment execution 