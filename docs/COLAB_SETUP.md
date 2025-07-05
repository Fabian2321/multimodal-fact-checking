# Google Colab Setup Guide

This guide explains how to run the multimodal fact-checking experiments in Google Colab.

## Prerequisites

1. A Google account to access Google Colab
2. The following files from this repository:
   - Your chosen script from the `scripts/` directory
   - Test data from `data/processed/test_balanced_pairs_clean.csv`
   - Image dataset (`colab_images.zip`)

## Setup Steps

### 1. Open Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Enable GPU acceleration:
   - Click Runtime → Change runtime type
   - Select "GPU" from the Hardware accelerator dropdown
   - Click "Save"

### 2. Install Dependencies

Copy and run this cell:
```python
!pip install transformers torch pandas pillow requests scikit-learn matplotlib seaborn nltk
```

### 3. Upload Required Files

1. Upload the data files:
```python
from google.colab import files

# Upload the test data
uploaded = files.upload()  # Select test_balanced_pairs_clean.csv

# Upload and extract images
uploaded = files.upload()  # Select colab_images.zip
!unzip -q colab_images.zip -d colab_images
```

2. Upload your chosen script:
```python
uploaded = files.upload()  # Select your script (e.g., colab_llava_runner.py)
```

### 4. Run the Experiment

Execute your chosen script:
```python
!python your_script_name.py
```

## Available Scripts

Choose one of these scripts for your experiment:

- `colab_llava_runner.py`: Basic LLaVA model
- `colab_clip_standalone_79_percent.py`: Basic CLIP model
- `colab_clip_optimized_80_percent.py`: Optimized CLIP model
- `llava_clip_ensemble.py`: Combined LLaVA and CLIP model

## Troubleshooting

### Check GPU Availability
If you experience performance issues, verify GPU access:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Common Issues

1. **Out of Memory**
   - Reduce batch size in the script
   - Restart runtime and run again

2. **Missing Files**
   - Ensure all required files are uploaded
   - Check file paths in error messages

3. **Slow Execution**
   - Verify GPU is enabled
   - Check internet connection for stable access 