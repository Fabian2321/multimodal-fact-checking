# Multimodal Fact-Checking Project

This project aims to develop and evaluate a multimodal fact-checking system using Vision-Language Models (VLMs).

## Project Structure

- `.venv/`: Python virtual environment.
- `data/`: Contains datasets.
  - `raw/`: Raw dataset files (e.g., Fakeddit CSVs).
  - `downloaded_fakeddit_images/`: Directory where images are downloaded by `data_loader.py` (if run directly).
- `models/`: Can store model-specific files, custom model definitions, or saved model checkpoints (though large checkpoints might be better handled with Git LFS or stored elsewhere).
- `notebooks/`: Jupyter notebooks for experimentation, analysis, and visualization.
- `results/`: For storing experiment results, figures, reports.
  - `figures/`
  - `reports/`
- `src/`: Source code for the project.
  - `data_loader.py`: Defines the `FakedditDataset` for loading and preprocessing data.
  - `model_handler.py`: Contains functions for loading models (CLIP, BLIP) and their processors, and for processing data batches for these models.
  - `pipeline.py`: Main script for running the fact-checking pipeline, including data loading, model inference, and basic output.
  - `evaluation.py`: (Placeholder) Will contain evaluation metrics and scripts.
  - `utils.py`: (Placeholder) For utility functions.
- `tests/`: For unit tests and integration tests.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `requirements.txt`: Lists project dependencies.
- `README.md`: This file.

## Setup

1. **Clone the repository (if applicable).**
2. **Create and activate a Python virtual environment:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download datasets:**
   - Place Fakeddit metadata CSV files (e.g., `multimodal_train.csv`) into the `data/raw/` directory.
   - Images will be downloaded automatically by the scripts when first accessed.

## Running the Pipeline

The main pipeline can be run using `src/pipeline.py`.

Example (run from the project root directory):

```bash
source .venv/bin/activate
python src/pipeline.py --model_type clip --clip_model_name openai/clip-vit-base-patch32 --experiment_name clip_initial_test --num_samples 10
```

Or for BLIP (VQA task by default):

```bash
python src/pipeline.py --model_type blip --blip_task vqa --num_test_batches 1
```

For BLIP captioning:

```bash
python src/pipeline.py --model_type blip --blip_task captioning --num_test_batches 1
```

Use `python src/pipeline.py --help` to see all available options.

## Development Notes

- Ensure your Python interpreter in VS Code is set to the project's `.venv/bin/python`.
- The `FakedditDataset` in `data_loader.py` is configured by `pipeline.py` to return PIL images (using `transform=None`). These PIL images are then processed by model-specific processors (e.g., `CLIPProcessor`, `BlipProcessor`) within `model_handler.py`, which handle the necessary image transformations. If using `FakedditDataset` or `model_handler.py` components directly, ensure this image handling strategy is maintained for compatibility.
