# Multimodal Fact-Checking Project

This project develops and evaluates a multimodal fact-checking system using Vision-Language Models (VLMs) and Retrieval-Augmented Generation (RAG).

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Setup & Installation](#setup--installation)
3. [Data & Preparation](#data--preparation)
4. [Running the Pipeline](#running-the-pipeline)
5. [RAG: Knowledge Base & Retrieval](#rag-knowledge-base--retrieval)
6. [RAG Parameter Optimization](#rag-parameter-optimization)
7. [Shell Scripts for Experiments](#shell-scripts-for-experiments)
8. [Tests & Analysis](#tests--analysis)
9. [Troubleshooting & Tips](#troubleshooting--tips)
10. [Dependencies](#dependencies)

---

## Project Structure

- `.venv/`: Python virtual environment (not versioned)
- `data/`: Datasets and knowledge base
  - `raw/`: Raw data (e.g., Fakeddit CSVs)
  - `downloaded_fakeddit_images/`: Images downloaded automatically
  - `external_knowledge/`: JSON files with external knowledge (e.g., guidelines, misconceptions)
  - `knowledge_base/`: Persistent knowledge base for RAG (FAISS index, documents)
  - `optimization_results/`: Results of parameter optimization
- `models/`: (Optional) Model checkpoints
- `notebooks/`: Jupyter notebooks for analysis and visualization
- `results/`: Experiment results, reports, figures
- `src/`: Source code
  - `pipeline.py`: Main fact-checking pipeline
  - `data_loader.py`: FakedditDataset and data preprocessing
  - `model_handler.py`: Loading and applying VLMs (CLIP, BLIP, LLaVA, BERT)
  - `evaluation.py`: Metrics and evaluation
  - `rag_handler.py`: RAG logic (knowledge base, retrieval, prompt formatting)
  - `create_knowledge_base.py`: Build the knowledge base for RAG
  - `optimize_rag_params.py`: Grid search for RAG parameters
  - `run_optimization.py`, `run_optimization_chunk.py`: Resource-safe optimization
  - Other helper scripts
- `tests/`: Analysis and test scripts
- `.gitignore`, `requirements.txt`, `README.md`

---

## Setup & Installation

1. **Clone the repository (if needed):**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Data & Preparation

1. **Place Fakeddit CSVs** in `data/raw/` (e.g., `multimodal_train.csv`, `multimodal_test_public.csv`).
2. **Images** are downloaded automatically on first run and stored in `data/downloaded_fakeddit_images/`.
3. **External knowledge sources** (optional, recommended for RAG):
   - Place JSON files like `common_misconceptions.json`, `fact_checking_guidelines.json` in `data/external_knowledge/`.

---

## Running the Pipeline

The main pipeline is in `src/pipeline.py`. It supports various models and options:

### Example Commands

**CLIP:**
```bash
python src/pipeline.py --model_type clip --clip_model_name openai/clip-vit-base-patch32 --experiment_name clip_test --num_samples 100
```

**BLIP (VQA):**
```bash
python src/pipeline.py --model_type blip --blip_task vqa --num_test_batches 1
```

**BLIP (Captioning):**
```bash
python src/pipeline.py --model_type blip --blip_task captioning --num_test_batches 1
```

**LLaVA:**
```bash
python src/pipeline.py --model_type llava --llava_model_name <name> --num_test_batches 1
```

**With RAG and Few-Shot:**
```bash
python src/pipeline.py --model_type blip --use_rag --rag_knowledge_base_path data/knowledge_base --use_few_shot --prompt_name fs_yesno_justification
```

**Show all options:**
```bash
python src/pipeline.py --help
```

### Key Arguments (Selection)

- `--model_type`: clip | blip | llava | bert
- `--clip_model_name`, `--blip_model_name`, `--llava_model_name`
- `--experiment_name`: Name for the experiment/results folder
- `--num_samples`, `--num_test_batches`
- `--use_few_shot`: Enable few-shot prompts
- `--use_rag`: Enable Retrieval-Augmented Generation
- `--rag_knowledge_base_path`: Path to the knowledge base
- `--prompt_name`: Name of the prompt template

---

## RAG: Knowledge Base & Retrieval

### Creating the Knowledge Base

Use `src/create_knowledge_base.py` to build an initial knowledge base for RAG:

```bash
python src/create_knowledge_base.py \
  --fakeddit_path data/raw/multimodal_train.csv \
  --external_path data/external_knowledge \
  --output_path data/knowledge_base \
  --embedding_model all-MiniLM-L6-v2
```

- The knowledge base consists of a FAISS index and a document list.
- External knowledge sources are automatically integrated if present in the specified directory.

### Using RAG in the Pipeline

- Enable RAG with `--use_rag` and specify the knowledge base (`--rag_knowledge_base_path`).
- The pipeline will retrieve relevant documents for each query and integrate them into the prompt.

---

## RAG Parameter Optimization

To maximize retrieval quality, you can perform grid search optimization:

### Start Optimization

```bash
python src/optimize_rag_params.py \
  --test_queries_path data/test_queries.json \
  --knowledge_base_path data/knowledge_base \
  --output_path data/optimization_results
```

- Various parameter combinations (embedding model, top_k, thresholds, etc.) are tested.
- The best parameters are saved in `optimization_results.json`.

### Resource-Safe Optimization (recommended for large grids)

Use `src/run_optimization.py` or the shell script:

```bash
bash run_optimization.sh
```

- Automatically pauses if RAM usage is high.
- Checkpoints are saved regularly.

---

## Shell Scripts for Experiments

- **run_optimization.sh**: Starts optimization with standard parameters and logging.
- **run_optimization_chunk.sh <model_name>**: Optimizes for a specific embedding model.

Example:
```bash
bash run_optimization_chunk.sh all-MiniLM-L6-v2
```

---

## Tests & Analysis

- The `tests/` folder contains analysis scripts for thresholds, model comparisons, etc.
- Example:
  ```bash
  python tests/analyze_blip_yes_no.py
  ```

---

## Troubleshooting & Tips

- **Import errors:** Always work from the project root and activate the virtual environment.
- **Memory issues:** Use the resource-safe optimization scripts.
- **Missing data:** Ensure all required CSVs and JSONs are in the correct directories.
- **Model downloads:** Large models are downloaded automatically on first run (internet connection required).

---

## Dependencies

All required packages are listed in `requirements.txt` with fixed versions. Install them with:

```bash
pip install -r requirements.txt
```

Key packages:
- torch, torchvision, transformers, sentence-transformers, faiss-cpu
- pandas, numpy, scikit-learn, matplotlib, seaborn
- tqdm, requests, Pillow, Jupyter, and more

---

**For questions or contributions: Please use issues or pull requests in the repository!**
