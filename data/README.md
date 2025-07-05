# MLLM Project - Data Directory

This directory contains all data for the MLLM (Multimodal Large Language Model) project, organized in a unified structure.

## Directory Structure

```
data/
├── README.md              # This file
├── knowledge_base/        # RAG (Retrieval-Augmented Generation) data
│   ├── documents.json     # Knowledge base documents (873MB)
│   └── faiss_index.bin    # FAISS index for fast retrieval (3.8GB)
├── knowledge/             # General knowledge resources
│   ├── misconceptions/    # Common misconceptions data
│   ├── guidelines/        # Fact-checking guidelines
│   └── processed/         # Processed knowledge data
├── datasets/              # Dataset files
├── images/                # Image data
├── few_shot_examples/     # Few-shot learning examples
├── optimization_results/  # Optimization experiment results
└── test_queries/          # Test query data
```

## Knowledge Base (RAG)

The `knowledge_base/` directory contains the RAG system data:

- **`documents.json`** (873MB): Structured knowledge documents for retrieval
- **`faiss_index.bin`** (3.8GB): FAISS index for fast similarity search

### Usage in Code

```python
# RAG configuration
rag_config = RAGConfig(knowledge_base_path="data/knowledge_base")

# Load knowledge base
with open("data/knowledge_base/documents.json", "r") as f:
    documents = json.load(f)
```

## Data Organization

### Knowledge Base
- **Purpose**: RAG system for enhanced fact-checking
- **Size**: ~4.7GB total
- **Format**: JSON documents + FAISS binary index

### Knowledge Resources
- **Purpose**: General knowledge and guidelines
- **Content**: Misconceptions, guidelines, processed data
- **Format**: Various (JSON, CSV, etc.)

### Datasets
- **Purpose**: Training and evaluation datasets
- **Content**: Fakeddit, test data, etc.
- **Format**: CSV, JSON

### Images
- **Purpose**: Image data for multimodal experiments
- **Content**: Downloaded images, processed images
- **Format**: JPG, PNG

## Migration Notes

**Before**: Data was scattered across:
- `/Users/fabian/mllm/data/` - Main data
- `/Users/fabian/mllm/src/data/knowledge_base/` - RAG data

**After**: Unified structure in `/Users/fabian/mllm/data/`

## Benefits

1. **Unified Location**: All data in one place
2. **Clear Organization**: Logical separation by purpose
3. **Easy Maintenance**: Consistent structure
4. **Better Performance**: RAG data optimized for fast retrieval

## File Sizes

- **Total Knowledge Base**: ~4.7GB
- **Documents**: 873MB
- **FAISS Index**: 3.8GB
- **Other Data**: Varies by dataset

## Usage Guidelines

1. **Large Files**: Knowledge base files are large, handle with care
2. **Backup**: Consider backing up knowledge base before major changes
3. **Version Control**: Large files should be in `.gitignore`
4. **Performance**: FAISS index enables fast similarity search 