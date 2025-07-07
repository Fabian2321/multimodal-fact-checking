# 🔍 Multimodal Fact-Checking Using Vision-Language Large Language Models

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.52+-yellow.svg)](https://huggingface.co/transformers/)
[![Fakeddit](https://img.shields.io/badge/Fakeddit-Dataset-green.svg)](https://fakeddit.netlify.app)

A comprehensive research project implementing and evaluating state-of-the-art multimodal models for **text-image matching** in fact-checking applications using the Fakeddit dataset. This project explores CLIP, BLIP2, LLaVA, and BERT architectures with advanced techniques including RAG (Retrieval-Augmented Generation), few-shot learning, and ensemble methods to determine whether textual claims accurately correspond to accompanying images.

## 🎯 Key Results

| Model | Architecture | Accuracy | F1-Score | ROC AUC | Speed (s/sample) | Best Feature |
|-------|-------------|----------|----------|---------|------------------|--------------|
| **CLIP** | ViT-Base/16 | **82.0%** | 0.784 | 0.814 | **0.1** | Fastest & Most Accurate |
| **CLIP** | ViT-Large/14 | 81.0% | **0.819** | **0.842** | 0.15 | Best F1 & AUC |
| **LLaVA** | 1.5-7B | 65.0% | 0.667 | 0.675 | 0.93 | Best Reasoning |
| **BLIP2** | OPT-2.7B | 62.0% | 0.296 | 0.428 | 12.48 | RAG Enhanced |

### 🏆 Performance Highlights

- **Best Overall**: CLIP ViT-Base/16 achieves 82.0% accuracy in text-image matching with 10x faster inference than LLaVA
- **Best ROC AUC**: CLIP ViT-Large/14 reaches 0.842 AUC with superior F1-score for claim verification
- **RAG Impact**: BLIP2 shows +6.0% improvement with RAG integration for contextual understanding
- **Efficiency**: CLIP processes 10x faster than LLaVA and 120x faster than BLIP2 for real-time fact-checking

## 🏗️ Architecture Overview

```
mllm/
├── 📁 src/                    # Core implementation
│   ├── core/                 # Pipeline, models, data handling
│   ├── models/               # RAG, parsers, prompts
│   ├── ensemble/             # Model combination methods
│   ├── analysis/             # Evaluation and metrics
│   └── experiments/          # CLI scripts for experiments
├── 📁 data/                  # Dataset and knowledge base
├── 📁 results/               # Experiment outputs and metrics
├── 📁 docs/                  # Documentation and reports
└── 📁 archive/               # Historical experiments
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mllm.git
cd mllm

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
# CLIP model with optimized threshold
python src/experiments/clip_optimized_80_percent.py

# BLIP2 with RAG enhancement
python src/experiments/colab_blip2_rag_fewshot.py

# LLaVA model evaluation
python src/experiments/test_llava_mini.py

# Ensemble experiments
python src/experiments/run_ensemble_experiments.sh
```

### 3. Google Colab Setup

For cloud-based experimentation, see [Colab Setup Guide](docs/COLAB_SETUP.md).

## 🔬 Research Contributions

### Model Implementations

#### CLIP (Contrastive Language-Image Pre-training)
- **Architecture**: ViT-Base/16 and ViT-Large/14 variants
- **Optimization**: Threshold tuning (0.272 optimal) for text-image similarity
- **Enhancements**: Multi-crop preprocessing, RAG integration for context
- **Performance**: 82.0% accuracy in claim verification, 0.1s/sample

#### BLIP2 (Bootstrapping Language-Image Pre-training)
- **Architecture**: OPT-2.7B language model
- **Features**: Few-shot learning, RAG enhancement for claim analysis
- **Improvements**: +6.0% with RAG, +2.0% with few-shot for contextual matching
- **Challenges**: High latency (12.48s/sample) but superior reasoning

#### LLaVA (Large Language and Vision Assistant)
- **Architecture**: 1.5-7B model with CLIP ViT-L/14 vision encoder
- **Strengths**: Natural language reasoning for claim verification
- **Performance**: 65.0% accuracy, excellent explanation quality for fact-checking
- **Optimization**: Memory-efficient processing for large-scale verification

### Advanced Techniques

#### RAG (Retrieval-Augmented Generation)
```python
# Knowledge base integration for fact-checking
rag_handler = RAGHandler(config)
retrieved_docs = rag_handler.retrieve(query)
enhanced_prompt = rag_handler.format_rag_prompt(text, docs)
```

#### Ensemble Methods
```python
# Weighted voting ensemble for robust fact-checking
weights = {'CLIP': 0.5, 'BLIP2': 0.3, 'LLaVA': 0.2}
ensemble_prediction = weighted_vote(predictions, weights)
```

#### Few-Shot Learning
- Dynamic example selection for claim verification
- Balanced positive/negative examples for robust training
- Context-aware prompting for improved accuracy

## 📊 Dataset & Evaluation

### Fakeddit Dataset
- **Size**: 1,000 balanced samples (500 matching, 500 mismatched)
- **Content**: Reddit posts with images and textual claims
- **Quality**: Manually verified text-image correspondence, cleaned metadata
- **Features**: Multi-modal, real-world distribution for fact-checking evaluation

### Evaluation Metrics
- **Primary**: Accuracy, F1-Score, ROC AUC for claim verification
- **Secondary**: Precision, Recall, Processing Time for real-time fact-checking
- **Qualitative**: Generated explanations, error analysis for interpretability

## 🛠️ Technical Implementation

### Core Pipeline
```python
from src.core import run_pipeline, FakedditDataset
from src.models import RAGHandler, BLIPAnswerParser

# Load dataset for text-image matching evaluation
dataset = FakedditDataset(metadata_dir, image_dir)

# Run fact-checking experiment
results = run_pipeline(
    model_type='clip',
    clip_model_name='openai/clip-vit-base-patch32',
    use_rag=True,
    batch_size=32
)
```

### Model Handler
```python
from src.core.model_handler import load_clip, load_blip_conditional

# Load models with automatic device detection for fact-checking
clip_model, clip_processor = load_clip('openai/clip-vit-base-patch32')
blip_model, blip_processor = load_blip_conditional('Salesforce/blip2-opt-2.7b')
```

### Evaluation Framework
```python
from src.core.evaluation import evaluate_model_outputs

# Comprehensive evaluation for claim verification
evaluate_model_outputs(
    results_df,
    true_label_col='true_label',
    pred_label_col='predicted_label',
    report_path='evaluation_report.txt'
)
```

## 📈 Results Analysis

### Performance Comparison
- **CLIP Dominance**: Best accuracy and speed across all metrics for text-image matching
- **RAG Effectiveness**: Significant improvements for BLIP2 (+6.0%) in contextual fact-checking
- **Ensemble Benefits**: Matches single model performance (82.0%) for robust verification
- **Resource Efficiency**: CLIP requires minimal GPU memory (2GB) for scalable deployment

### Error Analysis
- **Visual Artifacts**: 15-25% of errors across models in image interpretation
- **Context Errors**: 25-40% of errors in claim verification, addressed by RAG
- **Technical Issues**: 25-35% of errors in text-image alignment, reduced by optimization

## 🔧 Configuration & Customization

### Model Configuration
```python
# CLIP configuration for text-image matching
clip_config = {
    'threshold': 0.272,
    'multi_crop': True,
    'crop_count': 3,
    'embedding_norm': 'l2'
}
```

# BLIP2 configuration for claim verification
blip_config = {
    'max_new_tokens': 75,
    'temperature': 1.0,
    'top_k': 10,
    'do_sample': False
}
```

### RAG Configuration for Fact-Checking
```python
rag_config = RAGConfig(
    embedding_model='sentence-transformers/all-MiniLM-L6-v2',
    top_k=5,
    similarity_threshold=0.7,
    knowledge_base_path='src/data/knowledge_base/'
)
```

## 📚 Documentation

- **[Final Report](docs/FINAL_REPORT.md)**: Comprehensive analysis and findings
- **[Experiment Configurations](docs/experiment_configurations.md)**: Detailed model settings
- **[Colab Setup](docs/COLAB_SETUP.md)**: Cloud environment guide

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Fakeddit Dataset**: Reddit-based text-image matching dataset for fact-checking
- **Hugging Face**: Model implementations and transformers library
- **OpenAI**: CLIP model architecture for vision-language understanding
- **Salesforce**: BLIP2 model for multimodal reasoning
- **Microsoft**: LLaVA model for large language and vision assistance

## 📞 Contact

For questions, suggestions, or collaborations:
**Email**: [fabian.loeffler@tum.de]

---

**⭐ Star this repository if you find it useful for your research!**

*This project represents a comprehensive study in multimodal fact-checking through text-image matching, achieving state-of-the-art results with efficient, scalable implementations for real-world verification applications.*
