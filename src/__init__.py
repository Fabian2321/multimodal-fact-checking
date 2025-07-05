"""
Multimodal fact-checking 'src' package - Reorganized Structure

This package contains all components for multimodal fact-checking experiments:
- Core functionality (pipeline, models, data loading)
- Model-specific components (parsers, prompts, RAG)
- Ensemble methods
- Experiment scripts
- Analysis and reporting tools
"""

# Core functionality
from .core import (
    run_pipeline,
    load_clip, 
    process_batch_for_clip,
    load_blip_conditional,
    process_batch_for_blip_conditional,
    load_bert_classifier,
    process_batch_for_bert_classifier,
    load_llava,
    process_batch_for_llava,
    FakedditDataset,
    evaluate_model_outputs,
    compute_qualitative_stats,
    save_metrics_table,
    setup_logger
)

# Model-specific components
from .models import (
    BLIPAnswerParser, 
    parse_blip_answer, 
    LLaVAAnswerParser,
    parse_llava_answer,
    BLIP_PROMPTS,
    LLAVA_PROMPTS,
    CLIP_PROMPTS,
    FEW_SHOT_EXAMPLES,
    build_blip2_true_false_fewshot_prompt,
    prompt_library,
    RAGHandler,
    RAGConfig
)

# Ensemble methods
from .ensemble import EnsembleHandler

# Analysis tools
from .analysis import generate_final_results, generate_examples_and_analysis

print("Multimodal fact-checking 'src' package initialized with new modular structure.")

__all__ = [
    # Core
    'run_pipeline',
    'load_clip', 'process_batch_for_clip',
    'load_blip_conditional', 'process_batch_for_blip_conditional',
    'load_bert_classifier', 'process_batch_for_bert_classifier',
    'load_llava', 'process_batch_for_llava',
    'FakedditDataset',
    'evaluate_model_outputs', 'compute_qualitative_stats', 'save_metrics_table',
    'setup_logger',
    # Models
    'BLIPAnswerParser', 'parse_blip_answer',
    'LLaVAAnswerParser', 'parse_llava_answer',
    'BLIP_PROMPTS', 'LLAVA_PROMPTS', 'CLIP_PROMPTS',
    'FEW_SHOT_EXAMPLES', 'build_blip2_true_false_fewshot_prompt',
    'prompt_library', 'RAGHandler', 'RAGConfig',
    # Ensemble
    'EnsembleHandler',
    # Analysis
    'generate_final_results', 'generate_examples_and_analysis',
]
