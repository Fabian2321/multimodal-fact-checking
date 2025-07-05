"""
Core functionality for the MLLM project.
Contains the main pipeline, model handling, data loading, and evaluation components.
"""

from .pipeline import main as run_pipeline
from .model_handler import (
    load_clip, 
    process_batch_for_clip,
    load_blip_conditional,
    process_batch_for_blip_conditional,
    load_bert_classifier,
    process_batch_for_bert_classifier,
    load_llava,
    process_batch_for_llava
)
from .data_loader import FakedditDataset
from .evaluation import evaluate_model_outputs, compute_qualitative_stats, save_metrics_table
from .utils import setup_logger

__all__ = [
    'run_pipeline',
    'load_clip',
    'process_batch_for_clip', 
    'load_blip_conditional',
    'process_batch_for_blip_conditional',
    'load_bert_classifier',
    'process_batch_for_bert_classifier',
    'load_llava',
    'process_batch_for_llava',
    'FakedditDataset',
    'evaluate_model_outputs',
    'compute_qualitative_stats',
    'save_metrics_table',
    'setup_logger'
] 