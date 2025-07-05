"""
Centralized logging configuration for the MLLM project.
All logs are unified to the /logs directory at the project root.
"""

import os
import logging
from .utils import setup_logger

def get_experiment_logger(experiment_name, log_file=None):
    """
    Get a logger for experiments with structured logging.
    
    Args:
        experiment_name (str): Name of the experiment
        log_file (str, optional): Specific log file name
    
    Returns:
        logging.Logger: Configured logger for experiments
    """
    if log_file is None:
        log_file = f"experiments/{experiment_name}.log"
    
    return setup_logger(
        name=f"experiment.{experiment_name}",
        log_file=log_file
    )

def get_rag_logger(log_file=None):
    """
    Get a logger for RAG experiments.
    
    Args:
        log_file (str, optional): Specific log file name
    
    Returns:
        logging.Logger: Configured logger for RAG
    """
    if log_file is None:
        log_file = "rag/rag_experiments.log"
    
    return setup_logger(
        name="rag.experiments",
        log_file=log_file
    )

def get_error_logger(log_file=None):
    """
    Get a logger for error tracking.
    
    Args:
        log_file (str, optional): Specific log file name
    
    Returns:
        logging.Logger: Configured logger for errors
    """
    if log_file is None:
        log_file = "errors/error_tracking.log"
    
    return setup_logger(
        name="error.tracking",
        level=logging.ERROR,
        log_file=log_file
    )

def get_main_logger(log_file=None):
    """
    Get the main application logger.
    
    Args:
        log_file (str, optional): Specific log file name
    
    Returns:
        logging.Logger: Configured main logger
    """
    if log_file is None:
        log_file = "mllm.log"
    
    return setup_logger(
        name="mllm.main",
        log_file=log_file
    )

# Convenience function for backward compatibility
def setup_logger(name='mllm_project', level=logging.INFO, log_file=None, file_mode='a'):
    """
    Backward compatibility wrapper for the original setup_logger.
    """
    from .utils import setup_logger as original_setup_logger
    return original_setup_logger(name, level, log_file, file_mode) 