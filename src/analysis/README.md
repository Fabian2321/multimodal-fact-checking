# Analysis Module

This directory contains tools for analyzing experiment results and generating reports.

## Key Components

- **analyze_results.py**: Comprehensive result analysis and comparison
- **evaluate_rag.py**: RAG-specific evaluation metrics
- **generate_examples_and_analysis.py**: Example generation and analysis
- **generate_final_results.py**: Final result compilation and reporting

## Usage

```python
from src.analysis import analyze_results, evaluate_rag

# Analyze experiment results
analysis = analyze_results(results_path="results/experiments/")
analysis.generate_comparison_report()

# Evaluate RAG performance
rag_eval = evaluate_rag(rag_results_path="results/rag/")
rag_eval.calculate_retrieval_metrics()
```

## Notes
- Provides comprehensive evaluation and analysis tools
- Generates comparison reports and visualizations
- Supports both individual model and ensemble analysis 