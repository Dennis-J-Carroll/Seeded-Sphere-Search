"""
Evaluation module for SeededSphere search engine performance metrics and analysis.

This module provides tools for:
- Evaluating search performance with standard IR metrics
- Comparing different search configurations
- Analyzing echo layer impact
- Generating synthetic test data
- Visualizing evaluation results
"""

from .metrics import evaluate_search_performance
from .comparison import compare_configurations, plot_comparison_results
from .echo_analysis import evaluate_echo_layer_impact
from .test_data import generate_test_data

__all__ = [
    'evaluate_search_performance',
    'compare_configurations',
    'plot_comparison_results',
    'evaluate_echo_layer_impact',
    'generate_test_data',
]
