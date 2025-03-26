"""
Tools for comparing different search engine configurations and visualizing results.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .metrics import evaluate_search_performance

def compare_configurations(search_engine, test_data, configurations, output_file="comparison_results.json"):
    """
    Compare different configurations of the search engine
    
    Args:
        search_engine: Search engine instance
        test_data: Test data for evaluation
        configurations: List of configuration dictionaries
        output_file: File to save results
        
    Returns:
        dict: Evaluation results for each configuration
    """
    results = {}
    
    for i, config in enumerate(configurations):
        print(f"Evaluating configuration {i+1}/{len(configurations)}: {config}")
        
        # Backup original configuration
        original_config = search_engine.config.copy()
        
        # Apply new configuration
        for key, value in config.items():
            if key in search_engine.config:
                search_engine.config[key] = value
        
        # Re-run necessary initialization
        if 'dimensions' in config:
            # Would need to reinitialize the model
            print("Cannot change dimensions without reinitializing the model")
        elif any(k in config for k in ['min_echo_layers', 'max_echo_layers', 'delta']):
            # Re-run echo refinement
            if hasattr(search_engine, "_perform_echo_refinement"):
                print("Re-running echo refinement...")
                search_engine._perform_echo_refinement()
        
        # Evaluate performance
        metrics = evaluate_search_performance(search_engine, test_data)
        config_name = config.get('name', f"Config_{i+1}")
        results[config_name] = metrics
        
        # Restore original configuration
        search_engine.config = original_config.copy()
        if hasattr(search_engine, "_perform_echo_refinement"):
            print("Restoring original configuration...")
            search_engine._perform_echo_refinement()
        
        # Save intermediate results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

def plot_comparison_results(results, output_file="comparison_results.pdf"):
    """
    Plot comparison results
    
    Args:
        results: Dictionary with evaluation results
        output_file: Output file for the plot
        
    Returns:
        None
    """
    # Set up plotting
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics = ['precision', 'recall', 'ndcg', 'mrr']
    titles = ['Precision@k', 'Recall@k', 'NDCG@k', 'Mean Reciprocal Rank@k']
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        # Extract data for this metric
        data = []
        for config_name, config_results in results.items():
            for k_metric, value in config_results.items():
                if k_metric.startswith(f"{metric}@"):
                    k = int(k_metric.split('@')[1])
                    data.append({
                        'Configuration': config_name,
                        'k': k,
                        'Value': value
                    })
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create line plot
        sns.lineplot(x='k', y='Value', hue='Configuration', marker='o', data=df, ax=ax)
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('k')
        ax.set_ylabel(metric.upper())
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    print(f"Comparison plot saved to {output_file}")
