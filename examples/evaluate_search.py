"""
Example script demonstrating how to use the SeededSphere evaluation module.
"""

import pickle
from seededsphere import SeededSphereSearch
from seededsphere.evaluation import (
    evaluate_search_performance,
    compare_configurations,
    plot_comparison_results,
    evaluate_echo_layer_impact,
    generate_test_data
)

def main():
    # Load a pre-trained search engine
    print("Loading search engine...")
    with open('sss_model.pkl', 'rb') as f:
        search_engine = pickle.load(f)
    
    # Generate synthetic test data
    print("\nGenerating test data...")
    test_data = generate_test_data(
        search_engine,
        num_queries=20,
        num_relevant=5
    )
    
    # Basic performance evaluation
    print("\nEvaluating baseline performance...")
    metrics = evaluate_search_performance(search_engine, test_data)
    print("\nBaseline Evaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Compare different configurations
    print("\nComparing different configurations...")
    configurations = [
        {"name": "Baseline", "delta": 0.1625, "alpha": 0.5},
        {"name": "Higher Delta", "delta": 0.25, "alpha": 0.5},
        {"name": "Higher Alpha", "delta": 0.1625, "alpha": 1.0},
        {"name": "More Echo", "min_echo_layers": 2, "max_echo_layers": 8},
    ]
    
    comparison_results = compare_configurations(
        search_engine,
        test_data,
        configurations,
        output_file="results/configuration_comparison.json"
    )
    
    # Plot comparison results
    print("\nPlotting comparison results...")
    plot_comparison_results(
        comparison_results,
        output_file="results/configuration_comparison.pdf"
    )
    
    # Evaluate echo layer impact
    print("\nAnalyzing echo layer impact...")
    echo_results = evaluate_echo_layer_impact(
        search_engine,
        test_data,
        min_range=[1, 2, 3],
        max_range=[3, 5, 10],
        output_file="results/echo_layer_impact.pdf"
    )
    
    print("\nEvaluation complete! Results saved to results/ directory.")

if __name__ == "__main__":
    main()
