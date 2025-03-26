"""
Tools for analyzing the impact of echo layer configurations on search performance.

This module provides functions for:
- Evaluating echo layer parameter impact on search performance
- Visualizing echo refinement process
- Analyzing embedding evolution during echo refinements
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os

from .metrics import evaluate_search_performance
from .comparison import plot_comparison_results
from ..core.echo_mechanism import EchoLayer, SparseEchoLayer, analyze_embedding_changes, plot_embedding_evolution

def evaluate_echo_layer_impact(search_engine, test_data, min_range=[1, 2, 3], max_range=[3, 5, 10], output_file="echo_layer_impact.pdf"):
    """
    Evaluate the impact of echo layer configurations
    
    Args:
        search_engine: Search engine instance
        test_data: Test data for evaluation
        min_range: Range of min_echo_layers to evaluate
        max_range: Range of max_echo_layers to evaluate
        output_file: Output file for the plot
        
    Returns:
        dict: Evaluation results for each configuration
    """
    results = {}
    
    # Test each combination of min and max echo layers
    for min_echo in min_range:
        for max_echo in max_range:
            if min_echo <= max_echo:
                echo_config = {
                    "min_echo_layers": min_echo,
                    "max_echo_layers": max_echo
                }
                
                config_name = f"Echo_{min_echo}_{max_echo}"
                print(f"Evaluating echo configuration: {config_name}")
                
                # Evaluate performance with this configuration
                metrics = evaluate_search_performance(
                    search_engine, 
                    test_data,
                    echo_config=echo_config
                )
                
                results[config_name] = metrics
    
    # Plot the results
    plot_comparison_results(results, output_file)
    
    return results


def visualize_echo_refinement(search_engine, doc_ids, save_dir="results", experiment_name="echo_refinement"):
    """
    Visualize the echo refinement process for selected documents
    
    Args:
        search_engine: Search engine instance
        doc_ids: List of document IDs to visualize
        save_dir: Directory to save visualization results
        experiment_name: Prefix for saved files
        
    Returns:
        Path to saved visualization
    """
    # Ensure we have valid document IDs
    valid_doc_ids = [doc_id for doc_id in doc_ids if doc_id in search_engine.vocabulary]
    if len(valid_doc_ids) == 0:
        print("No valid document IDs provided for visualization")
        return None
    
    # Get document titles for labeling
    doc_titles = [search_engine.vocabulary[doc_id].get("title", doc_id)[:30] for doc_id in valid_doc_ids]
    
    # Get initial and refined embeddings
    initial_embeddings = np.stack([search_engine.embeddings[doc_id] for doc_id in valid_doc_ids])
    refined_embeddings = np.stack([search_engine.refined_embeddings[doc_id] for doc_id in valid_doc_ids])
    
    # Calculate similarity matrices
    initial_sim = cosine_similarity(initial_embeddings)
    refined_sim = cosine_similarity(refined_embeddings)
    
    # Estimate displacement based on difference between initial and refined
    # This is a simplified approach since we don't have the actual evolution steps
    displacement_matrix = np.zeros((1, len(valid_doc_ids)))
    displacement_matrix[0] = np.linalg.norm(refined_embeddings - initial_embeddings, axis=1)
    
    # Plot and save the results
    fig_path = plot_embedding_evolution(
        doc_titles,
        initial_sim,
        refined_sim,
        displacement_matrix,
        save_dir=save_dir,
        experiment_name=experiment_name
    )
    
    return fig_path


def perform_echo_layer_analysis(search_engine, doc_ids, delta=0.1625, freq_scale=0.325, 
                              min_layers=1, max_layers=5, save_dir="results", 
                              experiment_name="echo_analysis", track_mode="detailed"):
    """
    Perform detailed analysis of echo layer refinement using the enhanced echo layer
    
    Args:
        search_engine: Search engine instance
        doc_ids: List of document IDs to analyze
        delta: Step size for echo refinement
        freq_scale: Frequency scaling factor
        min_layers: Minimum echo layers
        max_layers: Maximum echo layers
        save_dir: Directory to save results
        experiment_name: Prefix for saved files
        track_mode: Evolution tracking detail level ('basic', 'detailed', or 'tokenlevel')
        
    Returns:
        Dictionary containing analysis results
    """
    # Ensure we have valid document IDs
    valid_doc_ids = [doc_id for doc_id in doc_ids if doc_id in search_engine.vocabulary]
    if len(valid_doc_ids) == 0:
        print("No valid document IDs provided for analysis")
        return None
    
    # Create EchoLayer instance
    echo_layer = EchoLayer(
        delta=delta, 
        freq_scale=freq_scale,
        min_layers=min_layers,
        max_layers=max_layers
    )
    
    # Get document titles for labeling
    doc_titles = [search_engine.vocabulary[doc_id].get("title", doc_id)[:30] for doc_id in valid_doc_ids]
    
    # Get initial embeddings and create relationship matrix
    initial_embeddings = np.stack([search_engine.embeddings[doc_id] for doc_id in valid_doc_ids])
    
    # Create relationship matrix based on search engine's stored relationships
    relationship_matrix = np.zeros((len(valid_doc_ids), len(valid_doc_ids)))
    for i, doc_id1 in enumerate(valid_doc_ids):
        for j, doc_id2 in enumerate(valid_doc_ids):
            if doc_id2 in search_engine.relationships[doc_id1]:
                relationship_matrix[i, j] = search_engine.relationships[doc_id1][doc_id2]
    
    # Normalize relationship matrix
    row_sums = relationship_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    relationship_matrix = relationship_matrix / row_sums
    
    # Calculate token frequencies
    frequencies = np.array([search_engine.word_frequencies.get(doc_id, 0.5) for doc_id in valid_doc_ids])
    frequencies = frequencies.reshape(1, -1)  # [1, num_docs]
    
    # Convert to PyTorch tensors
    embeddings_tensor = torch.tensor(initial_embeddings).unsqueeze(0).float()  # [1, num_docs, dim]
    relationship_tensor = torch.tensor(relationship_matrix).unsqueeze(0).float()  # [1, num_docs, num_docs]
    frequencies_tensor = torch.tensor(frequencies).float()
    
    # Apply echo refinement with detailed tracking
    refined_embeddings, evolution = echo_layer(
        embeddings_tensor, 
        frequencies_tensor, 
        relationship_tensor,
        track_mode=track_mode
    )
    
    # Analyze embedding changes
    analysis = analyze_embedding_changes(doc_titles, evolution)
    
    # Calculate similarity matrices before and after
    initial_sim = cosine_similarity(initial_embeddings)
    refined_sim = cosine_similarity(refined_embeddings[0].detach().numpy())
    
    # Plot and save results
    fig_path = plot_embedding_evolution(
        doc_titles,
        initial_sim,
        refined_sim,
        analysis['displacement_matrix'],
        save_dir=save_dir,
        experiment_name=experiment_name
    )
    
    # Return analysis results
    return {
        'doc_ids': valid_doc_ids,
        'doc_titles': doc_titles,
        'initial_embeddings': initial_embeddings,
        'refined_embeddings': refined_embeddings[0].detach().numpy(),
        'initial_similarity': initial_sim,
        'refined_similarity': refined_sim,
        'evolution': evolution,
        'analysis': analysis,
        'figure_path': fig_path
    }
