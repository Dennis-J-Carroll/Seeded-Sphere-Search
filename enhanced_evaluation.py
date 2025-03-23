import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from sklearn.metrics import ndcg_score, precision_recall_curve, average_precision_score
from tqdm import tqdm

def evaluate_search_performance(search_engine, test_data, top_k=[5, 10, 20, 50], echo_config=None):
    """
    Comprehensive evaluation of search engine performance
    
    Args:
        search_engine: Search engine instance
        test_data: Dictionary with 'queries' and 'relevance' keys
                  - 'queries': list of query strings
                  - 'relevance': dict mapping query to list of relevant document IDs
        top_k: List of k values for evaluation
        echo_config: Optional configuration for echo layers to evaluate
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    metrics = {}
    
    # Set echo layers if provided
    original_config = None
    if echo_config:
        original_config = {
            "min_echo_layers": search_engine.config["min_echo_layers"],
            "max_echo_layers": search_engine.config["max_echo_layers"]
        }
        search_engine.config["min_echo_layers"] = echo_config["min_echo_layers"]
        search_engine.config["max_echo_layers"] = echo_config["max_echo_layers"]
        # Re-run echo refinement if needed
        if hasattr(search_engine, "_perform_echo_refinement"):
            print(f"Re-running echo refinement with layers {echo_config['min_echo_layers']}-{echo_config['max_echo_layers']}...")
            search_engine._perform_echo_refinement()
    
    # Evaluate for each k
    for k in top_k:
        print(f"Evaluating performance at k={k}...")
        
        precision_at_k = []
        recall_at_k = []
        ndcg_at_k = []
        mean_reciprocal_rank = []
        query_times = []
        
        for i, query in enumerate(tqdm(test_data["queries"])):
            # Time search operation
            start_time = time.time()
            results = search_engine.search(query, top_k=k)
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            retrieved_ids = [result["id"] for result in results]
            relevant_ids = test_data["relevance"].get(query, [])
            
            # Skip if no relevant documents
            if len(relevant_ids) == 0:
                continue
            
            # Calculate precision@k
            relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
            precision = relevant_retrieved / min(k, len(retrieved_ids)) if retrieved_ids else 0
            precision_at_k.append(precision)
            
            # Calculate recall@k
            recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
            recall_at_k.append(recall)
            
            # Calculate NDCG@k
            # Create binary relevance scores for retrieved documents
            y_true = np.zeros(len(retrieved_ids))
            for i, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    y_true[i] = 1
            
            # Create scores for retrieved documents
            y_score = np.array([result["score"] for result in results])
            
            # Calculate NDCG
            if len(y_true) > 0 and len(y_score) > 0 and np.sum(y_true) > 0:
                try:
                    ndcg = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1))
                    ndcg_at_k.append(ndcg)
                except:
                    # Skip if NDCG calculation fails
                    pass
            
            # Calculate Mean Reciprocal Rank (MRR)
            reciprocal_rank = 0
            for rank, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_ids:
                    reciprocal_rank = 1.0 / (rank + 1)
                    break
            mean_reciprocal_rank.append(reciprocal_rank)
        
        # Calculate average metrics
        metrics[f"precision@{k}"] = np.mean(precision_at_k) if precision_at_k else 0
        metrics[f"recall@{k}"] = np.mean(recall_at_k) if recall_at_k else 0
        metrics[f"ndcg@{k}"] = np.mean(ndcg_at_k) if ndcg_at_k else 0
        metrics[f"mrr@{k}"] = np.mean(mean_reciprocal_rank) if mean_reciprocal_rank else 0
        metrics[f"query_time@{k}"] = np.mean(query_times)
    
    # Restore original config if modified
    if original_config:
        search_engine.config["min_echo_layers"] = original_config["min_echo_layers"]
        search_engine.config["max_echo_layers"] = original_config["max_echo_layers"]
        # Re-run echo refinement if needed
        if hasattr(search_engine, "_perform_echo_refinement"):
            print("Restoring original echo configuration...")
            search_engine._perform_echo_refinement()
    
    return metrics

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
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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

def generate_test_data(search_engine, num_queries=10, num_relevant=5, query_prefix="test_query_"):
    """
    Generate synthetic test data for evaluation
    
    Args:
        search_engine: Search engine instance
        num_queries: Number of test queries to generate
        num_relevant: Number of relevant documents per query
        query_prefix: Prefix for query IDs
        
    Returns:
        dict: Dictionary with 'queries' and 'relevance' keys
    """
    # Generate random queries based on document content
    import random
    
    test_data = {
        "queries": [],
        "relevance": {}
    }
    
    # Get random documents
    doc_ids = list(search_engine.vocabulary.keys())
    random.shuffle(doc_ids)
    
    for i in range(min(num_queries, len(doc_ids))):
        doc_id = doc_ids[i]
        
        # Generate query from document title and content
        title = search_engine.vocabulary[doc_id]["title"]
        content = search_engine.vocabulary[doc_id]["content"]
        
        # Extract some phrases from the document
        sentences = content.split('.')[:5]  # Take first 5 sentences
        random.shuffle(sentences)
        
        query = title if random.random() < 0.5 else sentences[0]
        # Truncate to reasonable length
        query = query[:100].strip()
        
        test_data["queries"].append(query)
        
        # Find relevant documents (the source document and some similar ones)
        relevant_docs = [doc_id]
        
        # Find similar documents based on relationship matrix
        neighbors = [(neighbor, score) for neighbor, score in search_engine.relationships[doc_id].items()]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Add top similar documents
        for neighbor, _ in neighbors[:num_relevant - 1]:
            if neighbor not in relevant_docs:
                relevant_docs.append(neighbor)
        
        test_data["relevance"][query] = relevant_docs
    
    return test_data

if __name__ == "__main__":
    import pickle
    from SeededSphereSearch import SeededSphereSearch
    
    # Load search engine
    print("Loading search engine...")
    with open('sss_model.pkl', 'rb') as f:
        search_engine = pickle.load(f)
    
    # Generate test data
    print("Generating test data...")
    test_data = generate_test_data(search_engine, num_queries=20, num_relevant=5)
    
    # Evaluate performance
    print("Evaluating search performance...")
    metrics = evaluate_search_performance(search_engine, test_data)
    
    # Print results
    print("\nEvaluation Results:")
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
    
    results = compare_configurations(search_engine, test_data, configurations)
    
    # Plot results
    print("Plotting comparison results...")
    plot_comparison_results(results)
    
    # Evaluate echo layer impact
    print("\nEvaluating echo layer impact...")
    echo_results = evaluate_echo_layer_impact(search_engine, test_data)
    
    print("Evaluation complete.")
