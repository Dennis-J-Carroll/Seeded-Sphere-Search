"""
Core evaluation metrics for search performance.
"""

import numpy as np
import time
from sklearn.metrics import ndcg_score
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
