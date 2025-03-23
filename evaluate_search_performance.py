
import numpy as np 


def evaluate_search_performance(search_engine, test_queries, ground_truth, top_k=10):
    """
    Evaluate search performance using precision, recall, and NDCG metrics
    
    Args:
        search_engine: Search engine instance
        test_queries: List of test query strings
        ground_truth: Dictionary mapping queries to relevant document IDs
        top_k: Number of top results to evaluate
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    from sklearn.metrics import ndcg_score
    
    precision_at_k = []
    recall_at_k = []
    ndcg_at_k = []
    
    for query in test_queries:
        # Get search results
        results = search_engine.search(query, top_k=top_k)
        retrieved_ids = [result["id"] for result in results]
        
        # Get relevant document IDs for this query
        relevant_ids = ground_truth.get(query, [])
        
        # Skip if no relevant documents
        if len(relevant_ids) == 0:
            continue
        
        # Calculate precision@k
        relevant_retrieved = len(set(retrieved_ids) & set(relevant_ids))
        precision = relevant_retrieved / min(top_k, len(retrieved_ids))
        precision_at_k.append(precision)
        
        # Calculate recall@k
        recall = relevant_retrieved / len(relevant_ids)
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
        ndcg = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1))
        ndcg_at_k.append(ndcg)
    
    # Calculate average metrics
    avg_precision = np.mean(precision_at_k) if precision_at_k else 0
    avg_recall = np.mean(recall_at_k) if recall_at_k else 0
    avg_ndcg = np.mean(ndcg_at_k) if ndcg_at_k else 0
    
    return {
        f"precision@{top_k}": avg_precision,
        f"recall@{top_k}": avg_recall,
        f"ndcg@{top_k}": avg_ndcg
    }