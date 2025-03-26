import numpy as np
import os
import json
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, precision_score, recall_score

class AnalyticsProcessor:
    """
    Handles analytics and metrics calculations for search evaluation
    """
    
    def __init__(self, search_engine):
        """
        Initialize the analytics processor with a search engine
        
        Args:
            search_engine: SeededSphereSearch instance to use for queries
        """
        self.search_engine = search_engine
        self.ground_truth_data = {}
        self.evaluation_results = {}
        
    def load_ground_truth(self, ground_truth_file):
        """
        Load ground truth data from a JSON file
        
        Args:
            ground_truth_file: Path to the JSON file with ground truth
                Format: {
                    "query1": ["doc_id1", "doc_id2", ...],
                    "query2": ["doc_id3", "doc_id4", ...],
                    ...
                }
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(ground_truth_file):
                return False
                
            with open(ground_truth_file, 'r') as f:
                self.ground_truth_data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading ground truth data: {e}")
            return False
    
    def create_ground_truth(self, queries, output_file=None):
        """
        Create ground truth data by running queries and storing results
        
        Args:
            queries: List of queries to run
            output_file: Optional path to save the ground truth data
        
        Returns:
            dict: The ground truth data
        """
        ground_truth = {}
        
        for query in queries:
            results = self.search_engine.search(query, top_k=50)
            ground_truth[query] = [r["id"] for r in results if r["score"] >= 0.7]
            
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(ground_truth, f, indent=2)
                
        self.ground_truth_data = ground_truth
        return ground_truth
    
    def calculate_metrics(self, query, results, ground_truth=None, thresholds=None):
        """
        Calculate metrics based on query results and ground truth
        
        Args:
            query: The query that was run
            results: List of result documents with scores
            ground_truth: Optional list of relevant document IDs for this query
                          If None, will look up in self.ground_truth_data
            thresholds: Optional list of threshold values to evaluate
                        If None, uses np.arange(0.1, 1.0, 0.05)
        
        Returns:
            dict: Dictionary of metrics
        """
        # Get ground truth for this query
        if ground_truth is None:
            if query in self.ground_truth_data:
                ground_truth = self.ground_truth_data[query]
            else:
                # If no ground truth, can't calculate metrics
                return {
                    "error": "No ground truth data available for this query"
                }
        
        if not ground_truth:
            return {
                "error": "Empty ground truth data for this query"
            }
            
        if not results:
            return {
                "error": "No results to evaluate"
            }
            
        # Set default thresholds if not provided
        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.05)
            
        # Extract result IDs and scores
        result_ids = [r["id"] for r in results]
        result_scores = [r["score"] if "score" in r else r["similarity"] for r in results]
        
        # Create binary relevance array (1 for relevant, 0 for not relevant)
        y_true = np.array([1 if doc_id in ground_truth else 0 for doc_id in result_ids])
        
        # Only proceed if we have at least one relevant document
        if sum(y_true) == 0:
            return {
                "error": "No relevant documents in results"
            }
            
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, result_scores)
        
        # Average precision
        ap = average_precision_score(y_true, result_scores)
        
        # Calculate metrics at different thresholds
        threshold_metrics = []
        for threshold in thresholds:
            # Predict relevance based on threshold
            y_pred = np.array([1 if score >= threshold else 0 for score in result_scores])
            
            # Only calculate if we have predictions
            if sum(y_pred) > 0:
                precision_at_t = precision_score(y_true, y_pred, zero_division=0)
                recall_at_t = recall_score(y_true, y_pred, zero_division=0)
                f1_at_t = f1_score(y_true, y_pred, zero_division=0)
                
                threshold_metrics.append({
                    "threshold": threshold,
                    "precision": precision_at_t,
                    "recall": recall_at_t,
                    "f1": f1_at_t
                })
        
        # Calculate precision@k for common k values
        precision_at_k = {}
        for k in [1, 3, 5, 10, 20]:
            if len(result_ids) >= k:
                # Get top k results
                top_k_true = y_true[:k]
                # Calculate precision
                precision_at_k[f"p@{k}"] = np.sum(top_k_true) / k
        
        # Store results
        metrics = {
            "query": query,
            "average_precision": ap,
            "precision_recall_curve": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
            },
            "threshold_metrics": threshold_metrics,
            "precision_at_k": precision_at_k
        }
        
        # Save to the evaluation results
        self.evaluation_results[query] = metrics
        
        return metrics
    
    def generate_precision_recall_curve(self, query, results, ground_truth=None):
        """
        Generate precision-recall curve data for a specific query
        
        Args:
            query: The query string
            results: Search results
            ground_truth: Optional ground truth data
            
        Returns:
            dict: Precision-recall curve data for plotting
        """
        metrics = self.calculate_metrics(query, results, ground_truth)
        
        if "error" in metrics:
            return {
                "success": False,
                "error": metrics["error"]
            }
            
        # Format data for chart.js
        curve_data = {
            "labels": metrics["precision_recall_curve"]["recall"],
            "datasets": [
                {
                    "label": "Precision-Recall Curve",
                    "data": metrics["precision_recall_curve"]["precision"],
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "fill": False
                }
            ],
            "average_precision": metrics["average_precision"]
        }
        
        return {
            "success": True,
            "curve_data": curve_data,
            "metrics": {
                "average_precision": metrics["average_precision"],
                "precision_at_k": metrics["precision_at_k"]
            }
        } 