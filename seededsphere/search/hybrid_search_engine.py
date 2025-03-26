import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict

class HybridSearchEngine:
    """
    Hybrid search engine that combines multiple search algorithms
    with configurable weights for optimal retrieval.
    """
    
    def __init__(self, config=None):
        """
        Initialize the hybrid search engine
        
        Args:
            config: Configuration dictionary
        """
        # Default configuration
        self.config = {
            "sss_weight": 0.7,           # Weight for Seeded Sphere Search
            "transformer_weight": 0.3,    # Weight for direct transformer similarity
            "bm25_weight": 0.0,           # Weight for BM25 (lexical) search
            "min_score_threshold": 0.1,   # Minimum score to consider
            "ensemble_method": "weighted_sum"  # or "reciprocal_rank"
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Search engines
        self.search_engines = {}
        
        # For caching results
        self.cache = {}
    
    def add_search_engine(self, name, engine):
        """
        Add a search engine to the hybrid ensemble
        
        Args:
            name: Name of the search engine
            engine: Search engine instance with a search() method
        """
        self.search_engines[name] = engine
        return self
    
    def search(self, query, top_k=10, engines=None, weights=None):
        """
        Perform hybrid search across all engines
        
        Args:
            query: Search query
            top_k: Number of top results to return
            engines: List of engine names to use (default: all)
            weights: Dictionary of engine weights (overrides config)
            
        Returns:
            List of search results with combined scores
        """
        if not self.search_engines:
            raise ValueError("No search engines added to hybrid search")
        
        # Determine which engines to use
        if engines is None:
            engines = list(self.search_engines.keys())
        else:
            # Validate engine names
            for engine in engines:
                if engine not in self.search_engines:
                    raise ValueError(f"Unknown search engine: {engine}")
        
        # Get default weights
        if weights is None:
            weights = {
                "sss": self.config["sss_weight"],
                "transformer": self.config["transformer_weight"],
                "bm25": self.config["bm25_weight"]
            }
            # Normalize weights for requested engines
            total = sum(weights[e] for e in engines if e in weights)
            if total > 0:
                weights = {e: weights.get(e, 0.0) / total for e in engines}
        
        # Get results from each engine
        all_results = {}
        for engine_name in engines:
            if engine_name not in self.search_engines:
                continue
                
            engine = self.search_engines[engine_name]
            # Different engines might have different search method signatures
            if engine_name == "sss":
                results = engine.search(query, top_k=top_k * 2)  # Retrieve more for better merging
            elif engine_name == "transformer":
                results = engine.search(query, top_k=top_k * 2)
            elif engine_name == "bm25":
                results = engine.search(query, top_k=top_k * 2)
            else:
                results = engine.search(query, top_k=top_k * 2)
            
            all_results[engine_name] = results
        
        # Combine results based on ensemble method
        if self.config["ensemble_method"] == "weighted_sum":
            combined = self._combine_weighted_sum(all_results, weights, top_k)
        elif self.config["ensemble_method"] == "reciprocal_rank":
            combined = self._combine_reciprocal_rank(all_results, weights, top_k)
        else:
            raise ValueError(f"Unknown ensemble method: {self.config['ensemble_method']}")
        
        return combined
    
    def _combine_weighted_sum(self, all_results, weights, top_k):
        """
        Combine results using weighted sum method
        
        Args:
            all_results: Dictionary of results from each engine
            weights: Dictionary of weights for each engine
            top_k: Number of top results to return
            
        Returns:
            List of combined results
        """
        # Collect all document IDs
        all_doc_ids = set()
        for engine_name, results in all_results.items():
            for item in results:
                all_doc_ids.add(item["id"])
        
        # Calculate combined scores
        combined_scores = {}
        for doc_id in all_doc_ids:
            combined_score = 0.0
            doc_info = None
            
            for engine_name, results in all_results.items():
                if engine_name not in weights or weights[engine_name] <= 0:
                    continue
                    
                # Find document in this engine's results
                for item in results:
                    if item["id"] == doc_id:
                        combined_score += item["score"] * weights[engine_name]
                        doc_info = item  # Keep document info
                        break
            
            if combined_score >= self.config["min_score_threshold"] and doc_info:
                combined_scores[doc_id] = {
                    "id": doc_id,
                    "title": doc_info.get("title", ""),
                    "score": combined_score,
                    "engine_scores": {e: 0.0 for e in all_results.keys()}
                }
                
                # Add individual engine scores
                for engine_name, results in all_results.items():
                    for item in results:
                        if item["id"] == doc_id:
                            combined_scores[doc_id]["engine_scores"][engine_name] = item["score"]
                            break
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def _combine_reciprocal_rank(self, all_results, weights, top_k):
        """
        Combine results using reciprocal rank fusion
        
        Args:
            all_results: Dictionary of results from each engine
            weights: Dictionary of weights for each engine
            top_k: Number of top results to return
            
        Returns:
            List of combined results
        """
        # Collect all document IDs and their ranks
        doc_ranks = defaultdict(dict)
        doc_info = {}
        
        for engine_name, results in all_results.items():
            for rank, item in enumerate(results):
                doc_id = item["id"]
                doc_ranks[doc_id][engine_name] = rank + 1  # 1-based rank
                if doc_id not in doc_info:
                    doc_info[doc_id] = item
        
        # Calculate combined scores using reciprocal rank fusion
        k = 60  # Constant from RRF paper, prevents high weights for low ranks
        combined_scores = {}
        
        for doc_id, ranks in doc_ranks.items():
            rrf_score = 0.0
            
            for engine_name, rank in ranks.items():
                if engine_name in weights and weights[engine_name] > 0:
                    # RRF formula: 1/(k + rank)
                    rrf_score += weights[engine_name] * (1.0 / (k + rank))
            
            combined_scores[doc_id] = {
                "id": doc_id,
                "title": doc_info[doc_id].get("title", ""),
                "score": rrf_score,
                "engine_scores": {e: 0.0 for e in all_results.keys()}
            }
            
            # Add individual engine scores
            for engine_name, results in all_results.items():
                for item in results:
                    if item["id"] == doc_id:
                        combined_scores[doc_id]["engine_scores"][engine_name] = item["score"]
                        break
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def explain_search_results(self, result):
        """
        Explain why a document was retrieved
        
        Args:
            result: A search result item
            
        Returns:
            Explanation string
        """
        explanation = []
        explanation.append(f"Document: {result['title']} (ID: {result['id']})")
        explanation.append(f"Combined Score: {result['score']:.4f}")
        
        if 'engine_scores' in result:
            explanation.append("Engine Contributions:")
            sorted_engines = sorted(
                result['engine_scores'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for engine_name, score in sorted_engines:
                if score > 0:
                    explanation.append(f"  - {engine_name}: {score:.4f}")
        
        return "\n".join(explanation)
    
    def analyze_query(self, query):
        """
        Analyze a query to determine which search engines would perform best
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary with analysis results
        """
        import re
        
        analysis = {
            "query_length": len(query),
            "word_count": len(query.split()),
            "contains_technical_terms": False,
            "contains_quotes": '"' in query,
            "is_natural_language": False,
            "recommended_engines": [],
            "recommended_weights": {}
        }
        
        # Check for technical terms (simplified)
        technical_pattern = r'\b(?:algorithm|function|class|method|api|interface|code|syntax|error)\b'
        if re.search(technical_pattern, query.lower()):
            analysis["contains_technical_terms"] = True
        
        # Check if it looks like natural language
        if analysis["word_count"] > 3 and not analysis["contains_quotes"]:
            analysis["is_natural_language"] = True
        
        # Make recommendations
        if analysis["is_natural_language"]:
            # Transformer and SSS good for natural language
            analysis["recommended_engines"] = ["sss", "transformer"]
            analysis["recommended_weights"] = {"sss": 0.6, "transformer": 0.4}
        elif analysis["contains_quotes"] or analysis["contains_technical_terms"]:
            # BM25 good for exact matches and technical terms
            if "bm25" in self.search_engines:
                analysis["recommended_engines"] = ["bm25", "sss"]
                analysis["recommended_weights"] = {"bm25": 0.6, "sss": 0.4}
            else:
                analysis["recommended_engines"] = ["sss"]
                analysis["recommended_weights"] = {"sss": 1.0}
        else:
            # Default to all engines
            analysis["recommended_engines"] = list(self.search_engines.keys())
            # Equal weights
            weights = {e: 1.0 / len(self.search_engines) for e in self.search_engines}
            analysis["recommended_weights"] = weights
        
        return analysis
        
    def optimize_weights(self, queries, relevant_docs, metric="ndcg"):
        """
        Optimize engine weights using training queries and relevance judgments
        
        Args:
            queries: List of query strings
            relevant_docs: Dictionary mapping queries to lists of relevant document IDs
            metric: Evaluation metric to optimize ('ndcg', 'precision', 'recall')
            
        Returns:
            Optimized weights
        """
        from scipy.optimize import minimize_scalar
        
        def objective(sss_weight):
            # Convert to all weights (sss and transformer/other)
            weights = {
                "sss": sss_weight,
                "transformer": 1.0 - sss_weight,
                "bm25": 0.0  # Fixed at 0 for this optimization
            }
            
            metrics = []
            for query in queries:
                results = self.search(query, weights=weights)
                retrieved_ids = [result["id"] for result in results]
                relevant_ids = relevant_docs.get(query, [])
                
                if metric == "precision":
                    # Precision@k
                    if len(retrieved_ids) > 0:
                        precision = len(set(retrieved_ids) & set(relevant_ids)) / len(retrieved_ids)
                        metrics.append(precision)
                elif metric == "recall":
                    # Recall@k
                    if len(relevant_ids) > 0:
                        recall = len(set(retrieved_ids) & set(relevant_ids)) / len(relevant_ids)
                        metrics.append(recall)
                elif metric == "ndcg":
                    # NDCG@k (simplified)
                    import numpy as np
                    from sklearn.metrics import ndcg_score
                    
                    y_true = np.zeros(len(retrieved_ids))
                    for i, doc_id in enumerate(retrieved_ids):
                        if doc_id in relevant_ids:
                            y_true[i] = 1
                    
                    y_score = np.array([result["score"] for result in results])
                    
                    if len(y_true) > 0 and len(y_score) > 0 and np.sum(y_true) > 0:
                        ndcg = ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1))
                        metrics.append(ndcg)
            
            # Return negative mean metric (for minimization)
            return -np.mean(metrics) if metrics else -0.0
        
        # Find optimal weight between 0 and 1
        result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        
        optimized_sss_weight = result.x
        
        return {
            "sss": optimized_sss_weight,
            "transformer": 1.0 - optimized_sss_weight,
            "bm25": 0.0
        }
