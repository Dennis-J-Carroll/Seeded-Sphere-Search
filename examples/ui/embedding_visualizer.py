import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import random

class EmbeddingVisualizer:
    """
    Handles dimensionality reduction for visualizing embeddings
    """
    
    def __init__(self, search_engine):
        """
        Initialize the embedding visualizer with a search engine
        
        Args:
            search_engine: SeededSphereSearch instance to use for embeddings
        """
        self.search_engine = search_engine
        self.cached_visualizations = {}
        
    def get_document_embeddings(self, doc_ids=None):
        """
        Get embeddings for documents
        
        Args:
            doc_ids: Optional list of document IDs to retrieve.
                     If None, get all documents.
        
        Returns:
            tuple: (embeddings, doc_ids, metadata)
        """
        embeddings = []
        metadata = []
        ids = []
        
        # If no doc_ids provided, use all documents
        if doc_ids is None:
            doc_ids = list(self.search_engine.vocabulary.keys())
        
        # Collect embeddings and metadata
        for doc_id in doc_ids:
            if doc_id in self.search_engine.refined_embeddings:
                embeddings.append(self.search_engine.refined_embeddings[doc_id])
                ids.append(doc_id)
                
                # Extract metadata
                doc_info = self.search_engine.vocabulary[doc_id]
                meta = {
                    "id": doc_id,
                    "title": doc_info.get("title", "Untitled"),
                    "tags": doc_info.get("metadata", {}).get("tags", [])
                }
                metadata.append(meta)
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        return embeddings, ids, metadata
    
    def reduce_dimensions(self, embeddings, method="tsne", n_components=2, random_state=42, **kwargs):
        """
        Reduce dimensionality of embeddings for visualization
        
        Args:
            embeddings: Array of embeddings to reduce
            method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
            n_components: Number of components to reduce to (2 or 3)
            random_state: Random state for reproducibility
            **kwargs: Additional parameters for the method
            
        Returns:
            np.array: Reduced embeddings
        """
        if len(embeddings) == 0:
            return np.array([])
            
        if method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=min(30, len(embeddings) - 1) if len(embeddings) > 1 else 1,
                random_state=random_state,
                **kwargs
            )
            reduced = reducer.fit_transform(embeddings)
        
        elif method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced = reducer.fit_transform(embeddings)
            
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                random_state=random_state,
                min_dist=0.1,
                n_neighbors=min(15, len(embeddings) - 1) if len(embeddings) > 1 else 1,
                **kwargs
            )
            reduced = reducer.fit_transform(embeddings)
            
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        return reduced
    
    def visualize_embeddings(self, doc_ids=None, method="tsne", n_components=2, cache_key=None):
        """
        Create a visualization of document embeddings
        
        Args:
            doc_ids: Optional list of document IDs to visualize
            method: Dimensionality reduction method
            n_components: Number of components (2 or 3)
            cache_key: Optional key to cache the results
            
        Returns:
            dict: Visualization data
        """
        # Check cache if a key is provided
        if cache_key and cache_key in self.cached_visualizations:
            return self.cached_visualizations[cache_key]
            
        # Get document embeddings
        embeddings, ids, metadata = self.get_document_embeddings(doc_ids)
        
        if len(embeddings) == 0:
            return {
                "success": False,
                "error": "No embeddings found for the specified documents"
            }
            
        # Reduce dimensions
        try:
            reduced_embeddings = self.reduce_dimensions(embeddings, method, n_components)
        except Exception as e:
            return {
                "success": False,
                "error": f"Error reducing dimensions: {str(e)}"
            }
            
        # Generate colors for visualization
        # Create color map based on document tags if available
        unique_tags = set()
        for meta in metadata:
            unique_tags.update(meta["tags"])
            
        color_map = {}
        colors = []
        
        # Generate distinct colors for each tag
        for i, tag in enumerate(unique_tags):
            r = random.randint(100, 255)
            g = random.randint(100, 255)
            b = random.randint(100, 255)
            color_map[tag] = f"rgb({r}, {g}, {b})"
            
        # Assign colors to each document based on its first tag
        for meta in metadata:
            if meta["tags"]:
                first_tag = meta["tags"][0]
                colors.append({
                    "color": color_map[first_tag],
                    "tag": first_tag
                })
            else:
                colors.append({
                    "color": "rgb(200, 200, 200)",
                    "tag": "untagged"
                })
                
        # Create point data for visualization
        points = []
        for i in range(len(ids)):
            point = {
                "id": ids[i],
                "title": metadata[i]["title"],
                "coords": reduced_embeddings[i].tolist(),
                "color": colors[i]["color"],
                "tag": colors[i]["tag"],
                "tags": metadata[i]["tags"]
            }
            points.append(point)
            
        # Create result object
        result = {
            "success": True,
            "method": method,
            "dimensions": n_components,
            "points": points,
            "color_map": color_map
        }
        
        # Cache the result if a key is provided
        if cache_key:
            self.cached_visualizations[cache_key] = result
            
        return result
    
    def visualize_query_results(self, query, results, method="tsne", n_components=2, include_query=True):
        """
        Visualize query results in the embedding space
        
        Args:
            query: Query string
            results: Search results
            method: Dimensionality reduction method
            n_components: Number of components
            include_query: Include the query point in visualization
            
        Returns:
            dict: Visualization data
        """
        # Get document IDs from results
        doc_ids = [r["id"] for r in results]
        
        # Get document embeddings
        embeddings, ids, metadata = self.get_document_embeddings(doc_ids)
        
        if len(embeddings) == 0:
            return {
                "success": False,
                "error": "No embeddings found for the result documents"
            }
            
        # If include_query, add query embedding
        if include_query:
            try:
                query_embedding = self.search_engine._encode_query(query)
                query_embedding = query_embedding.reshape(1, -1)
                
                # Add query embedding to the array
                embeddings = np.vstack((query_embedding, embeddings))
                
                # Add query metadata
                ids = ["query"] + ids
                metadata = [{"id": "query", "title": f"Query: {query}", "tags": ["query"]}] + metadata
            except Exception as e:
                print(f"Error including query in visualization: {e}")
                # Continue without query if there's an error
                include_query = False
            
        # Reduce dimensions
        try:
            reduced_embeddings = self.reduce_dimensions(embeddings, method, n_components)
        except Exception as e:
            return {
                "success": False,
                "error": f"Error reducing dimensions: {str(e)}"
            }
            
        # Generate colors - query is red, results are blue with color intensity based on score
        colors = []
        
        # Add query point color if included
        if include_query:
            colors.append({
                "color": "rgb(255, 0, 0)",
                "tag": "query"
            })
            
        # Add colors for results with intensity based on score
        for i, result in enumerate(results):
            score = result.get("score", result.get("similarity", 0.5))
            # Scale blue intensity based on score (higher score = more intense blue)
            intensity = int(100 + score * 155)
            colors.append({
                "color": f"rgb(0, 0, {intensity})",
                "tag": "result",
                "score": score
            })
            
        # Create point data for visualization
        points = []
        for i in range(len(ids)):
            point = {
                "id": ids[i],
                "title": metadata[i]["title"],
                "coords": reduced_embeddings[i].tolist(),
                "color": colors[i]["color"],
                "tag": colors[i].get("tag", "result"),
                "isQuery": i == 0 if include_query else False
            }
            
            # Add score for result points
            if i > 0 if include_query else i >= 0:
                result_idx = i - 1 if include_query else i
                if result_idx < len(results):
                    point["score"] = results[result_idx].get("score", 
                                              results[result_idx].get("similarity", 0))
                    
            points.append(point)
            
        # Create result object
        result = {
            "success": True,
            "method": method,
            "dimensions": n_components,
            "points": points,
            "includesQuery": include_query
        }
            
        return result 