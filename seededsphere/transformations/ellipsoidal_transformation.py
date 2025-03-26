import numpy as np

class EllipsoidalTransformation:
    def __init__(self, dimensions):
        """
        Initialize the ellipsoidal transformation
        
        Args:
            dimensions: Number of dimensions in the embedding
        """
        self.dimensions = dimensions
        self.weights = np.ones(dimensions)  # Initialize with uniform weights
    
    def train_weights(self, query_embeddings, relevant_doc_embeddings, 
                     non_relevant_doc_embeddings, learning_rate=0.01, max_epochs=100):
        """
        Train weights using contrastive loss
        
        Args:
            query_embeddings: Array of query embeddings
            relevant_doc_embeddings: List of arrays containing relevant doc embeddings for each query
            non_relevant_doc_embeddings: List of arrays containing non-relevant doc embeddings for each query
            learning_rate: Learning rate for gradient descent
            max_epochs: Maximum number of training epochs
        """
        print("Training ellipsoidal weights...")
        
        # Regularization strength
        reg_lambda = 0.01
        
        for epoch in range(max_epochs):
            total_loss = 0
            weight_updates = np.zeros(self.dimensions)
            
            # Process each query
            for i, query in enumerate(query_embeddings):
                relevant_docs = relevant_doc_embeddings[i]
                non_relevant_docs = non_relevant_doc_embeddings[i]
                
                # Skip if no relevant or non-relevant docs
                if len(relevant_docs) == 0 or len(non_relevant_docs) == 0:
                    continue
                
                # Calculate weighted query
                weighted_query = query * self.weights
                
                # Calculate distances to relevant and non-relevant docs
                relevant_distances = []
                for doc in relevant_docs:
                    dot_product = np.dot(weighted_query, doc)
                    dot_product = max(-1.0, min(1.0, dot_product))
                    distance = np.arccos(dot_product)
                    relevant_distances.append(distance)
                
                non_relevant_distances = []
                for doc in non_relevant_docs:
                    dot_product = np.dot(weighted_query, doc)
                    dot_product = max(-1.0, min(1.0, dot_product))
                    distance = np.arccos(dot_product)
                    non_relevant_distances.append(distance)
                
                # Calculate average distances
                avg_relevant_dist = np.mean(relevant_distances)
                avg_non_relevant_dist = np.mean(non_relevant_distances)
                
                # Hinge loss with margin
                margin = 0.2
                loss = max(0, margin + avg_relevant_dist - avg_non_relevant_dist)
                total_loss += loss
                
                # Skip gradient update if loss is zero
                if loss == 0:
                    continue
                
                # Calculate gradient
                for doc in relevant_docs:
                    # We want to decrease distance, so negative gradient
                    gradient = -self._distance_gradient(weighted_query, doc, query)
                    weight_updates += gradient
                
                for doc in non_relevant_docs:
                    # We want to increase distance, so positive gradient
                    gradient = self._distance_gradient(weighted_query, doc, query)
                    weight_updates += gradient
            
            # Add regularization gradient (L2)
            weight_updates += reg_lambda * 2 * self.weights
            
            # Update weights with learning rate
            self.weights -= learning_rate * weight_updates
            
            # Ensure weights are positive
            self.weights = np.maximum(0.01, self.weights)
            
            # Normalize weights to prevent collapse
            self.weights = self.weights / np.mean(self.weights)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{max_epochs}, Loss: {total_loss:.4f}")
        
        print("Training complete.")
        
        # Print the dimensions with highest and lowest weights
        top_dims = np.argsort(self.weights)[-10:]
        bottom_dims = np.argsort(self.weights)[:10]
        print(f"Top weighted dimensions: {top_dims}, weights: {self.weights[top_dims]}")
        print(f"Bottom weighted dimensions: {bottom_dims}, weights: {self.weights[bottom_dims]}")
        
        return self.weights
    
    def _distance_gradient(self, weighted_query, doc, original_query):
        """Calculate gradient of angular distance with respect to weights"""
        dot_product = np.dot(weighted_query, doc)
        dot_product = max(-1.0, min(1.0, dot_product))
        
        # Derivative of arccos(x) is -1/sqrt(1-x²)
        arccos_derivative = -1.0 / np.sqrt(1.0 - dot_product**2 + 1e-10)
        
        # Derivative of dot product with respect to weights
        dot_derivative = original_query * doc
        
        return arccos_derivative * dot_derivative
    
    def transform_query(self, query_embedding):
        """Apply ellipsoidal transformation to a query embedding"""
        transformed = query_embedding * self.weights
        return self._normalize_vector(transformed)
    
    def transform_document(self, doc_embedding):
        """Apply ellipsoidal transformation to a document embedding"""
        transformed = doc_embedding * self.weights
        return self._normalize_vector(transformed)
    
    def _normalize_vector(self, vector):
        """Normalize a vector to unit length"""
        norm = np.sqrt(np.sum(vector ** 2))
        if norm < 1e-8:
            return np.ones(len(vector)) / np.sqrt(len(vector))
        return vector / norm
    
    def save(self, filename):
        """Save the transformation weights to a file"""
        np.save(filename, self.weights)
    
    def load(self, filename):
        """Load the transformation weights from a file"""
        self.weights = np.load(filename)
        return self
