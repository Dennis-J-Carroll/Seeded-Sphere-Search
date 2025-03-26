import numpy as np
import torch

class HyperbolicTransformation:
    """
    Implementation of hyperbolic transformations for hierarchical data.
    Uses the Poincaré ball model for hyperbolic embeddings.
    """
    
    def __init__(self, dimensions, curvature=-1.0):
        """
        Initialize the hyperbolic transformation
        
        Args:
            dimensions: Number of dimensions in the embedding
            curvature: Curvature parameter of the hyperbolic space (negative for hyperbolic)
        """
        self.dimensions = dimensions
        self.curvature = curvature
        self.c = abs(curvature)  # Absolute curvature value
    
    def _project_to_ball(self, x, eps=1e-6):
        """Project a vector onto the Poincaré ball"""
        norm = np.linalg.norm(x)
        if norm >= 1.0 - eps:
            # If outside or close to boundary, project back
            return x * (1.0 - eps) / (norm + eps)
        return x
    
    def _mobius_addition(self, x, y):
        """
        Möbius addition in the Poincaré ball model
        
        x ⊕ y = (x + y(1 + 2c⟨x,y⟩ + c|y|²))/(1 + 2c⟨x,y⟩ + c²|x|²|y|²)
        """
        c = self.c
        x_dot_y = np.dot(x, y)
        x_norm_sq = np.sum(x**2)
        y_norm_sq = np.sum(y**2)
        
        denominator = 1 + 2 * c * x_dot_y + c**2 * x_norm_sq * y_norm_sq
        numerator = x + y * (1 + 2 * c * x_dot_y + c * y_norm_sq)
        
        return numerator / denominator
    
    def _hyperbolic_distance(self, x, y):
        """
        Compute the hyperbolic distance between two points
        
        d_H(x,y) = (2/√|c|) * arctanh(√|c| * |(-x) ⊕ y|)
        """
        c = self.c
        neg_x = -x
        diff = self._mobius_addition(neg_x, y)
        diff_norm = np.linalg.norm(diff)
        
        # Avoid numerical issues
        diff_norm = min(diff_norm, 1.0 - 1e-10)
        
        return (2.0 / np.sqrt(c)) * np.arctanh(np.sqrt(c) * diff_norm)
    
    def transform_embeddings(self, embeddings):
        """
        Transform Euclidean embeddings to hyperbolic space
        
        Args:
            embeddings: Dictionary of embeddings
            
        Returns:
            Dictionary of transformed embeddings
        """
        transformed = {}
        for doc_id, embedding in embeddings.items():
            # Rescale embedding to fit within Poincaré ball
            norm = np.linalg.norm(embedding)
            # Use exponential decay for smooth mapping to hyperbolic space
            # This preserves relative distances better than simple scaling
            scale = 2.0 / (1.0 + np.exp(norm)) - 1.0
            hyperbolic_embedding = embedding * scale
            
            # Ensure it's inside the ball
            hyperbolic_embedding = self._project_to_ball(hyperbolic_embedding)
            
            transformed[doc_id] = hyperbolic_embedding
            
        return transformed
    
    def calculate_similarity_score(self, x, y, alpha=1.0):
        """
        Calculate similarity score between two hyperbolic embeddings
        
        Args:
            x, y: Embeddings in hyperbolic space
            alpha: Temperature parameter for scoring
            
        Returns:
            Similarity score
        """
        distance = self._hyperbolic_distance(x, y)
        return np.exp(-alpha * distance)
    
    def train_embeddings(self, embeddings, hierarchical_relationships, learning_rate=0.01, iterations=100):
        """
        Refine embeddings based on hierarchical relationships
        
        Args:
            embeddings: Initial embeddings dictionary
            hierarchical_relationships: List of (parent, child, weight) tuples
            learning_rate: Learning rate for gradient updates
            iterations: Number of training iterations
            
        Returns:
            Dictionary of trained hyperbolic embeddings
        """
        # Initialize hyperbolic embeddings
        hyperbolic_embeddings = self.transform_embeddings(embeddings)
        doc_ids = list(hyperbolic_embeddings.keys())
        
        for iteration in range(iterations):
            total_loss = 0.0
            
            # Process each hierarchical relationship
            for parent, child, weight in hierarchical_relationships:
                if parent not in hyperbolic_embeddings or child not in hyperbolic_embeddings:
                    continue
                
                parent_embed = hyperbolic_embeddings[parent]
                child_embed = hyperbolic_embeddings[child]
                
                # Calculate current distance
                current_distance = self._hyperbolic_distance(parent_embed, child_embed)
                
                # Loss is the weighted squared difference between actual and desired distance
                # (closer nodes should have smaller distances)
                desired_distance = 1.0 / (1.0 + weight)
                loss = weight * (current_distance - desired_distance) ** 2
                total_loss += loss
                
                # Gradient step: move embeddings closer or further based on relationship
                if current_distance > desired_distance:
                    # Move child closer to parent
                    direction = (parent_embed - child_embed) / (np.linalg.norm(parent_embed - child_embed) + 1e-10)
                    child_embed += learning_rate * weight * direction
                    
                    # Move parent closer to child (with smaller step)
                    parent_embed -= 0.5 * learning_rate * weight * direction
                else:
                    # Move apart if too close
                    direction = (child_embed - parent_embed) / (np.linalg.norm(child_embed - parent_embed) + 1e-10)
                    child_embed += learning_rate * weight * direction
                    parent_embed -= 0.5 * learning_rate * weight * direction
                
                # Project back to Poincaré ball
                hyperbolic_embeddings[parent] = self._project_to_ball(parent_embed)
                hyperbolic_embeddings[child] = self._project_to_ball(child_embed)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}/{iterations}, Loss: {total_loss:.4f}")
        
        return hyperbolic_embeddings
    
    def visualize_embeddings(self, embeddings, labels=None, filename="hyperbolic_embeddings.png"):
        """
        Visualize embeddings in 2D Poincaré disk
        
        Args:
            embeddings: Dictionary of embeddings
            labels: Optional dictionary mapping IDs to labels
            filename: Output filename for the visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        
        # Extract first two dimensions for visualization
        coords = {}
        for doc_id, embed in embeddings.items():
            if len(embed) >= 2:
                coords[doc_id] = embed[:2]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw Poincaré disk boundary
        boundary = Circle((0, 0), 1, fill=False, color='black')
        ax.add_patch(boundary)
        
        # Plot points
        for doc_id, (x, y) in coords.items():
            label = labels.get(doc_id, '') if labels else ''
            ax.plot(x, y, 'o', markersize=5)
            ax.text(x, y, label, fontsize=8)
        
        # Set equal aspect and limits
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        
        # Add title and labels
        ax.set_title("Hyperbolic Embeddings (Poincaré Disk)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        
        print(f"Visualization saved to {filename}")
