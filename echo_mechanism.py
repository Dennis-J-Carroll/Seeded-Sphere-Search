# Echo Refinement Mechanism for Embedding Transformation
# A vectorized implementation with parameter tuning and enhanced evolution tracking

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional, Union
import os

class EchoLayer(nn.Module):
    def __init__(self, delta=0.1625, freq_scale=0.325, min_layers=1, max_layers=5):
        """
        Initialize the Echo Layer with configurable parameters
        
        Args:
            delta: Step size for echo refinement (default: 0.1625)
            freq_scale: Scaling factor for frequency-based layer assignment (default: 0.325)
            min_layers: Minimum number of echo iterations (default: 1)
            max_layers: Maximum number of echo iterations (default: 5)
        """
        super(EchoLayer, self).__init__()
        self.delta = delta
        self.freq_scale = freq_scale
        self.min_layers = min_layers
        self.max_layers = max_layers
        self.parameter_history = []
    
    def calculate_layers(self, frequencies):
        """
        Determine echo iterations based on token frequencies
        
        Args:
            frequencies: Tensor of shape [batch_size, num_tokens] with normalized frequency values
            
        Returns:
            Tensor of shape [batch_size, num_tokens] with integer number of layers to use
        """
        return torch.clamp(
            self.min_layers + torch.floor(frequencies * self.freq_scale),
            self.min_layers, 
            self.max_layers
        ).int()
    
    def forward(self, embeddings, frequencies=None, relationship_matrix=None, track_mode='basic'):
        """
        Apply echo refinement to input embeddings with optimized vectorized operations.
        
        Args:
            embeddings: Input embeddings of shape [batch_size, num_tokens, embedding_dim]
            frequencies: Tensor of shape [batch_size, num_tokens] with normalized frequency values
            relationship_matrix: Tensor of shape [batch_size, num_tokens, num_tokens] with relationship strengths
            track_mode: Evolution tracking detail level ('basic', 'detailed', or 'tokenlevel')
        
        Returns:
            Tuple containing:
            - Refined embeddings of same shape as input
            - Evolution tracking information (format depends on track_mode)
        """
        batch_size, num_tokens, embedding_dim = embeddings.shape
        
        # Default to max layers if frequencies not provided
        if frequencies is None:
            echo_layers = torch.ones(batch_size, num_tokens, 
                                    device=embeddings.device).int() * self.max_layers
        else:
            echo_layers = self.calculate_layers(frequencies)
        
        # Initialize with normalized input embeddings
        refined = F.normalize(embeddings, p=2, dim=2)
        
        # Initialize evolution tracking based on track_mode
        if track_mode == 'basic':
            # Only track initial and final states
            evolution = [refined.clone()]
        elif track_mode == 'detailed':
            # Track per-batch evolution
            evolution = [refined.clone()]
        elif track_mode == 'tokenlevel':
            # Track per-token evolution
            evolution = [refined.clone()]
            # Find max number of layers for pre-allocation
            max_layers_batch = echo_layers.max().item()
            # Pre-allocate token evolution tensor [batch, tokens, max_layers+1, dim]
            token_evolution = torch.zeros(
                batch_size, num_tokens, max_layers_batch + 1, embedding_dim,
                device=embeddings.device
            )
            # Set initial embeddings
            for b in range(batch_size):
                for t in range(num_tokens):
                    token_evolution[b, t, 0] = refined[b, t]
        
        # Process each item in batch
        for b in range(batch_size):
            # Create a mask to exclude self-relationships (diagonal elements)
            mask = torch.ones(num_tokens, num_tokens, device=embeddings.device) - torch.eye(num_tokens, device=embeddings.device)
            
            # Get maximum number of layers in this batch for iteration
            max_layers_in_batch = echo_layers[b].max().item()
            
            # Track which tokens have been processed for each layer
            processed_tokens = torch.zeros(num_tokens, dtype=torch.bool, device=embeddings.device)
            
            # Apply echo iterations, tracking per layer or per token as needed
            for layer in range(max_layers_in_batch):
                # Which tokens need this layer
                active_tokens = layer < echo_layers[b]
                
                if not active_tokens.any():
                    continue
                    
                # Get current embeddings for this batch
                current_embeddings = refined[b]
                
                # Option 1: Vectorized calculation for all tokens at once
                # Calculate all pairwise differences [num_tokens, num_tokens, embedding_dim]
                # This optimizes the nested loop in the original implementation
                diffs = current_embeddings.unsqueeze(1) - current_embeddings.unsqueeze(0)
                
                # Apply relationship weights and mask out self-relationships
                # [num_tokens, num_tokens, 1] * [num_tokens, num_tokens, embedding_dim]
                weighted_diffs = relationship_matrix[b].unsqueeze(-1) * diffs * mask.unsqueeze(-1)
                
                # Sum up all influences for each token [num_tokens, embedding_dim]
                shifts = weighted_diffs.sum(dim=1) * self.delta
                
                # Apply shifts only to active tokens
                shifts = shifts * active_tokens.unsqueeze(-1)
                
                # Apply shifts and normalize
                updated_embeddings = F.normalize(current_embeddings + shifts, p=2, dim=1)
                refined[b] = updated_embeddings
                
                # Update processed tokens
                processed_tokens = processed_tokens | active_tokens
                
                # Track evolution based on the track mode
                if track_mode == 'detailed':
                    # Only track if any tokens were active
                    if active_tokens.any():
                        evolution.append(refined.clone())
                
                elif track_mode == 'tokenlevel':
                    # Store token states for active tokens
                    for t in range(num_tokens):
                        if active_tokens[t]:
                            token_evolution[b, t, layer+1] = refined[b, t]
                    
            # For basic tracking, we only add the final state after processing all tokens
            if track_mode == 'basic':
                evolution.append(refined.clone())
        
        # Return different evolution formats based on track_mode
        if track_mode == 'tokenlevel':
            return refined, (evolution, token_evolution)
        else:
            return refined, evolution
    
    def tune_parameters(self, embeddings, frequencies, relationship_matrix, target_similarity_matrix=None):
        """
        Auto-tune delta and frequency_scale parameters based on dataset statistics
        
        Args:
            embeddings: Input embeddings
            frequencies: Token frequencies
            relationship_matrix: Token relationship matrix
            target_similarity_matrix: Optional target similarity matrix to optimize towards
            
        Returns:
            Tuple of (delta, freq_scale) optimized parameters
        """
        # Start with default parameters
        best_delta = self.delta
        best_freq_scale = self.freq_scale
        best_score = float('inf')
        
        # Define parameter ranges to search
        delta_range = [0.05, 0.1, 0.15, 0.2, 0.25]
        freq_scale_range = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # If we have a target similarity matrix, optimize towards it
        # Otherwise, optimize for stability and convergence
        for delta in delta_range:
            for freq_scale in freq_scale_range:
                # Create temporary layer with these parameters
                temp_layer = EchoLayer(delta=delta, freq_scale=freq_scale, 
                                      min_layers=self.min_layers, max_layers=self.max_layers)
                
                # Run forward pass and get results
                refined, evolution = temp_layer(embeddings, frequencies, relationship_matrix)
                
                if target_similarity_matrix is not None:
                    # Calculate similarity matrix from refined embeddings
                    batch_size = refined.shape[0]
                    refined_sim = torch.zeros_like(target_similarity_matrix)
                    for b in range(batch_size):
                        refined_sim[b] = torch.mm(refined[b], refined[b].transpose(0, 1))
                    
                    # Calculate error against target
                    error = torch.norm(refined_sim - target_similarity_matrix)
                    score = error.item()
                else:
                    # Measure stability by checking how much embeddings change in final iterations
                    if isinstance(evolution, tuple):
                        # Handle tokenlevel tracking
                        last_changes = torch.norm(evolution[0][-1] - evolution[0][-2])
                    else:
                        last_changes = torch.norm(evolution[-1] - evolution[-2])
                    score = last_changes.item()
                
                if score < best_score:
                    best_score = score
                    best_delta = delta
                    best_freq_scale = freq_scale
        
        # Save results to history
        self.parameter_history.append({
            'delta': best_delta,
            'freq_scale': best_freq_scale,
            'score': best_score
        })
        
        # Update parameters
        self.delta = best_delta
        self.freq_scale = best_freq_scale
        
        return best_delta, best_freq_scale

class SparseEchoLayer(EchoLayer):
    """
    Extension of EchoLayer that uses sparse relationship matrices for large vocabularies
    """
    def forward(self, embeddings, frequencies=None, relationship_matrix=None, k_neighbors=10, track_mode='basic'):
        """
        Apply echo refinement using sparse approximation for large vocabularies
        
        Args:
            embeddings: Input embeddings 
            frequencies: Token frequencies
            relationship_matrix: Dense or sparse relationship matrix
            k_neighbors: Number of neighbors to consider for each token (when using kNN)
            track_mode: Evolution tracking mode
            
        Returns:
            Refined embeddings and evolution tracking
        """
        batch_size, num_tokens, embedding_dim = embeddings.shape
        
        # If relationship matrix is sparse or None, compute it using kNN
        if relationship_matrix is None or torch.is_sparse(relationship_matrix):
            # Case for large vocabularies: use k nearest neighbors
            relationship_matrix = self._compute_knn_relationships(embeddings, k=k_neighbors)
            
        # Default to max layers if frequencies not provided
        if frequencies is None:
            echo_layers = torch.ones(batch_size, num_tokens, 
                                    device=embeddings.device).int() * self.max_layers
        else:
            echo_layers = self.calculate_layers(frequencies)
        
        # Call the parent class implementation with the computed relationships
        return super().forward(embeddings, frequencies, relationship_matrix, track_mode)
    
    def _compute_knn_relationships(self, embeddings, k=10):
        """
        Compute k-nearest neighbor relationship matrix
        
        Args:
            embeddings: Token embeddings [batch_size, num_tokens, embedding_dim]
            k: Number of neighbors to consider
            
        Returns:
            Sparse relationship matrix
        """
        batch_size, num_tokens, _ = embeddings.shape
        relationship_matrix = torch.zeros(batch_size, num_tokens, num_tokens, device=embeddings.device)
        
        for b in range(batch_size):
            # Compute cosine similarity for this batch
            emb_normalized = F.normalize(embeddings[b], p=2, dim=1)
            similarities = torch.mm(emb_normalized, emb_normalized.t())
            
            # Set diagonal to -inf to exclude self-similarity
            similarities.fill_diagonal_(-float('inf'))
            
            # Get top-k neighbors for each token
            _, top_k_indices = torch.topk(similarities, k=k, dim=1)
            
            # Create sparse relationship matrix
            for i in range(num_tokens):
                for j_idx in range(k):
                    j = top_k_indices[i, j_idx].item()
                    # Use similarity as relationship strength
                    relationship_matrix[b, i, j] = similarities[i, j].item()
            
            # Normalize relationship strengths
            relationship_matrix[b] = F.normalize(relationship_matrix[b], p=1, dim=1)
            
        return relationship_matrix

def create_test_scenario() -> Tuple[List[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a test scenario with word embeddings and their relationships.
    
    Returns:
        Tuple containing:
        - words: List of test words
        - embeddings: Initial word embeddings tensor
        - relationships: Matrix of word relationships
        - frequencies: Token frequency tensor
    """
    # Create initial random embeddings for 5 words
    words = ["cat", "dog", "tiger", "house", "building"]
    
    # Initial random embeddings (normalized)
    dim = 64  # Embedding dimension
    embeddings = F.normalize(torch.randn(1, len(words), dim), p=2, dim=2)
    
    # Define relationship matrix (higher value = stronger relationship)
    relationships = torch.tensor([
        [  # Relationships for batch item 0
            [0.0, 0.8, 0.7, 0.2, 0.1],  # cat's relationships
            [0.8, 0.0, 0.6, 0.3, 0.2],  # dog's relationships
            [0.7, 0.6, 0.0, 0.1, 0.1],  # tiger's relationships
            [0.2, 0.3, 0.1, 0.0, 0.9],  # house's relationships
            [0.1, 0.2, 0.1, 0.9, 0.0],  # building's relationships
        ]
    ]).float()
    
    # Define token frequencies (normalized to [0,1])
    frequencies = torch.tensor([[0.8, 0.9, 0.4, 0.7, 0.5]]).float()
    
    return words, embeddings, relationships, frequencies

def analyze_embedding_changes(
    words: List[str],
    evolution: Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]
) -> Dict[str, np.ndarray]:
    """
    Analyze how embeddings change through echo layers.
    
    Args:
        words: List of words
        evolution: Evolution tracking data (either list of tensors or tuple for tokenlevel tracking)
        
    Returns:
        Dictionary containing analysis metrics
    """
    # Handle different evolution formats
    if isinstance(evolution, tuple):
        # For tokenlevel tracking, use the batch evolution
        evolution_tensors = evolution[0]
    else:
        evolution_tensors = evolution
    
    # Convert evolution tensors to numpy for analysis
    evolution_np = [e[0].detach().numpy() for e in evolution_tensors]
    
    # Calculate displacement for each word through evolution
    displacements = []
    for i in range(len(evolution_np) - 1):
        step_displacement = np.linalg.norm(
            evolution_np[i+1] - evolution_np[i],
            axis=1
        )
        displacements.append(step_displacement)
    
    # Handle case when there are no displacements (only one evolution step)
    if len(displacements) == 0:
        # Create zero displacement for each word
        displacement_matrix = np.zeros((1, len(words)))
        final_displacement = np.zeros(len(words))
    else:
        # Stack displacements for each step
        displacement_matrix = np.stack(displacements, axis=0)
        final_displacement = np.sum(displacement_matrix, axis=0)
    
    return {
        'displacement_matrix': displacement_matrix,
        'final_displacement': final_displacement
    }

def plot_and_save_experiment_results(
    words: List[str],
    initial_sim: np.ndarray,
    refined_sim: np.ndarray,
    displacement_matrix: np.ndarray,
    save_dir: str = "results",
    experiment_name: str = "echo_experiment"
) -> str:
    """
    Plot experiment results and save them as PNG files.
    
    Args:
        words: List of token words
        initial_sim: Initial similarity matrix
        refined_sim: Refined similarity matrix
        displacement_matrix: Matrix of displacements through iterations
        save_dir: Directory to save results
        experiment_name: Prefix for saved files
        
    Returns:
        Path to the overview figure
    """
    # Create results directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot initial similarities
    im1 = axes[0].imshow(initial_sim, cmap='viridis')
    axes[0].set_title('Initial Similarities')
    axes[0].set_xticks(range(len(words)))
    axes[0].set_yticks(range(len(words)))
    axes[0].set_xticklabels(words, rotation=45)
    axes[0].set_yticklabels(words)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Plot refined similarities
    im2 = axes[1].imshow(refined_sim, cmap='viridis')
    axes[1].set_title('Refined Similarities')
    axes[1].set_xticks(range(len(words)))
    axes[1].set_yticks(range(len(words)))
    axes[1].set_xticklabels(words, rotation=45)
    axes[1].set_yticklabels(words)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot displacement over echo steps
    axes[2].plot(displacement_matrix)
    axes[2].set_title('Embedding Displacement')
    axes[2].set_xlabel('Echo Step')
    axes[2].set_ylabel('Displacement')
    axes[2].legend(words)
    
    plt.tight_layout()
    
    # Save the figure
    fig_path = os.path.join(save_dir, f"{experiment_name}_overview.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also save individual plots for better detail
    # Initial similarities
    fig_initial = plt.figure(figsize=(8, 6))
    im = plt.imshow(initial_sim, cmap='viridis')
    plt.title('Initial Similarities')
    plt.xticks(range(len(words)), words, rotation=45)
    plt.yticks(range(len(words)), words)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_initial_sim.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Refined similarities
    fig_refined = plt.figure(figsize=(8, 6))
    im = plt.imshow(refined_sim, cmap='viridis')
    plt.title('Refined Similarities')
    plt.xticks(range(len(words)), words, rotation=45)
    plt.yticks(range(len(words)), words)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_refined_sim.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Displacement plot
    fig_disp = plt.figure(figsize=(10, 6))
    for i, word in enumerate(words):
        plt.plot(displacement_matrix[:, i], label=word)
    plt.title('Embedding Displacement Over Echo Steps')
    plt.xlabel('Echo Step')
    plt.ylabel('Displacement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{experiment_name}_displacement.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results saved to {save_dir}/")
    return fig_path

def visualize_token_evolution(
    words: List[str],
    token_evolution: torch.Tensor,
    save_dir: str = "results",
    experiment_name: str = "token_evolution"
) -> None:
    """
    Visualize token-level evolution through echo layers.
    
    Args:
        words: List of token words
        token_evolution: Tensor of token evolution [batch, tokens, layers, dim]
        save_dir: Directory to save results
        experiment_name: Prefix for saved files
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size, num_tokens, num_layers, dim = token_evolution.shape
    
    # For each token, visualize its movement
    for b in range(batch_size):
        # Use PCA to project high-dimensional embeddings to 2D for visualization
        from sklearn.decomposition import PCA
        
        # Reshape to [tokens*layers, dim]
        all_embeddings = token_evolution[b].reshape(-1, dim).detach().cpu().numpy()
        
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings)
        
        # Reshape back to [tokens, layers, 2]
        token_paths = embeddings_2d.reshape(num_tokens, num_layers, 2)
        
        # Plot token paths
        plt.figure(figsize=(10, 8))
        
        # Plot paths with increasing color intensity
        for i, word in enumerate(words):
            x, y = token_paths[i, :, 0], token_paths[i, :, 1]
            
            # Create a colormap from light to dark
            colors = plt.cm.viridis(np.linspace(0.3, 1.0, num_layers))
            
            # Plot each segment with increasing color intensity
            for j in range(num_layers-1):
                plt.plot(x[j:j+2], y[j:j+2], 'o-', color=colors[j], 
                         linewidth=2, markersize=6, label=f"{word} {j}" if j==0 else "")
            
            # Add word labels at start and end points
            plt.annotate(f"{word} (start)", (x[0], y[0]), 
                         xytext=(10, 5), textcoords='offset points')
            plt.annotate(f"{word} (end)", (x[-1], y[-1]), 
                         xytext=(10, -5), textcoords='offset points')
        
        plt.title(f"Token Evolution Paths in PCA Space")
        plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.grid(True, alpha=0.3)
        
        # Add a legend with only unique entries
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_token_paths.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Also create a heatmap showing total displacement for each token and layer
        displacements = np.zeros((num_tokens, num_layers-1))
        
        for i in range(num_tokens):
            for j in range(num_layers-1):
                # Calculate Euclidean distance between consecutive embeddings
                emb1 = token_evolution[b, i, j].detach().cpu().numpy()
                emb2 = token_evolution[b, i, j+1].detach().cpu().numpy()
                displacements[i, j] = np.linalg.norm(emb2 - emb1)
        
        # Plot heatmap
        plt.figure(figsize=(10, 6))
        im = plt.imshow(displacements, cmap='magma')
        plt.colorbar(im, label='Displacement Magnitude')
        plt.title('Token Displacement by Echo Layer')
        plt.xlabel('Echo Layer')
        plt.ylabel('Token')
        plt.yticks(range(num_tokens), words)
        plt.xticks(range(num_layers-1), [f"Layer {i+1}" for i in range(num_layers-1)])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{experiment_name}_displacement_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()

def run_echo_experiment_with_visualization(
    echo_layer: EchoLayer,
    save_results: bool = True,
    save_dir: str = "results",
    experiment_name: str = "echo_experiment",
    track_mode: str = "detailed"
) -> Dict:
    """
    Run experiment with echo refinement and save visualizations
    
    Args:
        echo_layer: Instance of EchoLayer to test
        save_results: Whether to save visualization results
        save_dir: Directory to save results
        experiment_name: Prefix for saved files
        track_mode: Level of evolution tracking detail
        
    Returns:
        Dictionary containing experiment results and analysis
    """
    # Create test scenario
    words, embeddings, relationships, frequencies = create_test_scenario()
    
    # Apply echo refinement with specified tracking mode
    refined_embeddings, evolution = echo_layer(
        embeddings, frequencies, relationships, track_mode=track_mode
    )
    
    # If token-level evolution is available
    has_token_evolution = isinstance(evolution, tuple) and len(evolution) == 2
    
    if has_token_evolution:
        batch_evolution, token_evolution = evolution
    else:
        batch_evolution = evolution
    
    # Calculate similarity matrices before and after
    initial_sim = cosine_similarity(embeddings[0].detach().numpy())
    refined_sim = cosine_similarity(refined_embeddings[0].detach().numpy())
    
    # Analyze embedding evolution
    evolution_analysis = analyze_embedding_changes(words, evolution)
    
    # Save results if requested
    if save_results:
        fig_path = plot_and_save_experiment_results(
            words, 
            initial_sim, 
            refined_sim, 
            evolution_analysis['displacement_matrix'],
            save_dir=save_dir,
            experiment_name=experiment_name
        )
        
        if has_token_evolution:
            visualize_token_evolution(
                words,
                token_evolution,
                save_dir=save_dir,
                experiment_name=experiment_name
            )
        
        print(f"Visualizations saved to {fig_path}")
    
    return {
        'words': words,
        'initial_embeddings': embeddings,
        'refined_embeddings': refined_embeddings,
        'initial_similarity': initial_sim,
        'refined_similarity': refined_sim,
        'evolution': evolution,
        'analysis': evolution_analysis
    }

def parameter_sweep(
    embeddings: torch.Tensor,
    frequencies: torch.Tensor,
    relationship_matrix: torch.Tensor,
    delta_range: List[float] = [0.05, 0.1, 0.15, 0.2, 0.25],
    freq_scale_range: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5],
    save_dir: str = "results"
) -> Dict:
    """
    Perform parameter sweep to find optimal parameters
    
    Args:
        embeddings: Input embeddings
        frequencies: Token frequencies
        relationship_matrix: Relationship matrix
        delta_range: List of delta values to try
        freq_scale_range: List of frequency scale values to try
        save_dir: Directory to save results
    
    Returns:
        Dictionary with parameter sweep results
    """
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Store results
    results = {}
    best_params = None
    best_score = float('inf')
    
    # Grid search
    for delta in delta_range:
        for freq_scale in freq_scale_range:
            # Create layer with these parameters
            echo_layer = EchoLayer(
                delta=delta,
                freq_scale=freq_scale, 
                min_layers=1,
                max_layers=5
            )
            
            # Run experiment
            result = run_echo_experiment_with_visualization(
                echo_layer,
                save_results=False
            )
            
            # Calculate score (total displacement as a simple metric)
            score = np.sum(result['analysis']['final_displacement'])
            
            # Store result
            params = (delta, freq_scale)
            results[params] = {
                'score': score,
                'final_displacement': result['analysis']['final_displacement'],
                'refined_similarity': result['refined_similarity']
            }
            
            # Check if this is the best so far
            if score < best_score:
                best_score = score
                best_params = params
    
    # Generate heatmap of results
    delta_values = sorted(delta_range)
    freq_scale_values = sorted(freq_scale_range)
    scores = np.zeros((len(delta_values), len(freq_scale_values)))
    
    for i, delta in enumerate(delta_values):
        for j, freq_scale in enumerate(freq_scale_values):
            params = (delta, freq_scale)
            if params in results:
                scores[i, j] = results[params]['score']
    
    # Plot heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(scores, cmap='viridis')
    plt.colorbar(im, label='Score (lower is better)')
    plt.title('Parameter Sweep Results')
    plt.xlabel('Frequency Scale')
    plt.ylabel('Delta')
    plt.xticks(range(len(freq_scale_values)), freq_scale_values)
    plt.yticks(range(len(delta_values)), delta_values)
    
    # Mark best parameters
    best_i = delta_values.index(best_params[0])
    best_j = freq_scale_values.index(best_params[1])
    plt.plot(best_j, best_i, 'r*', markersize=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"parameter_sweep.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'results': results,
        'best_params': best_params,
        'best_score': best_score,
        'score_matrix': scores
    }

# Example usage
if __name__ == "__main__":
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Create echo layer with default parameters
    echo_layer = EchoLayer(
        delta=0.1625,  # Step size for refinement
        freq_scale=0.325,  # Frequency scaling factor
        min_layers=1,
        max_layers=5
    )

    # Run experiment with detailed tracking and generate visualizations
    print("Running echo experiment with detailed tracking...")
    results = run_echo_experiment_with_visualization(
        echo_layer, 
        save_results=True,
        track_mode="detailed"
    )
    
    # Also run with token-level tracking for more detailed analysis
    print("\nRunning echo experiment with token-level tracking...")
    results_token = run_echo_experiment_with_visualization(
        echo_layer, 
        save_results=True,
        experiment_name="echo_experiment_tokenlevel",
        track_mode="tokenlevel"
    )
    
    # Create sparse layer for larger vocabulary example
    sparse_layer = SparseEchoLayer(
        delta=0.15,
        freq_scale=0.3,
        min_layers=1,
        max_layers=4
    )
    
    print("\nComparing initial vs final similarities for 'cat':")
    print(f"Initial: {results['initial_similarity'][0]}")
    print(f"Refined: {results['refined_similarity'][0]}")

    # Check total displacement for each word
    print("\nTotal embedding displacement:")
    for word, disp in zip(results['words'], results['analysis']['final_displacement']):
        print(f"{word}: {disp:.4f}")
    
    print("\nOptimized Echo Mechanism implementation complete.")
    print("Vectorized operations replace O(N²) nested loops")
    print("Parameter auto-tuning capabilities added")
    print("Enhanced evolution tracking with per-token details")
    print("Results saved to ./results/ directory")
