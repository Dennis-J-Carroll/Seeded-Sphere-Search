Seeded Sphere Search Mechanism

A neural network-based search optimization that uses dual networks (DNN and DeNN)
to create dynamic spherical word embeddings with echo-based relationship mapping.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter
from cachetools import TTLCache

@dataclass
class SphereConfig:
    """Configuration for Seeded Sphere search"""
    # Base parameters
    min_echo_layers: int = 2
    max_echo_layers: int = 10
    base_anchor_weight: float = 0.7  # Initial weight for anchored context
    context_window_size: int = 5
    
    # Neural Network parameters
    embedding_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8  # For attention mechanism
    dropout: float = 0.1
    
    # Cache settings
    cache_ttl: int = 3600  # 1 hour
    cache_maxsize: int = 10000
    
    # Weight adjustment parameters
    frequency_scale: float = 0.1  # How much frequency affects weights
    relationship_threshold: float = 0.3  # Minimum relationship strength

class WordEmbeddingDNN(nn.Module):
    """Primary DNN for word embedding and context analysis"""
    
    def __init__(self, config: SphereConfig):
        super().__init__()
        self.config = config
        
        # Base embedding layer
        self.word_embedding = nn.Linear(3, config.embedding_dim)  # 3 for freq, pos, mode
        
        # Spherical harmonic attention heads
        self.attention_heads = nn.ModuleList([
            SphericalAttention(
                config.embedding_dim,
                angle=(i * np.pi / 4),  # 8 heads at π/4 intervals
                dropout=config.dropout
            ) for i in range(config.num_heads)
        ])
        
        # Cosine similarity layer for radius calculation
        self.radius_calc = CosineSimilarityRadius(
            config.embedding_dim,
            config.hidden_dim
        )
        
        # Temperature scaling components
        self.base_temperature = nn.Parameter(torch.ones(1) / np.sqrt(config.embedding_dim))
        self.temp_scale = nn.Parameter(torch.ones(config.num_heads))
        self.temp_bias = nn.Parameter(torch.zeros(config.num_heads))
        
        # Adaptive temperature MLP
        self.temp_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_heads)
        )
        
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with micro-residuals and normalization.
        
        Args:
            x: Input tensor of shape (batch_size, 3) containing [freq, pos, mode]
            
        Returns:
            Tuple of (embedded_output, radius)
        """
        batch_size = x.shape[0]
        
        # Initial embedding
        embed = self.word_embedding(x)  # [batch_size, embedding_dim]
        seed_value = embed.mean(dim=0, keepdim=True)  # [1, embedding_dim]
        seed_value = seed_value.expand(batch_size, -1)  # [batch_size, embedding_dim]
        
        # Process through attention heads with micro-residuals
        attention_outputs = []
        for head in self.attention_heads:
            # Apply attention with residual
            head_out = head(embed, seed_value)
            micro_residual = embed + self.dropout(head_out)
            # Normalize each head's output independently
            norm_out = self.layer_norm(micro_residual)
            attention_outputs.append(norm_out)
        
        # Combine attention heads with temperature-scaled softmax
        combined = torch.stack(attention_outputs, dim=1)  # [batch_size, num_heads, embedding_dim]
        
        # Dynamic temperature scaling
        # Get base positional temperatures
        pos_temps = torch.arange(self.config.num_heads, device=combined.device) * (np.pi / 4)
        pos_temps = torch.cos(pos_temps) * 0.5 + 0.5  # Scale to [0, 1]
        
        # Calculate content-based temperature adjustment
        content_temps = self.temp_mlp(combined.mean(dim=1))  # [batch_size, num_heads]
        content_temps = torch.sigmoid(content_temps)  # Scale to [0, 1]
        
        # Combine different temperature components
        temperatures = self.base_temperature * (
            self.temp_scale * pos_temps +
            self.temp_bias +
            content_temps
        )  # [batch_size, num_heads]
        
        # Apply temperature scaling
        attention_logits = combined.mean(dim=-1) / temperatures  # [batch_size, num_heads]
        
        # Spherical softmax with temperature
        head_weights = F.softmax(attention_logits, dim=1)  # [batch_size, num_heads]
        
        # Weight heads with scaled contribution
        weighted_heads = (combined * head_weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, embedding_dim]
        
        # Apply gating mechanism to control information flow
        gate = torch.sigmoid(self.layer_norm(weighted_heads).mean(dim=-1, keepdim=True))
        weighted_heads = weighted_heads * gate
        
        # Final normalization before radius calculation
        pre_radius = self.layer_norm(weighted_heads)
        
        # Calculate radius using mean seed value
        radius = self.radius_calc(pre_radius, seed_value)
        
        return weighted_heads, radius

class SphericalAttention(nn.Module):
    """Attention mechanism using spherical harmonics"""
    
    def __init__(self, dim: int, angle: float, dropout: float = 0.1):
        super().__init__()
        self.angle = angle
        self.scale = dim ** -0.5
        
        # Spherical harmonic transformation
        self.q_transform = nn.Linear(dim, dim)
        self.k_transform = nn.Linear(dim, dim)
        self.v_transform = nn.Linear(dim, dim)
        
        # Learnable rotation parameters
        self.sin_rot = nn.Parameter(torch.sin(torch.ones(dim) * angle))
        self.cos_rot = nn.Parameter(torch.cos(torch.ones(dim) * angle))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, seed_value: torch.Tensor) -> torch.Tensor:
        """Apply spherical attention with learned rotations.
        
        Args:
            x: Input tensor (batch_size, embedding_dim)
            seed_value: Seed tensor (batch_size, embedding_dim)
            
        Returns:
            Attended tensor (batch_size, embedding_dim)
        """
        # Transform inputs
        q = self.q_transform(x)
        k = self.k_transform(seed_value)
        v = self.v_transform(x)
        
        # Apply spherical rotation
        q_rot = q * self.cos_rot + torch.roll(q, 1, dims=-1) * self.sin_rot
        k_rot = k * self.cos_rot + torch.roll(k, 1, dims=-1) * self.sin_rot
        
        # Compute attention scores with temperature scaling
        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * self.scale
        
        # Apply spherical softmax
        attn = self.dropout(F.softmax(scores, dim=-1))
        
        # Mix with original value
        attended = torch.matmul(attn, v)
        
        return attended

class CosineSimilarityRadius(nn.Module):
    """Calculate radius using cosine similarity against seed value"""
    
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor, seed_value: torch.Tensor) -> torch.Tensor:
        """Calculate radius using cosine similarity.
        
        Args:
            x: Input tensor of shape (batch_size, embedding_dim)
            seed_value: Seed tensor of shape (batch_size, embedding_dim)
            
        Returns:
            Radius tensor of shape (batch_size,)
        """
        # Project both tensors
        x_proj = self.projection(x)
        seed_proj = self.projection(seed_value)
        
        # Ensure same shape
        if x_proj.shape != seed_proj.shape:
            # Expand seed_proj to match x_proj's batch dimension
            seed_proj = seed_proj.expand(x_proj.shape[0], -1)
        
        # Normalize for cosine similarity
        x_norm = F.normalize(x_proj, p=2, dim=-1)
        seed_norm = F.normalize(seed_proj, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(x_norm, seed_norm, dim=-1)
        
        # Convert to radius (inverse relationship)
        radius = 1 - similarity  # Larger difference = larger radius
        
        return radius
        


class EchoNetwork(nn.Module):
    """Deep Echo Neural Network for relationship analysis"""
    
    def __init__(self, config: SphereConfig, num_layers: int):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        # To be implemented: Echo network architecture
        
    def forward(self, x, relationships):
        # To be implemented: Forward pass with relationship echoing
        pass

class SeededSphereSearch:
    """
    Main class for Seeded Sphere search mechanism.
    Combines word frequency, position, and contextual relationships
    in a spherical space for optimized search.
    """
    
    def __init__(self, config: Optional[SphereConfig] = None):
        self.config = config or SphereConfig()
        self.primary_dnn = WordEmbeddingDNN(self.config)
        self.echo_networks: Dict[int, EchoNetwork] = {}
        self.word_frequencies = defaultdict(int)
        self.relationship_cache = TTLCache(
            maxsize=self.config.cache_maxsize,
            ttl=self.config.cache_ttl
        )
        
    def __init__(self, config: Optional[SphereConfig] = None):
        self.config = config or SphereConfig()
        self.primary_dnn = WordEmbeddingDNN(self.config)
        self.echo_networks: Dict[int, EchoNetwork] = {}
        self.word_frequencies = defaultdict(int)
        self.relationship_cache = TTLCache(
            maxsize=self.config.cache_maxsize,
            ttl=self.config.cache_ttl
        )
        # Initialize weights optimized for natural text
        self.freq_weight = 0.3   # Moderate weight for frequency variation
        self.pos_weight = 0.3    # Moderate weight for position distribution
        self.mode_weight = 0.4   # Slightly higher weight for common patterns

    def _calculate_mode(self, numbers: List[int]) -> List[int]:
        """Calculate mode(s) of a list of numbers."""
        if not numbers:
            return []
        counter = Counter(numbers)
        max_freq = max(counter.values())
        # Return all values that appear with maximum frequency
        return [num for num, freq in counter.items() if freq == max_freq]

    def analyze_variance_components(self, text: str) -> Dict[str, float]:
        """Analyze and break down the components of variance calculation.
        
        Returns:
            Dict containing individual and combined variance components:
            - raw_variances: Individual variance values before weighting
            - weighted_variances: Variance values after applying weights
            - final_std: The final standard deviation
            - component_percentages: Contribution percentage of each component
        """
        words = text.lower().split()
        if not words:
            return {
                'raw_variances': {'frequency': 0, 'median': 0, 'mode': 0},
                'weighted_variances': {'frequency': 0, 'median': 0, 'mode': 0},
                'final_std': 0,
                'component_percentages': {'frequency': 0, 'median': 0, 'mode': 0}
            }
        
        # Calculate word frequencies and positions
        word_freq = defaultdict(int)
        word_positions = defaultdict(list)
        
        for pos, word in enumerate(words):
            word_freq[word] += 1
            word_positions[word].append(pos)
            
        # Calculate frequency variance
        frequencies = list(word_freq.values())
        mean_freq = sum(frequencies) / len(frequencies)
        freq_variance = sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)
        
        # Calculate position-based statistics
        median_stats = []
        mode_stats = []
        
        for positions in word_positions.values():
            if len(positions) > 1:
                distances = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                
                # Median calculation
                sorted_distances = sorted(distances)
                median_dist = sorted_distances[len(distances)//2]
                median_stats.append(median_dist)
                
                # Mode calculation
                modes = self._calculate_mode(distances)
                if modes:
                    mode_stats.append(sum(modes) / len(modes))
        
        # Calculate individual variances
        median_variance = np.var(median_stats) if len(median_stats) > 1 else 0
        mode_variance = np.var(mode_stats) if len(mode_stats) > 1 else 0
        
        # Store raw variances
        raw_variances = {
            'frequency': freq_variance,
            'median': median_variance,
            'mode': mode_variance
        }
        
        # Calculate weighted variances
        weighted_variances = {
            'frequency': self.freq_weight * freq_variance,
            'median': self.pos_weight * median_variance,
            'mode': self.mode_weight * mode_variance
        }
        
        # Calculate combined variance and final std
        combined_variance = sum(weighted_variances.values())
        final_std = np.sqrt(combined_variance)
        
        # Calculate contribution percentages
        total_raw_variance = sum(v for v in raw_variances.values() if v > 0)
        component_percentages = {
            'frequency': (freq_variance / total_raw_variance * 100) if total_raw_variance > 0 else 0,
            'median': (median_variance / total_raw_variance * 100) if total_raw_variance > 0 else 0,
            'mode': (mode_variance / total_raw_variance * 100) if total_raw_variance > 0 else 0
        }
        
        return {
            'raw_variances': raw_variances,
            'weighted_variances': weighted_variances,
            'final_std': final_std,
            'component_percentages': component_percentages
        }

    def calculate_std(self, text: str) -> float:
        """Calculate standard deviation based on word statistics using frequency, median, and mode."""
        return self.analyze_variance_components(text)['final_std']
        
    def get_echo_layers(self, word: str) -> int:
        """Determine number of echo layers based on word frequency"""
        freq = self.word_frequencies[word]
        # Scale layers based on frequency
        scaled_layers = int(
            self.config.min_echo_layers + 
            (freq * self.config.frequency_scale)
        )
        return min(scaled_layers, self.config.max_echo_layers)
        
    def calculate_context_weights(
        self, 
        word: str, 
        anchor_context: float, 
        calculated_context: float
    ) -> float:
        """
        Balance anchored and calculated context weights
        based on word frequency and relationships
        """
        # To be implemented: Dynamic weight calculation
        pass
        
    def project_to_sphere(self, vector: np.ndarray) -> np.ndarray:
        """Project vector onto unit sphere"""
        # Normalize vector to unit length
        return vector / np.linalg.norm(vector)
        
    def build_relationship_web(self, text: str) -> Dict[str, Dict[str, float]]:
        """Build web of word relationships with weights"""
        # To be implemented: Relationship web construction
        pass
        
    def search(
        self, 
        query: str, 
        corpus: List[str], 
        use_cache: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Perform search using seeded sphere mechanism
        Returns list of (document, relevance_score) tuples
        """
        # To be implemented: Main search logic
        pass

