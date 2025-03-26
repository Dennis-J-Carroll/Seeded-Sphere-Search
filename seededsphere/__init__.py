"""
SeededSphere Transformers Package

A neural network-based search optimization library that uses advanced embedding techniques,
including hyperbolic and ellipsoidal transformations, to create high-quality 
representations for information retrieval.

Core Components:
- Seeded Sphere Search: Core search mechanism with spherical embeddings
- Echo Layer: Advanced echo mechanism for embedding refinement
- Hyperbolic Transformations: Specialized embeddings for hierarchical data
- Ellipsoidal Transformations: Optimized weight matrices for improved retrieval
- Hybrid Search Engine: Combines multiple search algorithms
- Evaluation Tools: Comprehensive metrics and analysis tools
"""

from .core.seeded_sphere_search import SeededSphereSearch
from .core.echo_mechanism import EchoLayer, SparseEchoLayer
from .transformations.hyperbolic_transformation import HyperbolicTransformation
from .transformations.ellipsoidal_transformation import EllipsoidalTransformation
from .search.hybrid_search_engine import HybridSearchEngine
from . import evaluation

__version__ = "0.2.0"

__all__ = [
    'SeededSphereSearch',
    'EchoLayer',
    'SparseEchoLayer',
    'HyperbolicTransformation',
    'EllipsoidalTransformation',
    'HybridSearchEngine',
    'evaluation',
]
