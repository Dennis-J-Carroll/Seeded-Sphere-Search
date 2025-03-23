# SeededSphere Transformers

A neural network-based search optimization library that uses advanced embedding techniques, including hyperbolic and ellipsoidal transformations, to create high-quality representations for information retrieval.

## Overview

SeededSphere Transformers combines multiple geometric approaches to embeddings, particularly for hierarchical and complex data structures:

- **Seeded Sphere Search**: Core search mechanism with spherical embeddings and echo-based relationship mapping
- **Hyperbolic Transformations**: Specialized embeddings for hierarchical data using the Poincaré ball model
- **Ellipsoidal Transformations**: Optimized weight matrices for improved retrieval performance
- **Hybrid Search Engine**: Combines multiple search algorithms with configurable weights

## Key Features

- Dynamic embedding refinement using echo-based relationship mapping
- Support for hyperbolic geometry to better represent hierarchical data
- Ellipsoidal transformation for fine-tuning search quality
- Hybrid search capabilities that combine multiple algorithms
- Comprehensive evaluation metrics and performance benchmarking

## Installation

```bash
# Clone the repository
git clone [your-repository-url]
cd SeededSphere_Transformers

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Search Example

```python
from SeededSphereSearch import SeededSphereSearch

# Initialize search system with documents
sss = SeededSphereSearch()
sss.initialize(documents)

# Perform search
results = sss.search("your query here", top_k=10)
```

### Using Hyperbolic Transformations

```python
from HyperbolicTransformation import HyperbolicTransformation

# Initialize transformation
hyperbolic = HyperbolicTransformation(dimensions=384)

# Transform embeddings
hyperbolic_embeddings = hyperbolic.transform_embeddings(embeddings)

# Calculate similarity scores
score = hyperbolic.calculate_similarity_score(emb1, emb2)
```

### Using Ellipsoidal Transformations

```python
from EllipsoidalTransformation import EllipsoidalTransformation

# Initialize and train
ellipsoidal = EllipsoidalTransformation(dimensions=384)
ellipsoidal.train_weights(query_embeddings, relevant_docs, non_relevant_docs)

# Transform query
transformed_query = ellipsoidal.transform_query(query_embedding)
```

## Project Structure

- `SeededSphereSearch.py`: Main search implementation
- `HyperbolicTransformation.py`: Hyperbolic space transformation for hierarchical data
- `EllipsoidalTransformation.py`: Ellipsoidal weight transformation for refined search
- `HybridSearchEngine.py`: Combines multiple search algorithms
- `evaluate_search_performance.py`: Metrics and evaluation tools
- `demo_sss.py`: Demonstration of the search system
- `Test_ellipsoidal_transform_sci-fi_corpus.py`: Test script for ellipsoidal transformations

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]
