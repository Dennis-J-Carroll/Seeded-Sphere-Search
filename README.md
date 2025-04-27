# Seeded Sphere Search Transformers

This repository contains an implementation of the Seeded Sphere Search algorithm with transformer-based embeddings and additional transformations for enhanced search capabilities.
![sss_imag2](https://github.com/user-attachments/assets/294b8f35-b6c5-4191-9766-e1e5f29d319e)

## Features

- **Transformer-Based Embeddings**: Leverage the power of language models for semantic search
- **Seeded Search**: Use example documents to influence search results
- **Echo Layer Mechanism**: Advanced echo refinement with vectorized operations and parameter tuning
- **Ellipsoidal Transformation**: Apply dimension-wise weighting to improve search relevance
- **Hyperbolic Transformation**: Specialized embeddings for hierarchical relationships
- **Hybrid Search**: Combine multiple search approaches with optimized weighting
- **Evaluation Platform**: Analyze, compare, and visualize search results

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/SeededSphere_Transformers.git
cd SeededSphere_Transformers
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

Additional dependencies for the evaluation platform:

```bash
pip install scikit-learn umap-learn plotly
```

## Project Structure

```
SeededSphere_Transformers/
├── seededsphere/          # Main package
│   ├── core/              # Core search algorithm and echo mechanism
│   ├── transformations/   # Embedding transformation modules
│   ├── search/            # Search engine implementations
│   ├── evaluation/        # Evaluation metrics and analysis tools
│   └── extensions/        # Domain-specific extensions
├── data/                  # Data storage
│   ├── corpora/           # Document corpora in JSON format
│   ├── models/            # Saved models and configurations
│   └── raw_text/          # Raw text files
├── examples/              # Example scripts demonstrating features
├── scripts/               # Utility scripts for data processing
├── tests/                 # Test cases
└── docs/                  # Documentation
```

## Usage

### Basic Search

```python
from seededsphere import SeededSphereSearch

# Initialize the search engine
search = SeededSphereSearch()

# Add documents
search.add_document(id="doc1", title="Document 1", content="This is a sample document about AI.")
search.add_document(id="doc2", title="Document 2", content="Exploring machine learning concepts.")

# Build the index
search.build_index()

# Perform a search
results = search.search("artificial intelligence")
```

### Seeded Search

```python
# Perform a seeded search
results = search.seeded_search(query="neural networks", seed_doc_id="doc1")
```

### Advanced Echo Layer

```python
from seededsphere import EchoLayer
import torch
import numpy as np

# Initialize with custom parameters
echo_layer = EchoLayer(delta=0.15, freq_scale=0.3, min_layers=1, max_layers=5)

# Apply to embeddings
embeddings = torch.tensor(your_embeddings).float()
frequencies = torch.tensor(your_frequencies).float()
relationship_matrix = torch.tensor(your_relationships).float()

# Apply echo refinement with detailed tracking
refined_embeddings, evolution_data = echo_layer(
    embeddings, 
    frequencies, 
    relationship_matrix,
    track_mode='detailed'
)
```

### Ellipsoidal Transformation

```python
from seededsphere import EllipsoidalTransformation

# Initialize the transformation
transformer = EllipsoidalTransformation(dimensions=384)

# Train weights using relevant/non-relevant examples
transformer.train_weights(
    query_embeddings, 
    relevant_doc_embeddings,
    non_relevant_doc_embeddings
)

# Apply to search
results = search_engine.search(
    query, 
    use_ellipsoidal=True, 
    ellipsoidal_transformer=transformer
)
```

### Evaluation

```python
from seededsphere.evaluation import evaluate_search_performance, perform_echo_layer_analysis

# Evaluate search performance
metrics = evaluate_search_performance(search_engine, test_data)

# Analyze echo layer impact
analysis_results = perform_echo_layer_analysis(
    search_engine, 
    doc_ids=['doc1', 'doc2', 'doc3'],
    track_mode='detailed'
)
```

### Running Examples

To run the example scripts:

```bash
# Quick demo
python examples/quick_demo_sss.py

# Full feature demo
python examples/demo_sss.py

# Search UI
python examples/run_search_ui.py --model data/models/sss_model.pkl

# Evaluation platform
python examples/run_evaluation_platform.py --model data/models/sss_model.pkl
```

## Evaluation Platform Features

### Analytics

The analytics tab allows you to:

- Analyze search quality with precision-recall curves
- Calculate metrics like Average Precision and Precision@K
- Create and use ground truth data for evaluation

### Algorithm Comparison

Compare different search approaches:

- Standard semantic search
- Seeded search with different seed documents
- Ellipsoidal and hyperbolic transformations
- Hybrid search approaches

### Embedding Visualization

Visualize your document embeddings:

- View all documents or query results
- Choose between t-SNE, PCA, or UMAP visualizations
- See the relationships in 2D or 3D space
- Analyze echo refinement impact on embeddings

### Session Management

Track and compare your experiments:

- Save search results and analytics
- Export and import sessions
- Compare multiple sessions

## License

[MIT License](LICENSE)
