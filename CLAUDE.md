# SeededSphere Transformers Project Guide

This document contains useful information about the SeededSphere Transformers project for future reference.

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

## Common Commands

### Installation

```bash
# Install in development mode
pip install -e .

# Install requirements
pip install -r requirements.txt
```

### Running Examples

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

### Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test
pytest tests/test_vectorizer.py
```

## Code Style Preferences

- **Naming Conventions**:
  - Classes: PascalCase (e.g., `SeededSphereSearch`)
  - Functions/Methods: snake_case (e.g., `search_documents`)
  - Variables: snake_case (e.g., `doc_embeddings`)
  - Constants: UPPER_SNAKE_CASE (e.g., `MAX_BATCH_SIZE`)

- **Documentation**:
  - Use Google-style docstrings for all public functions/methods
  - Include type hints for parameters and return values

- **Imports**:
  - Group imports in the following order:
    1. Standard library imports
    2. Third-party library imports
    3. Local application imports
  - Sort imports alphabetically within each group

## Key Components

### SeededSphereSearch

The core search algorithm that implements:
- Document embedding generation
- Relationship mapping
- Echo refinement
- Standard and seeded search

### Echo Layer Mechanism

Advanced echo refinement with:
- Vectorized operations for efficiency
- Parameter tuning capabilities
- Evolution tracking

### Transformations

- **EllipsoidalTransformation**: Applies dimension-wise weighting
- **HyperbolicTransformation**: Specialized for hierarchical relationships

### HybridSearchEngine

Combines multiple search approaches with optimized weighting.

### Evaluation Tools

Comprehensive metrics, analysis tools, and visualization capabilities.

## Dependencies

Key dependencies include:
- torch
- numpy
- scikit-learn
- transformers
- matplotlib
- pandas
- flask (for UI)