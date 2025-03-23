# Changelog

All notable changes to the Seeded Sphere Search Mechanism project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-03-02

### Added
- Hyperbolic transformation implementation for hierarchical data (HyperbolicTransformation.py)
- Hybrid search engine for combining multiple search algorithms (HybridSearchEngine.py)
- Enhanced evaluation framework with comprehensive metrics (enhanced_evaluation.py)
- Demonstration script showcasing all features (demo_sss.py)
- Detailed README.md and CHANGELOG.md files
- New requirements.txt file with all dependencies

### Changed
- Optimized echo refinement process with early stopping and progress tracking
- Improved relationship matrix construction with batch processing and thresholding
- Enhanced embedding generation with batched processing and GPU support
- Updated search function to support ellipsoidal transformation and ANN for large corpora

### Fixed
- Critical bug in refined embeddings assignment (`self.refyined_embeddings` → `self.refined_embeddings`)
- Memory inefficiency during relationship matrix computation for large corpora
- Performance bottlenecks in embedding generation process

## [0.1.0] - 2025-02-15

### Added
- Initial implementation of Seeded Sphere Search mechanism
- Basic transformer-based embedding generation
- Relationship mapping using TF-IDF and cosine similarity
- Echo refinement process for improving embeddings
- Basic search functionality with spherical distance scoring
- Ellipsoidal transformation for dimension weighting
- Simple evaluation script (evaluate_search_performance.py)
- Basic run script (run-sss.py)

### Known Issues
- Memory inefficiency for large corpora
- Slow embedding generation process
- Limited evaluation metrics
- No support for non-Euclidean geometries
