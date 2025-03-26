# Seeded Sphere Search Evaluation Platform

This document provides an overview of the Seeded Sphere Search Evaluation Platform, a comprehensive tool for analyzing, comparing, and visualizing search results.

## Features

The evaluation platform provides the following capabilities:

1. **Search**: Perform standard and seeded searches with configurable parameters
2. **Analytics**: Calculate precision, recall, and other metrics for search quality evaluation
3. **Comparison**: Compare different search algorithms side-by-side
4. **Visualization**: Visualize document embeddings in 2D or 3D space
5. **Session Management**: Save and load search results and analytics for later reference

## Installation

To use the evaluation platform, you need to install the following dependencies:

```bash
pip install flask numpy scikit-learn umap-learn plotly
```

## Running the Platform

To launch the evaluation platform:

```bash
python run_evaluation_platform.py --model path/to/your/model.pkl
```

Optional parameters:
- `--weights`: Path to ellipsoidal weights file (optional)
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to bind to (default: 5000)
- `--debug`: Run in debug mode

Once launched, open your browser to `http://localhost:5000` to access the interface.

## Using the Platform

### Search Tab

The search tab provides the basic search functionality:

1. Enter a search query in the input field
2. Optionally select a seed document to influence search results
3. Configure advanced options like threshold and maximum results
4. Click "Search" to run the query

Results will be displayed on the right side of the screen, showing document titles, scores, and tags.

### Analytics Tab

The analytics tab allows you to evaluate search quality:

1. Enter a search query
2. Optionally upload a ground truth file containing relevant document IDs
3. Click "Create Ground Truth" to generate ground truth based on top results
4. Click "Run Analytics" to calculate metrics

The platform will display:
- Precision-recall curve
- Metrics at different thresholds
- Precision@K values

### Comparison Tab

The comparison tab allows you to compare different search algorithms:

1. Enter a search query
2. Select which algorithms to compare (standard, seeded, ellipsoidal)
3. If using seeded search, select a seed document
4. Click "Run Comparison"

Results will show side-by-side metrics and execution times for each algorithm.

### Visualization Tab

The visualization tab allows you to explore the embedding space:

1. Choose a visualization method (t-SNE, PCA, or UMAP)
2. Select visualization type (all documents or query results)
3. If visualizing query results, enter a search query
4. Select dimensions (2D or 3D)
5. Click "Create Visualization"

The visualization will show document relationships in the embedding space, with different colors for different document types.

### Sessions Tab

The sessions tab allows you to manage evaluation sessions:

1. Create new sessions with a name
2. View, export, and delete existing sessions
3. Import sessions from exported files

Sessions store search results, analytics, comparisons, and visualizations for later reference.

## API Endpoints

The evaluation platform provides the following API endpoints:

### Search
- `POST /api/search`: Perform a search

### Analytics
- `POST /api/analytics/ground-truth`: Create ground truth data
- `POST /api/analytics/metrics`: Calculate metrics for search results

### Comparison
- `POST /api/compare`: Compare different search algorithms

### Visualization
- `POST /api/visualize/embeddings`: Visualize embeddings

### Session Management
- `GET /api/sessions`: List all available sessions
- `POST /api/sessions`: Create a new session
- `GET /api/sessions/<session_id>`: Get details for a specific session
- `DELETE /api/sessions/<session_id>`: Delete a specific session
- `GET /api/sessions/<session_id>/export`: Export a session to a file
- `POST /api/sessions/import`: Import a session from a file

## Extending the Platform

The evaluation platform is designed to be extensible. You can add new features by:

1. Adding new API endpoints in `seeded_sphere_search_tester.py`
2. Creating new UI components in the template files
3. Adding new JavaScript functionality in the static files

## Troubleshooting

If you encounter issues:

1. Check that all dependencies are installed
2. Verify that the model file exists and is valid
3. Check browser console for JavaScript errors
4. Look at the server logs for backend errors

For ellipsoidal transformation support, make sure you provide a weights file. 