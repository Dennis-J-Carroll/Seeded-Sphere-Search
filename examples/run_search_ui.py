#!/usr/bin/env python3
"""
Launcher script for the Seeded Sphere Search UI
"""

import os
import sys
import argparse
import shutil

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Launch the Seeded Sphere Search UI")
    parser.add_argument("--model", default="sss_model.pkl", help="Path to the SSS model file")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()
    
    # Get the current directory and UI directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ui_dir = os.path.join(current_dir, "examples", "ui")
    
    # Create the templates directory if it doesn't exist
    templates_dir = os.path.join(ui_dir, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create the index.html file directly
    index_path = os.path.join(templates_dir, "index.html")
    if not os.path.exists(index_path):
        print(f"Creating template file at {index_path}")
        with open(index_path, 'w') as f:
            f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Seeded Sphere Search Evaluation Platform</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <link rel="stylesheet" href="/static/css/main.css">
    <!-- Add Chart.js for visualizations -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
    <!-- Add Plotly.js for 3D visualizations -->
    <script src="https://cdn.plot.ly/plotly-2.12.1.min.js"></script>
</head>
<body>
    <div class="container-fluid">
        <header class="py-3 mb-4 border-bottom">
            <h1 class="display-5 fw-bold">Seeded Sphere Search Evaluation Platform</h1>
        </header>

        <!-- Tabs Navigation -->
        <ul class="nav nav-tabs mb-4" id="mainTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="search-tab" data-bs-toggle="tab" data-bs-target="#search" type="button" role="tab" aria-controls="search" aria-selected="true">
                    Search Testing
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="analytics-tab" data-bs-toggle="tab" data-bs-target="#analytics" type="button" role="tab" aria-controls="analytics" aria-selected="false">
                    Analytics & Metrics
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="compare-tab" data-bs-toggle="tab" data-bs-target="#compare" type="button" role="tab" aria-controls="compare" aria-selected="false">
                    Algorithm Comparison
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="visualize-tab" data-bs-toggle="tab" data-bs-target="#visualize" type="button" role="tab" aria-controls="visualize" aria-selected="false">
                    Embedding Visualization
                </button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="sessions-tab" data-bs-toggle="tab" data-bs-target="#sessions" type="button" role="tab" aria-controls="sessions" aria-selected="false">
                    Saved Sessions
                </button>
            </li>
        </ul>

        <!-- Tab Content -->
        <div class="tab-content" id="mainTabsContent">
            <!-- Search Testing Tab (original UI) -->
            <div class="tab-pane fade show active" id="search" role="tabpanel" aria-labelledby="search-tab">
                <!-- Original search UI goes here -->
                <div class="row">
                    <div class="col-md-4">
                        <!-- Search Parameters Panel -->
                        <!-- (Existing search form) -->
                    </div>
                    <div class="col-md-8">
                        <!-- Results Panel -->
                        <!-- (Existing results display) -->
                    </div>
                </div>
            </div>

            <!-- Analytics & Metrics Tab -->
            <div class="tab-pane fade" id="analytics" role="tabpanel" aria-labelledby="analytics-tab">
                <div class="row">
                    <div class="col-md-4">
                        <!-- Analytics Controls -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title">Analytics Configuration</h5>
                            </div>
                            <div class="card-body">
                                <form id="analytics-form">
                                    <div class="mb-3">
                                        <label for="ground-truth-set" class="form-label">Ground Truth Dataset</label>
                                        <select id="ground-truth-set" class="form-select">
                                            <option value="default">Default Evaluation Set</option>
                                            <option value="custom">Custom Judgments</option>
                                        </select>
                                    </div>
                                    <div class="mb-3">
                                        <label for="analytics-query" class="form-label">Query</label>
                                        <input type="text" class="form-control" id="analytics-query" placeholder="Enter your query...">
                                    </div>
                                    <div class="mb-3">
                                        <label for="metrics-type" class="form-label">Metrics</label>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="precision_recall" id="precision-recall-check" checked>
                                            <label class="form-check-label" for="precision-recall-check">
                                                Precision-Recall Curve
                                            </label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="f1_threshold" id="f1-threshold-check" checked>
                                            <label class="form-check-label" for="f1-threshold-check">
                                                F1 vs. Threshold
                                            </label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="precision_k" id="precision-k-check" checked>
                                            <label class="form-check-label" for="precision-k-check">
                                                Precision@K
                                            </label>
                                        </div>
                                    </div>
                                    <button type="submit" class="btn btn-primary">Generate Analytics</button>
                                </form>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-8">
                        <!-- Analytics Charts -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title">Performance Metrics</h5>
                            </div>
                            <div class="card-body">
                                <!-- Charts will be inserted here -->
                                <div class="row">
                                    <div class="col-md-6">
                                        <canvas id="precision-recall-chart"></canvas>
                                    </div>
                                    <div class="col-md-6">
                                        <canvas id="f1-threshold-chart"></canvas>
                                    </div>
                                </div>
                                <div class="row mt-4">
                                    <div class="col-md-6">
                                        <canvas id="precision-k-chart"></canvas>
                                    </div>
                                    <div class="col-md-6">
                                        <div id="metrics-summary" class="p-3 border rounded">
                                            <!-- Summary metrics will go here -->
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Algorithm Comparison Tab -->
            <div class="tab-pane fade" id="compare" role="tabpanel" aria-labelledby="compare-tab">
                <div class="row">
                    <div class="col-md-4">
                        <!-- Comparison Controls -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title">Comparison Configuration</h5>
                            </div>
                            <div class="card-body">
                                <form id="comparison-form">
                                    <div class="mb-3">
                                        <label for="comparison-query" class="form-label">Query</label>
                                        <input type="text" class="form-control" id="comparison-query" placeholder="Enter your query...">
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">Algorithms to Compare</label>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="standard" id="standard-check" checked>
                                            <label class="form-check-label" for="standard-check">
                                                Standard Search
                                            </label>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="seeded" id="seeded-check" checked>
                                            <label class="form-check-label" for="seeded-check">
                                                Seeded Search
                                            </label>
                                        </div>
                                        <div id="seeded-options" class="ms-4 mb-2">
                                            <label for="seed-document" class="form-label">Seed Document</label>
                                            <select id="seed-document" class="form-select form-select-sm">
                                                <!-- Documents will be loaded here -->
                                            </select>
                                            <label for="seed-weight" class="form-label mt-2">Seed Weight</label>
                                            <input type="range" class="form-range" min="0" max="1" step="0.1" value="0.5" id="seed-weight">
                                            <span id="seed-weight-value">0.5</span>
                                        </div>
                                        <div class="form-check">
                                            <input class="form-check-input" type="checkbox" value="ellipsoidal" id="ellipsoidal-check">
                                            <label class="form-check-label" for="ellipsoidal-check">
                                                Ellipsoidal Transformation
                                            </label>
                                        </div>
                                    </div>
                                    <div class="mb-3">
                                        <label for="comparison-threshold" class="form-label">Threshold: <span id="comparison-threshold-value">0.7</span></label>
                                        <input type="range" class="form-range" min="0" max="1" step="0.05" value="0.7" id="comparison-threshold">
                                    </div>
                                    <button type="submit" class="btn btn-primary">Compare Algorithms</button>
                                </form>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-8">
                        <!-- Comparison Results -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title">Comparison Results</h5>
                            </div>
                            <div class="card-body">
                                <!-- Performance metrics comparison -->
                                <h6>Performance Metrics</h6>
                                <div class="table-responsive">
                                    <table class="table table-sm" id="metrics-table">
                                        <thead>
                                            <tr>
                                                <th>Algorithm</th>
                                                <th>Precision</th>
                                                <th>Recall</th>
                                                <th>F1</th>
                                                <th>MAP</th>
                                                <th>Time (ms)</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <!-- Metrics will be inserted here -->
                                        </tbody>
                                    </table>
                                </div>
                                
                                <!-- Charts for visual comparison -->
                                <div class="row mt-4">
                                    <div class="col-md-6">
                                        <canvas id="comparison-precision-chart"></canvas>
                                    </div>
                                    <div class="col-md-6">
                                        <canvas id="comparison-recall-chart"></canvas>
                                    </div>
                                </div>
                                
                                <!-- Side-by-side results -->
                                <h6 class="mt-4">Result Comparison</h6>
                                <div class="row" id="side-by-side-results">
                                    <!-- Results will be inserted here -->
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Embedding Visualization Tab -->
            <div class="tab-pane fade" id="visualize" role="tabpanel" aria-labelledby="visualize-tab">
                <div class="row">
                    <div class="col-md-3">
                        <!-- Visualization Controls -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title">Visualization Settings</h5>
                            </div>
                            <div class="card-body">
                                <form id="visualization-form">
                                    <div class="mb-3">
                                        <label for="viz-type" class="form-label">Visualization Type</label>
                                        <select id="viz-type" class="form-select">
                                            <option value="all">All Documents</option>
                                            <option value="query">Query Results</option>
                                        </select>
                                    </div>
                                    
                                    <div id="query-viz-options">
                                        <div class="mb-3">
                                            <label for="viz-query" class="form-label">Query</label>
                                            <input type="text" class="form-control" id="viz-query" placeholder="Enter your query...">
                                        </div>
                                        <div class="mb-3">
                                            <label for="viz-seed" class="form-label">Seed Document (Optional)</label>
                                            <select id="viz-seed" class="form-select">
                                                <option value="">No seed document</option>
                                                <!-- Documents will be loaded here -->
                                            </select>
                                        </div>
                                    </div>
                                    
                                    <div class="mb-3">
                                        <label for="viz-method" class="form-label">Dimension Reduction</label>
                                        <select id="viz-method" class="form-select">
                                            <option value="tsne">t-SNE</option>
                                            <option value="umap">UMAP</option>
                                        </select>
                                    </div>
                                    
                                    <div id="tsne-options">
                                        <div class="mb-3">
                                            <label for="perplexity" class="form-label">Perplexity: <span id="perplexity-value">30</span></label>
                                            <input type="range" class="form-range" min="5" max="50" step="5" value="30" id="perplexity">
                                        </div>
                                    </div>
                                    
                                    <div id="umap-options" style="display: none;">
                                        <div class="mb-3">
                                            <label for="n-neighbors" class="form-label">Neighbors: <span id="n-neighbors-value">15</span></label>
                                            <input type="range" class="form-range" min="5" max="50" step="5" value="15" id="n-neighbors">
                                        </div>
                                    </div>
                                    
                                    <button type="submit" class="btn btn-primary">Generate Visualization</button>
                                </form>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-9">
                        <!-- Visualization Display -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title">Embedding Visualization</h5>
                            </div>
                            <div class="card-body">
                                <div id="embedding-plot" style="height: 600px;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Session Management Tab -->
            <div class="tab-pane fade" id="sessions" role="tabpanel" aria-labelledby="sessions-tab">
                <div class="row">
                    <div class="col-md-4">
                        <!-- Session Controls -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title">Session Management</h5>
                            </div>
                            <div class="card-body">
                                <button id="create-session" class="btn btn-success mb-3">Create New Session</button>
                                
                                <div class="mb-3">
                                    <label for="session-name" class="form-label">Session Name</label>
                                    <input type="text" class="form-control" id="session-name" placeholder="My Session">
                                </div>
                                
                                <div class="d-grid gap-2">
                                    <button id="export-session" class="btn btn-outline-primary">Export Current Session</button>
                                    <button id="import-session" class="btn btn-outline-secondary">Import Session</button>
                                </div>
                                
                                <hr>
                                
                                <h6>Saved Sessions</h6>
                                <div class="list-group" id="session-list">
                                    <!-- Sessions will be listed here -->
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-8">
                        <!-- Session Content -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title" id="current-session-name">No Session Selected</h5>
                            </div>
                            <div class="card-body">
                                <div id="session-queries" class="mb-4">
                                    <h6>Queries</h6>
                                    <div class="list-group" id="query-list">
                                        <!-- Queries will be listed here -->
                                    </div>
                                </div>
                                
                                <div id="session-results">
                                    <h6>Results</h6>
                                    <div id="result-container">
                                        <!-- Selected query results will be shown here -->
                                        <p class="text-muted">Select a query to view results</p>
                                    </div>
                                </div>
                                
                                <div id="session-metrics" class="mt-4">
                                    <h6>Metrics</h6>
                                    <div id="metrics-container">
                                        <!-- Selected query metrics will be shown here -->
                                    </div>
                                </div>
                                
                                <div id="session-visualization" class="mt-4">
                                    <h6>Visualizations</h6>
                                    <div id="visualization-container">
                                        <!-- Selected query visualization will be shown here -->
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap and custom scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/charts.js"></script>
    <script src="/static/js/visualizations.js"></script>
    <script src="/static/js/session-manager.js"></script>
    <script src="/static/js/main.js"></script>
</body>
</html>

            <!-- Session Management Tab -->
            <div class="tab-pane fade" id="sessions" role="tabpanel" aria-labelledby="sessions-tab">
                <div class="row">
                    <div class="col-md-4">
                        <!-- Session Controls -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title">Session Management</h5>
                            </div>
                            <div class="card-body">
                                <button id="create-session" class="btn btn-success mb-3">Create New Session</button>
                                
                                <div class="mb-3">
                                    <label for="session-name" class="form-label">Session Name</label>
                                    <input type="text" class="form-control" id="session-name" placeholder="My Session">
                                </div>
                                
                                <div class="d-grid gap-2">
                                    <button id="export-session" class="btn btn-outline-primary">Export Current Session</button>
                                    <button id="import-session" class="btn btn-outline-secondary">Import Session</button>
                                </div>
                                
                                <hr>
                                
                                <h6>Saved Sessions</h6>
                                <div class="list-group" id="session-list">
                                    <!-- Sessions will be listed here -->
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-8">
                        <!-- Session Content -->
                        <div class="card">
                            <div class="card-header">
                                <h5 class="card-title" id="current-session-name">No Session Selected</h5>
                            </div>
                            <div class="card-body">
                                <div id="session-queries" class="mb-4">
                                    <h6>Queries</h6>
                                    <div class="list-group" id="query-list">
                                        <!-- Queries will be listed here -->
                                    </div>
                                </div>
                                
                                <div id="session-results">
                                    <h6>Results</h6>
                                    <div id="result-container">
                                        <!-- Selected query results will be shown here -->
                                        <p class="text-muted">Select a query to view results</p>
                                    </div>
                                </div>
                                
                                <div id="session-metrics" class="mt-4">
                                    <h6>Metrics</h6>
                                    <div id="metrics-container">
                                        <!-- Selected query metrics will be shown here -->
                                    </div>
                                </div>
                                
                                <div id="session-visualization" class="mt-4">
                                    <h6>Visualizations</h6>
                                    <div id="visualization-container">
                                        <!-- Selected query visualization will be shown here -->
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap and custom scripts -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/static/js/charts.js"></script>
    <script src="/static/js/visualizations.js"></script>
    <script src="/static/js/session-manager.js"></script>
    <script src="/static/js/main.js"></script>
</body>
</html>''')
    
    # Add examples/ui to the Python path
    sys.path.append(ui_dir)
    
    # Import and run the UI with the properly set template directory
    try:
        # Modify the seeded_sphere_search_tester.py to ensure it works with our templates
        tester_path = os.path.join(ui_dir, "seeded_sphere_search_tester.py")
        if os.path.exists(tester_path):
            from seeded_sphere_search_tester import app, load_model, load_ellipsoidal_transformer
            
            # Explicitly set the template folder for Flask
            app.template_folder = templates_dir
            
            # Load the model
            if not load_model(args.model):
                print(f"Failed to load model from {args.model}")
                sys.exit(1)
            
            # Try to load ellipsoidal transformation
            load_ellipsoidal_transformer()
            
            # Run the app
            print(f"Starting server on http://localhost:{args.port}")
            app.run(host='0.0.0.0', port=args.port, debug=True)
        else:
            print(f"Error: Could not find {tester_path}")
            sys.exit(1)
    except ImportError as e:
        print(f"Error importing UI module: {e}")
        print("Make sure you have run:")
        print("  pip install flask flask-cors")
        sys.exit(1)
    except Exception as e:
        print(f"Error starting UI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 