import os
import sys
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
import time
import tempfile
import werkzeug.utils
from werkzeug.utils import secure_filename

# Add parent directory to path to import seededsphere
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import Seeded Sphere Search modules
from seededsphere import SeededSphereSearch
from seededsphere.transformations import EllipsoidalTransformation

# Import evaluation modules
from analytics_processor import AnalyticsProcessor
from embedding_visualizer import EmbeddingVisualizer
from session_manager import SessionManager

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables
search_engine = None
ellipsoidal_transformer = None
analytics_processor = None
embedding_visualizer = None
session_manager = None

# Set up folders for templates and static files
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

# Create directories if they don't exist
os.makedirs(template_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, 'js'), exist_ok=True)
os.makedirs(os.path.join(static_dir, 'css'), exist_ok=True)

# Configure app
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

def load_model(model_path='sss_model.pkl'):
    """
    Load the Seeded Sphere Search model
    
    Args:
        model_path: Path to the model file
        
    Returns:
        SeededSphereSearch: Loaded model or None if loading fails
    """
    global search_engine
    
    try:
        if not os.path.isfile(model_path):
            print(f"Model file not found: {model_path}")
            return None
            
        with open(model_path, 'rb') as f:
            search_engine = pickle.load(f)
            
        print(f"Model loaded from {model_path}")
        print(f"Documents in corpus: {len(search_engine.vocabulary)}")
        
        return search_engine
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
        
def load_ellipsoidal_transformer(weights_path='ellipsoidal_weights.npy'):
    """
    Load the ellipsoidal transformer weights
    
    Args:
        weights_path: Path to the weights file
        
    Returns:
        EllipsoidalTransformation: Transformer or None if loading fails
    """
    global ellipsoidal_transformer
    
    try:
        if not os.path.isfile(weights_path):
            print(f"Weights file not found: {weights_path}")
            return None
            
        weights = np.load(weights_path)
        dimensions = search_engine.model.get_dimension()
        ellipsoidal_transformer = EllipsoidalTransformation(dimensions)
        ellipsoidal_transformer.weights = weights
        
        print(f"Ellipsoidal weights loaded from {weights_path}")
        return ellipsoidal_transformer
    except Exception as e:
        print(f"Error loading ellipsoidal weights: {e}")
        return None

def init_evaluation_components():
    """Initialize components for the evaluation platform"""
    global analytics_processor, embedding_visualizer, session_manager
    
    # Only initialize if search engine is loaded
    if search_engine is not None:
        try:
            # Initialize analytics processor
            analytics_processor = AnalyticsProcessor(search_engine)
            
            # Initialize embedding visualizer
            embedding_visualizer = EmbeddingVisualizer(search_engine)
            
            # Initialize session manager
            sessions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sessions')
            os.makedirs(sessions_dir, exist_ok=True)
            session_manager = SessionManager(sessions_dir)
            
            print("Evaluation components initialized")
            return True
        except Exception as e:
            print(f"Error initializing evaluation components: {e}")
            return False
    
    return False

@app.route('/')
def index():
    """Render the main page"""
    try:
        # Check if enhanced template exists
        if os.path.exists(os.path.join(template_dir, 'index.html')):
            return render_template('index.html')
        else:
            # Create a basic index.html with an error message
            index_html = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Seeded Sphere Search</title>
                <script src="https://cdn.tailwindcss.com"></script>
            </head>
            <body class="bg-gray-50">
                <div class="max-w-7xl mx-auto px-4 py-8">
                    <div class="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
                        <strong class="font-bold">Error!</strong>
                        <span class="block sm:inline">Template file not found. Please ensure the template directory contains index.html.</span>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Write the file
            with open(os.path.join(template_dir, 'index.html'), 'w') as f:
                f.write(index_html)
                
            return index_html
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/api/search', methods=['POST'])
def search():
    """
    Perform a search with the given parameters
    
    Request JSON:
    {
        "query": "search query",
        "seed_doc_id": "optional seed document ID",
        "threshold": 0.7,
        "max_results": 10,
        "use_ellipsoidal": false,
        "session_id": "optional session ID to save results"
    }
    """
    if search_engine is None:
        return jsonify({"success": False, "error": "Search engine not initialized"})
    
    try:
        # Get request data
        data = request.json
        query = data.get('query', '')
        seed_doc_id = data.get('seed_doc_id', '')
        threshold = float(data.get('threshold', 0.7))
        max_results = int(data.get('max_results', 10))
        use_ellipsoidal = data.get('use_ellipsoidal', False)
        session_id = data.get('session_id', '')
        
        # Validate query
        if not query:
            return jsonify({"success": False, "error": "Query is required"})
            
        # Start timing
        start_time = time.time()
        
        # Perform search based on parameters
        if seed_doc_id and seed_doc_id in search_engine.vocabulary:
            # Seeded search
            results = search_engine.seeded_search(
                query=query,
                seed_doc_id=seed_doc_id,
                top_k=max_results
            )
        else:
            # Standard search
            results = search_engine.search(
                query=query,
                top_k=max_results
            )
            
        # Apply ellipsoidal transformation if requested
        if use_ellipsoidal and ellipsoidal_transformer is not None:
            # Get query embedding
            query_embedding = search_engine._encode_query(query)
            
            # Transform query embedding
            transformed_query = ellipsoidal_transformer.transform_query(query_embedding)
            
            # Calculate similarities with transformed query
            results_with_scores = []
            for doc_id in search_engine.vocabulary:
                if doc_id in search_engine.refined_embeddings:
                    doc_embedding = search_engine.refined_embeddings[doc_id]
                    score = search_engine._calculate_similarity(transformed_query, doc_embedding)
                    
                    if score >= threshold:
                        doc_info = search_engine.vocabulary[doc_id]
                        results_with_scores.append({
                            "id": doc_id,
                            "title": doc_info.get("title", "Untitled"),
                            "content": doc_info.get("content", "")[:500],
                            "tags": doc_info.get("tags", []),
                            "score": float(score)
                        })
            
            # Sort by score in descending order
            results = sorted(results_with_scores, key=lambda x: x["score"], reverse=True)[:max_results]
        else:
            # Filter results by threshold
            results = [r for r in results if r.get("score", r.get("similarity", 0)) >= threshold]
            
        # End timing
        execution_time = time.time() - start_time
            
        # Save to session if requested
        if session_id and session_manager is not None:
            session = session_manager.get_session(session_id)
            if session:
                metadata = {
                    "threshold": threshold,
                    "max_results": max_results,
                    "use_ellipsoidal": use_ellipsoidal,
                    "seed_doc_id": seed_doc_id if seed_doc_id else None,
                    "execution_time": execution_time
                }
                
                session.add_query_results(query, results, metadata)
                
        # Return results
        return jsonify({
            "success": True,
            "results": results,
            "execution_time": execution_time
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all documents in the corpus"""
    if search_engine is None:
        return jsonify({"success": False, "error": "Search engine not initialized"})
        
    try:
        documents = []
        
        # Get all documents
        for doc_id, doc_info in search_engine.vocabulary.items():
            documents.append({
                "id": doc_id,
                "title": doc_info.get("title", "Untitled"),
                "tags": doc_info.get("tags", [])
            })
            
        return jsonify({
            "success": True,
            "documents": documents,
            "count": len(documents)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/transformations', methods=['GET'])
def get_transformations():
    """Get information about available transformations"""
    try:
        # Check if ellipsoidal transformation is available
        ellipsoidal_available = ellipsoidal_transformer is not None
        
        return jsonify({
            "success": True,
            "ellipsoidal_available": ellipsoidal_available
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/analytics/ground-truth', methods=['POST'])
def create_ground_truth():
    """
    Create ground truth data for queries
    
    Request JSON:
    {
        "queries": ["query1", "query2", ...],
        "output_file": "optional file path to save ground truth"
    }
    """
    if search_engine is None or analytics_processor is None:
        return jsonify({"success": False, "error": "Search engine or analytics processor not initialized"})
        
    try:
        # Get request data
        data = request.json
        queries = data.get('queries', [])
        output_file = data.get('output_file', None)
        
        if not queries:
            return jsonify({"success": False, "error": "No queries provided"})
            
        # Create ground truth
        if output_file:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
        ground_truth = analytics_processor.create_ground_truth(queries, output_file)
        
        return jsonify({
            "success": True,
            "ground_truth": ground_truth,
            "queries": queries,
            "output_file": output_file
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/analytics/metrics', methods=['POST'])
def calculate_metrics():
    """
    Calculate metrics for search results
    
    Request JSON:
    {
        "query": "search query",
        "ground_truth": ["doc_id1", "doc_id2", ...] or null,
        "ground_truth_file": "optional path to ground truth file",
        "session_id": "optional session ID to save results"
    }
    """
    if search_engine is None or analytics_processor is None:
        return jsonify({"success": False, "error": "Search engine or analytics processor not initialized"})
        
    try:
        # Get request data
        data = request.json
        query = data.get('query', '')
        ground_truth = data.get('ground_truth')
        ground_truth_file = data.get('ground_truth_file')
        session_id = data.get('session_id', '')
        
        if not query:
            return jsonify({"success": False, "error": "Query is required"})
            
        # Load ground truth from file if provided
        if ground_truth_file and not ground_truth:
            if not analytics_processor.load_ground_truth(ground_truth_file):
                return jsonify({"success": False, "error": "Failed to load ground truth file"})
                
        # Perform search to get results
        results = search_engine.search(query, top_k=50)
        
        # Calculate metrics
        metrics = analytics_processor.calculate_metrics(query, results, ground_truth)
        
        if "error" in metrics:
            return jsonify({"success": False, "error": metrics["error"]})
            
        # Generate precision-recall curve
        curve_data = analytics_processor.generate_precision_recall_curve(query, results, ground_truth)
        
        if not curve_data["success"]:
            return jsonify({"success": False, "error": curve_data["error"]})
            
        # Save to session if requested
        if session_id and session_manager is not None:
            session = session_manager.get_session(session_id)
            if session:
                # Save search results
                result_id = session.add_query_results(query, results, {
                    "type": "analytics",
                    "ground_truth": ground_truth
                })
                
                # Save analytics
                session.add_analytics(query, metrics, result_id)
                
        # Return results
        return jsonify({
            "success": True,
            "metrics": metrics,
            "curve_data": curve_data["curve_data"]
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    """
    Compare different search algorithms
    
    Request JSON:
    {
        "query": "search query",
        "algorithms": ["standard", "seeded", "ellipsoidal"],
        "seed_doc_id": "optional seed document ID for seeded search",
        "ground_truth": ["doc_id1", "doc_id2", ...] or null,
        "session_id": "optional session ID to save results"
    }
    """
    if search_engine is None or analytics_processor is None:
        return jsonify({"success": False, "error": "Search engine or analytics processor not initialized"})
        
    try:
        # Get request data
        data = request.json
        query = data.get('query', '')
        algorithms = data.get('algorithms', ['standard'])
        seed_doc_id = data.get('seed_doc_id', '')
        ground_truth = data.get('ground_truth')
        session_id = data.get('session_id', '')
        
        if not query:
            return jsonify({"success": False, "error": "Query is required"})
            
        # Validate algorithms
        valid_algorithms = ["standard", "seeded", "ellipsoidal"]
        algorithms = [a for a in algorithms if a in valid_algorithms]
        
        if not algorithms:
            return jsonify({"success": False, "error": "No valid algorithms specified"})
            
        # Check if seeded search can be performed
        if "seeded" in algorithms and (not seed_doc_id or seed_doc_id not in search_engine.vocabulary):
            return jsonify({"success": False, "error": "Invalid seed document for seeded search"})
            
        # Check if ellipsoidal transformation can be performed
        if "ellipsoidal" in algorithms and ellipsoidal_transformer is None:
            return jsonify({"success": False, "error": "Ellipsoidal transformation not available"})
            
        # Store comparison results
        comparison_results = []
        
        # Run each algorithm
        for algorithm in algorithms:
            start_time = time.time()
            
            if algorithm == "standard":
                # Standard search
                results = search_engine.search(query, top_k=50)
                algorithm_name = "Standard Search"
                
            elif algorithm == "seeded":
                # Seeded search
                results = search_engine.seeded_search(
                    query=query,
                    seed_doc_id=seed_doc_id,
                    top_k=50
                )
                algorithm_name = f"Seeded Search (Seed: {search_engine.vocabulary[seed_doc_id].get('title', seed_doc_id)})"
                
            elif algorithm == "ellipsoidal":
                # Ellipsoidal transformation
                # Get query embedding
                query_embedding = search_engine._encode_query(query)
                
                # Transform query embedding
                transformed_query = ellipsoidal_transformer.transform_query(query_embedding)
                
                # Calculate similarities with transformed query
                results_with_scores = []
                for doc_id in search_engine.vocabulary:
                    if doc_id in search_engine.refined_embeddings:
                        doc_embedding = search_engine.refined_embeddings[doc_id]
                        score = search_engine._calculate_similarity(transformed_query, doc_embedding)
                        
                        doc_info = search_engine.vocabulary[doc_id]
                        results_with_scores.append({
                            "id": doc_id,
                            "title": doc_info.get("title", "Untitled"),
                            "content": doc_info.get("content", "")[:500],
                            "tags": doc_info.get("tags", []),
                            "score": float(score)
                        })
                
                # Sort by score in descending order
                results = sorted(results_with_scores, key=lambda x: x["score"], reverse=True)[:50]
                algorithm_name = "Ellipsoidal Transformation"
                
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Calculate metrics
            metrics = analytics_processor.calculate_metrics(query, results, ground_truth)
            
            # Add to comparison results
            comparison_results.append({
                "algorithm": algorithm,
                "name": algorithm_name,
                "results": results,
                "execution_time": execution_time,
                "metrics": metrics
            })
            
        # Save to session if requested
        if session_id and session_manager is not None:
            session = session_manager.get_session(session_id)
            if session:
                session.add_comparison(query, {
                    "algorithms": algorithms,
                    "seed_doc_id": seed_doc_id if seed_doc_id else None,
                    "results": comparison_results
                })
                
        # Return results
        return jsonify({
            "success": True,
            "comparison": comparison_results,
            "query": query
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/visualize/embeddings', methods=['POST'])
def visualize_embeddings():
    """
    Visualize embeddings
    
    Request JSON:
    {
        "method": "tsne", "pca", or "umap",
        "type": "all" or "query",
        "query": "search query", (required if type is "query")
        "dimensions": 2 or 3,
        "session_id": "optional session ID to save results"
    }
    """
    if search_engine is None or embedding_visualizer is None:
        return jsonify({"success": False, "error": "Search engine or embedding visualizer not initialized"})
        
    try:
        # Get request data
        data = request.json
        method = data.get('method', 'tsne')
        vis_type = data.get('type', 'all')
        query = data.get('query', '')
        dimensions = int(data.get('dimensions', 2))
        session_id = data.get('session_id', '')
        
        # Validate parameters
        if method not in ['tsne', 'pca', 'umap']:
            return jsonify({"success": False, "error": "Invalid visualization method"})
            
        if vis_type not in ['all', 'query']:
            return jsonify({"success": False, "error": "Invalid visualization type"})
            
        if vis_type == 'query' and not query:
            return jsonify({"success": False, "error": "Query is required for query visualization"})
            
        if dimensions not in [2, 3]:
            return jsonify({"success": False, "error": "Dimensions must be 2 or 3"})
            
        # Create visualization
        if vis_type == 'all':
            # Visualize all documents
            visualization = embedding_visualizer.visualize_embeddings(
                method=method,
                n_components=dimensions
            )
        else:
            # Visualize query results
            # Perform search to get results
            results = search_engine.search(query, top_k=50)
            
            # Create visualization
            visualization = embedding_visualizer.visualize_query_results(
                query=query,
                results=results,
                method=method,
                n_components=dimensions
            )
            
        if not visualization.get('success', False):
            return jsonify({"success": False, "error": visualization.get('error', 'Visualization failed')})
            
        # Save to session if requested
        if session_id and session_manager is not None:
            session = session_manager.get_session(session_id)
            if session:
                # If query visualization, save results first
                result_id = None
                if vis_type == 'query':
                    result_id = session.add_query_results(query, results, {
                        "type": "visualization",
                        "method": method,
                        "dimensions": dimensions
                    })
                    
                # Save visualization
                session.add_visualization(
                    query if vis_type == 'query' else 'all_documents',
                    visualization,
                    result_id
                )
                
        # Return results
        return jsonify({
            "success": True,
            "visualization": visualization,
            "type": vis_type,
            "method": method,
            "dimensions": dimensions
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

#
# Session Management API Endpoints
#

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all available sessions"""
    if session_manager is None:
        return jsonify({"success": False, "error": "Session manager not initialized"})
        
    try:
        sessions = session_manager.list_sessions()
        
        return jsonify({
            "success": True,
            "sessions": sessions,
            "count": len(sessions)
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/sessions', methods=['POST'])
def create_session():
    """
    Create a new session
    
    Request JSON:
    {
        "name": "Session name"
    }
    """
    if session_manager is None:
        return jsonify({"success": False, "error": "Session manager not initialized"})
        
    try:
        # Get request data
        data = request.json
        name = data.get('name', 'Unnamed Session')
        
        # Create session
        session = session_manager.create_session(name)
        
        return jsonify({
            "success": True,
            "session": session.to_dict()
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get details for a specific session"""
    if session_manager is None:
        return jsonify({"success": False, "error": "Session manager not initialized"})
        
    try:
        # Get session
        session = session_manager.get_session(session_id)
        
        if not session:
            return jsonify({"success": False, "error": "Session not found"})
            
        return jsonify({
            "success": True,
            "session": session.to_dict()
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a specific session"""
    if session_manager is None:
        return jsonify({"success": False, "error": "Session manager not initialized"})
        
    try:
        # Delete session
        success = session_manager.delete_session(session_id)
        
        if not success:
            return jsonify({"success": False, "error": "Failed to delete session"})
            
        return jsonify({
            "success": True,
            "message": "Session deleted successfully"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/sessions/<session_id>/export', methods=['GET'])
def export_session(session_id):
    """Export a session to a file"""
    if session_manager is None:
        return jsonify({"success": False, "error": "Session manager not initialized"})
        
    try:
        # Export session
        session_data = session_manager.export_session(session_id)
        
        if not session_data:
            return jsonify({"success": False, "error": "Failed to export session"})
            
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(session_data.encode())
        temp_file.close()
        
        # Get session details for filename
        session = session_manager.get_session(session_id)
        file_name = f"session_{secure_filename(session.name)}_{session_id[:8]}.json"
        
        # Return the file
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=file_name,
            mimetype='application/json'
        )
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/sessions/import', methods=['POST'])
def import_session():
    """Import a session from a file"""
    if session_manager is None:
        return jsonify({"success": False, "error": "Session manager not initialized"})
        
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file provided"})
            
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({"success": False, "error": "Empty file provided"})
            
        # Read the file
        session_data = file.read().decode()
        
        # Import session
        session = session_manager.import_session(session_data)
        
        if not session:
            return jsonify({"success": False, "error": "Failed to import session"})
            
        return jsonify({
            "success": True,
            "session": session.to_dict(),
            "message": "Session imported successfully"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/sessions/<session_id>/results/<result_id>', methods=['GET'])
def get_session_result(session_id, result_id):
    """Get search results from a session"""
    if session_manager is None:
        return jsonify({"success": False, "error": "Session manager not initialized"})
        
    try:
        # Get session
        session = session_manager.get_session(session_id)
        
        if not session:
            return jsonify({"success": False, "error": "Session not found"})
            
        # Get results
        results = session.get_query_results(result_id)
        
        if not results:
            return jsonify({"success": False, "error": "Results not found"})
            
        return jsonify({
            "success": True,
            "results": results
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/sessions/<session_id>/analytics/<analytics_id>', methods=['GET'])
def get_session_analytics(session_id, analytics_id):
    """Get analytics data from a session"""
    if session_manager is None:
        return jsonify({"success": False, "error": "Session manager not initialized"})
        
    try:
        # Get session
        session = session_manager.get_session(session_id)
        
        if not session:
            return jsonify({"success": False, "error": "Session not found"})
            
        # Get analytics
        analytics = session.get_analytics(analytics_id)
        
        if not analytics:
            return jsonify({"success": False, "error": "Analytics not found"})
            
        return jsonify({
            "success": True,
            "analytics": analytics
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/sessions/<session_id>/analytics', methods=['POST'])
def add_session_analytics(session_id):
    """
    Add analytics data to a session
    
    Request JSON:
    {
        "query": "search query",
        "metrics": { metrics data },
        "result_id": "optional result ID"
    }
    """
    if session_manager is None:
        return jsonify({"success": False, "error": "Session manager not initialized"})
        
    try:
        # Get session
        session = session_manager.get_session(session_id)
        
        if not session:
            return jsonify({"success": False, "error": "Session not found"})
            
        # Get request data
        data = request.json
        query = data.get('query', '')
        metrics = data.get('metrics', {})
        result_id = data.get('result_id', None)
        
        if not query or not metrics:
            return jsonify({"success": False, "error": "Query and metrics are required"})
            
        # Add analytics
        analytics_id = session.add_analytics(query, metrics, result_id)
        
        return jsonify({
            "success": True,
            "analytics_id": analytics_id,
            "message": "Analytics added successfully"
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

def main():
    """Main function to run the server"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Seeded Sphere Search UI')
    parser.add_argument('--model', type=str, default='sss_model.pkl', 
                        help='Path to the model file')
    parser.add_argument('--weights', type=str, default='ellipsoidal_weights.npy', 
                        help='Path to ellipsoidal weights file')
    parser.add_argument('--host', type=str, default='127.0.0.1', 
                        help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, 
                        help='Port to bind to')
    parser.add_argument('--debug', action='store_true', 
                        help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Load model
    if not load_model(args.model):
        print("Failed to load model. Exiting.")
        return 1
        
    # Load ellipsoidal transformer if weights file exists
    if os.path.isfile(args.weights):
        load_ellipsoidal_transformer(args.weights)
    else:
        print(f"Ellipsoidal weights file not found: {args.weights}")
        print("Ellipsoidal transformation will not be available.")
        
    # Initialize evaluation components
    init_evaluation_components()
    
    # Print setup info
    print(f"Starting server at http://{args.host}:{args.port}")
    
    # Run the app
    app.run(host=args.host, port=args.port, debug=args.debug)
    
    return 0

if __name__ == '__main__':
    main() 