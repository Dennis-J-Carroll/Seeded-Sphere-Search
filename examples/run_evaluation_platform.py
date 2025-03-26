#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Seeded Sphere Search Evaluation Platform Launcher

This script launches the enhanced UI with evaluation capabilities.
"""

import os
import sys
import argparse
import subprocess

def main():
    """Main function to launch the evaluation platform"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Launch Seeded Sphere Search Evaluation Platform')
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
    
    # Copy enhanced template if needed
    src_dir = os.path.dirname(os.path.abspath(__file__))
    ui_dir = os.path.join(src_dir, 'examples', 'ui')
    templates_dir = os.path.join(ui_dir, 'templates')
    
    # Create templates directory if it doesn't exist
    os.makedirs(templates_dir, exist_ok=True)
    
    # Check if static directory exists, create if needed
    static_dir = os.path.join(ui_dir, 'static')
    js_dir = os.path.join(static_dir, 'js')
    os.makedirs(js_dir, exist_ok=True)
    
    # Check if dependencies are installed
    try:
        import flask
        import numpy
        import sklearn
        import umap
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required dependencies:")
        print("  pip install flask numpy scikit-learn umap-learn plotly")
        return 1
    
    # Launch the server
    server_script = os.path.join(ui_dir, 'seeded_sphere_search_tester.py')
    
    cmd = [
        sys.executable,
        server_script,
        '--model', args.model,
        '--weights', args.weights,
        '--host', args.host,
        '--port', str(args.port)
    ]
    
    if args.debug:
        cmd.append('--debug')
    
    print(f"Launching Seeded Sphere Search Evaluation Platform at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 