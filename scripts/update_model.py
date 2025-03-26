#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Update the sss_model.pkl file to include the seeded_search method
This script loads the existing pickle file, ensures the seeded_search method
is available on the SeededSphereSearch class, and saves the model back.
"""

import os
import sys
import pickle
import numpy as np

# Add parent directory to path to import seededsphere
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the updated SeededSphereSearch class that has the seeded_search method
from seededsphere.core.seeded_sphere_search import SeededSphereSearch

def update_model(model_path='sss_model.pkl'):
    """
    Update the model pickle file to ensure it has the seeded_search method
    
    Args:
        model_path: Path to the model file
    """
    print(f"Updating model file: {model_path}")
    
    try:
        # Check if the model file exists
        if not os.path.isfile(model_path):
            print(f"Error: Model file not found: {model_path}")
            return False
            
        # Load the existing model
        with open(model_path, 'rb') as f:
            original_model = pickle.load(f)
            
        print(f"Model loaded from {model_path}")
        print(f"Documents in corpus: {len(original_model.vocabulary)}")
        
        # Create a new model instance with the same configuration
        new_model = SeededSphereSearch(original_model.config)
        
        # Transfer the state from the original model to the new model
        new_model.vocabulary = original_model.vocabulary
        new_model.word_frequencies = original_model.word_frequencies
        new_model.embeddings = original_model.embeddings
        new_model.refined_embeddings = original_model.refined_embeddings
        new_model.relationships = original_model.relationships
        new_model.initialized = original_model.initialized
        new_model.tokenizer = original_model.tokenizer
        new_model.model = original_model.model
        
        # Check if the seeded_search method is available
        if hasattr(new_model, 'seeded_search'):
            print("Confirmed that seeded_search method is available")
        else:
            print("Error: seeded_search method is still not available")
            return False
            
        # Save the updated model
        with open(model_path, 'wb') as f:
            pickle.dump(new_model, f)
            
        print(f"Updated model saved to {model_path}")
        return True
        
    except Exception as e:
        print(f"Error updating model: {e}")
        return False

if __name__ == '__main__':
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Update SSS model to include seeded_search method')
    parser.add_argument('--model', type=str, default='sss_model.pkl', 
                        help='Path to the model file')
    
    args = parser.parse_args()
    
    # Update the model
    success = update_model(args.model)
    
    if success:
        print("Model update completed successfully")
    else:
        print("Model update failed")
        sys.exit(1) 