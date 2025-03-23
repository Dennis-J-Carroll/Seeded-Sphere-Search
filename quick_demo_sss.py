"""
Quick Demo Script for Seeded Sphere Search System

This script demonstrates the core features of the Seeded Sphere Search system
with a focus on quick execution and essential functionality.
"""

import json
import pickle
import numpy as np
import os
import time
from tqdm import tqdm

from SeededSphereSearch import SeededSphereSearch

def load_corpus(file_path='scifi_corpus.json'):
    """Load document corpus from a JSON file"""
    print(f"Loading corpus from {file_path}...")
    with open(file_path, 'r') as f:
        corpus = json.load(f)
    print(f"Loaded {len(corpus)} documents")
    return corpus

def initialize_search_engine(corpus, config=None):
    """Initialize the Seeded Sphere Search engine with the corpus"""
    print("Initializing Seeded Sphere Search engine...")
    
    if config is None:
        config = {
            "dimensions": 384,
            "min_echo_layers": 1,
            "max_echo_layers": 3,  # Reduced for quicker demo
            "frequency_scale": 0.325,
            "delta": 0.1625,
            "alpha": 0.5,
            "use_pretrained": True,
            "pretrained_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    
    sss = SeededSphereSearch(config)
    
    # Limit corpus size for quick demo if needed
    if len(corpus) > 1000:
        print(f"Limiting corpus to 1000 documents for quick demo (original size: {len(corpus)})")
        limited_corpus = {}
        for i, (doc_id, doc) in enumerate(corpus.items()):
            if i < 1000:
                limited_corpus[doc_id] = doc
            else:
                break
        corpus = limited_corpus
    
    # Convert JSON corpus to the format expected by SeededSphereSearch
    formatted_corpus = []
    for doc_id, doc_data in corpus.items():
        formatted_corpus.append({
            "id": doc_id,
            "title": doc_data["title"],
            "content": doc_data["content"],
            "metadata": doc_data.get("source", "")
        })
    
    sss.initialize(formatted_corpus)
    return sss

def save_model(model, filename='sss_model_quick.pkl'):
    """Save model to a file"""
    print(f"Saving model to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully")

def load_model(filename='sss_model_quick.pkl'):
    """Load model from a file"""
    print(f"Loading model from {filename}...")
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
    return model

def demo_search(search_engine, queries):
    """Demonstrate search functionality"""
    print("\n===== Search Demonstration =====")
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        start_time = time.time()
        results = search_engine.search(query, top_k=5)
        elapsed_time = time.time() - start_time
        
        print(f"Found {len(results)} results in {elapsed_time:.4f} seconds:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
            # Show a snippet of the content
            content = result.get('content', search_engine.vocabulary[result['id']].get('content', ''))
            if content:
                snippet = content[:150] + "..." if len(content) > 150 else content
                print(f"   Snippet: {snippet}")

def main():
    """Main demonstration script"""
    # Check if model exists, otherwise create it
    model_path = 'sss_model_quick.pkl'
    
    if os.path.exists(model_path):
        # Load existing model
        search_engine = load_model(model_path)
    else:
        # Create new model
        corpus = load_corpus()
        search_engine = initialize_search_engine(corpus)
        save_model(search_engine, model_path)
    
    # Define example queries
    queries = [
        "space exploration with alien technology",
        "time travel paradoxes",
        "artificial intelligence sentience",
        "vampire horror stories",
        "dystopian future society"
    ]
    
    # Run search demonstration
    demo_search(search_engine, queries)
    
    print("\nQuick demonstration complete!")

if __name__ == "__main__":
    main()
