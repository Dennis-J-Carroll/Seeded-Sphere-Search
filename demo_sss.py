"""
Demo Script for Seeded Sphere Search System

This script demonstrates the main features of the Seeded Sphere Search system,
including embedding generation, relationship mapping, echo refinement,
and search with both ellipsoidal and hyperbolic transformations.
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from SeededSphereSearch import SeededSphereSearch
from EllipsoidalTransformation import EllipsoidalTransformation
from HyperbolicTransformation import HyperbolicTransformation
from HybridSearchEngine import HybridSearchEngine
from enhanced_evaluation import evaluate_search_performance, generate_test_data

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
            "max_echo_layers": 5,
            "frequency_scale": 0.325,
            "delta": 0.1625,
            "alpha": 0.5,
            "use_pretrained": True,
            "pretrained_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
    
    sss = SeededSphereSearch(config)
    sss.initialize(corpus)
    
    return sss

def save_model(model, filename='sss_model.pkl'):
    """Save model to a file"""
    print(f"Saving model to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("Model saved successfully")

def load_model(filename='sss_model.pkl'):
    """Load model from a file"""
    print(f"Loading model from {filename}...")
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
    return model

def demo_basic_search(search_engine, queries):
    """Demonstrate basic search functionality"""
    print("\n===== Basic Search Demonstration =====")
    
    for query in queries:
        print(f"\nQuery: {query}")
        
        start_time = time.time()
        results = search_engine.search(query, top_k=5)
        elapsed_time = time.time() - start_time
        
        print(f"Found {len(results)} results in {elapsed_time:.4f} seconds:")
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")

def train_ellipsoidal_transformer(search_engine, test_data):
    """Train and demonstrate ellipsoidal transformation"""
    print("\n===== Ellipsoidal Transformation =====")
    
    # Initialize ellipsoidal transformation
    dimensions = search_engine.config["dimensions"]
    transformer = EllipsoidalTransformation(dimensions)
    
    # Prepare training data
    query_embeddings = []
    relevant_doc_embeddings = []
    non_relevant_doc_embeddings = []
    
    for query in test_data["queries"]:
        # Encode query
        query_embedding = search_engine._encode_query(query)
        query_embeddings.append(query_embedding)
        
        # Get relevant and non-relevant document embeddings
        relevant_ids = test_data["relevance"].get(query, [])
        
        # Collect relevant document embeddings
        rel_embeddings = []
        for doc_id in relevant_ids:
            if doc_id in search_engine.refined_embeddings:
                rel_embeddings.append(search_engine.refined_embeddings[doc_id])
        relevant_doc_embeddings.append(rel_embeddings)
        
        # Collect non-relevant document embeddings (sample 10)
        non_rel_embeddings = []
        non_rel_count = 0
        for doc_id in search_engine.refined_embeddings:
            if doc_id not in relevant_ids:
                non_rel_embeddings.append(search_engine.refined_embeddings[doc_id])
                non_rel_count += 1
                if non_rel_count >= 10:
                    break
        non_relevant_doc_embeddings.append(non_rel_embeddings)
    
    # Train ellipsoidal transformation
    transformer.train_weights(
        query_embeddings,
        relevant_doc_embeddings,
        non_relevant_doc_embeddings,
        learning_rate=0.01,
        max_epochs=50
    )
    
    # Save the weights
    transformer.save("ellipsoidal_weights.npy")
    
    # Demonstrate search with ellipsoidal transformation
    print("\nSearch with Ellipsoidal Transformation:")
    for query in test_data["queries"][:3]:  # Test with first 3 queries
        print(f"\nQuery: {query}")
        
        # Standard search
        standard_results = search_engine.search(query, top_k=5)
        
        # Ellipsoidal search
        ellipsoidal_results = search_engine.search(
            query, 
            top_k=5,
            use_ellipsoidal=True, 
            ellipsoidal_transformer=transformer
        )
        
        # Compare results
        print("Standard Search Results:")
        for i, result in enumerate(standard_results):
            print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
        
        print("\nEllipsoidal Search Results:")
        for i, result in enumerate(ellipsoidal_results):
            print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
    
    return transformer

def demo_hyperbolic_transformation(search_engine, test_data):
    """Demonstrate hyperbolic transformation"""
    print("\n===== Hyperbolic Transformation =====")
    
    # Initialize hyperbolic transformation
    dimensions = search_engine.config["dimensions"]
    hyperbolic_transformer = HyperbolicTransformation(dimensions)
    
    # Generate hierarchical relationships from existing relationships
    print("Extracting hierarchical relationships...")
    hierarchical_relationships = []
    
    # Process top documents and their neighbors
    for doc_id in tqdm(list(search_engine.vocabulary.keys())[:100]):
        # Get top neighbors
        neighbors = sorted(
            search_engine.relationships[doc_id].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]  # Top 5 neighbors
        
        for neighbor_id, score in neighbors:
            hierarchical_relationships.append((doc_id, neighbor_id, score))
    
    # Transform embeddings
    print("Transforming embeddings to hyperbolic space...")
    hyperbolic_embeddings = hyperbolic_transformer.transform_embeddings(
        {doc_id: search_engine.refined_embeddings[doc_id] 
         for doc_id in list(search_engine.refined_embeddings.keys())[:100]}
    )
    
    # Train embeddings if desired
    print("Training hyperbolic embeddings...")
    trained_embeddings = hyperbolic_transformer.train_embeddings(
        hyperbolic_embeddings,
        hierarchical_relationships,
        learning_rate=0.01,
        iterations=20
    )
    
    # Visualize embeddings
    print("Visualizing hyperbolic embeddings...")
    labels = {doc_id: search_engine.vocabulary[doc_id]["title"][:20] 
              for doc_id in trained_embeddings}
    hyperbolic_transformer.visualize_embeddings(
        trained_embeddings,
        labels=labels,
        filename="hyperbolic_embeddings.png"
    )
    
    print("Hyperbolic transformation demonstration complete. See hyperbolic_embeddings.png for visualization.")

def demo_hybrid_search(search_engine, test_data):
    """Demonstrate hybrid search"""
    print("\n===== Hybrid Search Demonstration =====")
    
    # Initialize hybrid search engine
    hybrid_engine = HybridSearchEngine({
        "sss_weight": 0.7,
        "transformer_weight": 0.3,
        "bm25_weight": 0.0,
        "ensemble_method": "weighted_sum"
    })
    
    # Add search engines
    hybrid_engine.add_search_engine("sss", search_engine)
    
    # Create a simple transformer search engine (using SSS as a proxy)
    class TransformerSearchEngine:
        def __init__(self, sss_engine):
            self.sss = sss_engine
        
        def search(self, query, top_k=10):
            # Skip echo refinement, use initial embeddings only
            query_embedding = self.sss._encode_query(query)
            
            # Score against initial embeddings
            scores = []
            for doc_id, embedding in self.sss.embeddings.items():
                dot_product = np.dot(query_embedding, embedding)
                dot_product = max(-1.0, min(1.0, dot_product))
                distance = np.arccos(dot_product)
                score = np.exp(-self.sss.config["alpha"] * distance)
                
                scores.append({
                    "id": doc_id,
                    "title": self.sss.vocabulary[doc_id]["title"],
                    "score": score,
                    "distance": distance
                })
            
            scores.sort(key=lambda x: x["score"], reverse=True)
            return scores[:top_k]
    
    transformer_engine = TransformerSearchEngine(search_engine)
    hybrid_engine.add_search_engine("transformer", transformer_engine)
    
    # Demonstrate hybrid search
    for query in test_data["queries"][:3]:  # Test with first 3 queries
        print(f"\nQuery: {query}")
        
        # Standard SSS search
        sss_results = search_engine.search(query, top_k=5)
        
        # Transformer-only search
        transformer_results = transformer_engine.search(query, top_k=5)
        
        # Hybrid search
        hybrid_results = hybrid_engine.search(query, top_k=5)
        
        # Display results
        print("SSS Search Results:")
        for i, result in enumerate(sss_results):
            print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
        
        print("\nTransformer Search Results:")
        for i, result in enumerate(transformer_results):
            print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
        
        print("\nHybrid Search Results:")
        for i, result in enumerate(hybrid_results):
            print(f"{i+1}. {result['title']} (Score: {result['score']:.4f})")
        
        # Explain top hybrid result
        if hybrid_results:
            print("\nExplanation for top hybrid result:")
            explanation = hybrid_engine.explain_search_results(hybrid_results[0])
            print(explanation)
    
    # Optimize weights
    print("\nOptimizing hybrid weights...")
    optimized_weights = hybrid_engine.optimize_weights(
        test_data["queries"],
        test_data["relevance"],
        metric="ndcg"
    )
    
    print(f"Optimized weights: {optimized_weights}")

def demo_performance_evaluation(search_engine, test_data):
    """Demonstrate performance evaluation"""
    print("\n===== Performance Evaluation =====")
    
    # Evaluate performance
    metrics = evaluate_search_performance(search_engine, test_data)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot metrics by k
    ks = [5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_to_plot = ['precision', 'recall', 'ndcg', 'mrr']
    markers = ['o', 's', '^', 'x']
    
    for i, metric_name in enumerate(metrics_to_plot):
        values = [metrics.get(f"{metric_name}@{k}", 0) for k in ks]
        ax.plot(ks, values, marker=markers[i], label=metric_name.upper())
    
    ax.set_xlabel('k')
    ax.set_ylabel('Score')
    ax.set_title('Search Performance Metrics')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("search_performance.png")
    print("Performance plot saved to search_performance.png")

def main():
    """Main demonstration script"""
    # Check if model exists, otherwise create it
    import os
    if os.path.exists('sss_model.pkl'):
        # Load existing model
        search_engine = load_model()
    else:
        # Create new model
        corpus = load_corpus()
        search_engine = initialize_search_engine(corpus)
        save_model(search_engine)
    
    # Generate test data
    test_data = generate_test_data(search_engine, num_queries=10, num_relevant=5)
    
    # Run demonstrations
    queries = [
        "space exploration with alien technology",
        "time travel paradoxes and alternate timelines",
        "artificial intelligence becoming sentient",
        "interstellar war between human colonies",
        "genetic engineering creating new species"
    ]
    
    demo_basic_search(search_engine, queries)
    
    ellipsoidal_transformer = train_ellipsoidal_transformer(search_engine, test_data)
    
    demo_hyperbolic_transformation(search_engine, test_data)
    
    demo_hybrid_search(search_engine, test_data)
    
    demo_performance_evaluation(search_engine, test_data)
    
    print("\nDemonstration complete!")

if __name__ == "__main__":
    main()
