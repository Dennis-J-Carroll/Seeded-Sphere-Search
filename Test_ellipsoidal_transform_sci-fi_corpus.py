from evaluate_search_performance import evaluate_search_performance
import SeededSphereSearch
import numpy as np
from EllipsoidalTransformation import EllipsoidalTransformation

# Create a simple corpus for testing
test_corpus = [
    {
        "id": "doc1",
        "title": "Space Exploration",
        "content": "Space exploration is the ongoing discovery and exploration of celestial structures beyond Earth's atmosphere by means of continuously evolving technology."
    },
    {
        "id": "doc2",
        "title": "Time Travel",
        "content": "Time travel is the concept of movement between certain points in time, analogous to movement between different points in space."
    },
    {
        "id": "doc3",
        "title": "Artificial Intelligence",
        "content": "Artificial intelligence is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals."
    },
    {
        "id": "doc4",
        "title": "Interstellar Colonization",
        "content": "Interstellar colonization is a hypothetical process of humans traveling to and settling on planets around other stars."
    },
    {
        "id": "doc5",
        "title": "Dystopian Future",
        "content": "A dystopia is a community or society that is undesirable or frightening, often characterized by dehumanization and totalitarian regimes."
    }
]

# Initialize the search system
sss = SeededSphereSearch.SeededSphereSearch()
sss.initialize(test_corpus)

# Create some example test queries and ground truth
# In a real scenario, you would have human relevance judgments
test_queries = [
    "space exploration with alien technology",
    "time travel paradoxes",
    "artificial intelligence uprising",
    "interstellar colonization",
    "dystopian future society"
]

# For demo purposes, we'll create synthetic ground truth
# In practice, these would come from human judgments
def create_synthetic_ground_truth(sss, test_queries):
    ground_truth = {}
    
    for query in test_queries:
        # Get top results with standard search
        results = sss.search(query, top_k=3)
        
        # Use these as "ground truth" for this example
        ground_truth[query] = [result["id"] for result in results]
    
    return ground_truth

ground_truth = create_synthetic_ground_truth(sss, test_queries)

# Evaluate baseline performance
baseline_metrics = evaluate_search_performance(sss, test_queries, ground_truth, top_k=10)
print("Baseline metrics:")
for metric, value in baseline_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Now let's train and test the ellipsoidal transformation
# First, we need to create training data
def create_training_data(sss, train_queries, ground_truth):
    query_embeddings = []
    relevant_doc_embeddings = []
    non_relevant_doc_embeddings = []
    
    for query in train_queries:
        # Encode query
        query_embedding = sss._encode_query(query)
        query_embeddings.append(query_embedding)
        
        # Get relevant doc embeddings
        relevant_ids = ground_truth.get(query, [])
        relevant_embs = [sss.refined_embeddings[doc_id] for doc_id in relevant_ids 
                         if doc_id in sss.refined_embeddings]
        relevant_doc_embeddings.append(relevant_embs)
        
        # Get some non-relevant doc embeddings (randomly sampled)
        non_relevant_ids = [doc_id for doc_id in sss.refined_embeddings 
                            if doc_id not in relevant_ids]
        
        # Sample up to 10 non-relevant docs
        import random
        sampled_non_relevant = random.sample(non_relevant_ids, 
                                           min(10, len(non_relevant_ids)))
        
        non_relevant_embs = [sss.refined_embeddings[doc_id] for doc_id in sampled_non_relevant]
        non_relevant_doc_embeddings.append(non_relevant_embs)
    
    return np.array(query_embeddings), relevant_doc_embeddings, non_relevant_doc_embeddings

# Use the same queries for training in this example
train_queries = test_queries
query_embeddings, relevant_doc_embeddings, non_relevant_doc_embeddings = create_training_data(
    sss, train_queries, ground_truth)

# Train the ellipsoidal transformation
ellipsoidal = EllipsoidalTransformation(dimensions=sss.config["dimensions"])
ellipsoidal.train_weights(
    query_embeddings, 
    relevant_doc_embeddings, 
    non_relevant_doc_embeddings,
    learning_rate=0.01,
    max_epochs=100
)

# Save the trained transformation
ellipsoidal.save('ellipsoidal_weights.npy')

# Create a modified search function that uses the ellipsoidal transformation
def ellipsoidal_search(sss, ellipsoidal, query, top_k=10):
    # Encode the query
    query_embedding = sss._encode_query(query)
    
    # Apply ellipsoidal transformation
    weighted_query = ellipsoidal.transform_query(query_embedding)
    
    # Score all documents
    scores = []
    for doc_id, embedding in sss.refined_embeddings.items():
        # Apply transformation to document embedding
        # Note: In a production system, you would precompute these
        transformed_doc = ellipsoidal.transform_document(embedding)
        
        # Calculate angular distance
        dot_product = np.dot(weighted_query, transformed_doc)
        dot_product = max(-1.0, min(1.0, dot_product))
        distance = np.arccos(dot_product)
        
        # Calculate score
        score = np.exp(-sss.config["alpha"] * distance)
        
        scores.append({
            "id": doc_id,
            "title": sss.vocabulary[doc_id]["title"],
            "score": score,
            "distance": distance
        })
    
    # Sort by score (descending)
    scores.sort(key=lambda x: x["score"], reverse=True)
    
    # Return top k results
    return scores[:top_k]

# Create a wrapper class for evaluation
class EllipsoidalSearchWrapper:
    def __init__(self, sss, ellipsoidal):
        self.sss = sss
        self.ellipsoidal = ellipsoidal
    
    def search(self, query, top_k=10):
        return ellipsoidal_search(self.sss, self.ellipsoidal, query, top_k)

# Create the wrapper
ellipsoidal_search_engine = EllipsoidalSearchWrapper(sss, ellipsoidal)

# Evaluate ellipsoidal performance
ellipsoidal_metrics = evaluate_search_performance(
    ellipsoidal_search_engine, 
    test_queries, 
    ground_truth, 
    top_k=10
)

print("\nEllipsoidal metrics:")
for metric, value in ellipsoidal_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Compare improvement
print("\nImprovement:")
for metric in baseline_metrics:
    improvement = ellipsoidal_metrics[metric] - baseline_metrics[metric]
    print(f"  {metric}: {improvement:.4f} ({improvement/baseline_metrics[metric]*100:.1f}%)")
