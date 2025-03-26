"""
Tools for generating synthetic test data for search engine evaluation.
"""

import random

def generate_test_data(search_engine, num_queries=10, num_relevant=5, query_prefix="test_query_"):
    """
    Generate synthetic test data for evaluation
    
    Args:
        search_engine: Search engine instance
        num_queries: Number of test queries to generate
        num_relevant: Number of relevant documents per query
        query_prefix: Prefix for query IDs
        
    Returns:
        dict: Dictionary with 'queries' and 'relevance' keys
    """
    test_data = {
        "queries": [],
        "relevance": {}
    }
    
    # Get random documents
    doc_ids = list(search_engine.vocabulary.keys())
    random.shuffle(doc_ids)
    
    for i in range(min(num_queries, len(doc_ids))):
        doc_id = doc_ids[i]
        
        # Generate query from document title and content
        title = search_engine.vocabulary[doc_id]["title"]
        content = search_engine.vocabulary[doc_id]["content"]
        
        # Extract some phrases from the document
        sentences = content.split('.')[:5]  # Take first 5 sentences
        random.shuffle(sentences)
        
        query = title if random.random() < 0.5 else sentences[0]
        # Truncate to reasonable length
        query = query[:100].strip()
        
        test_data["queries"].append(query)
        
        # Find relevant documents (the source document and some similar ones)
        relevant_docs = [doc_id]
        
        # Find similar documents based on relationship matrix
        neighbors = [(neighbor, score) for neighbor, score in search_engine.relationships[doc_id].items()]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # Add top similar documents
        for neighbor, _ in neighbors[:num_relevant - 1]:
            if neighbor not in relevant_docs:
                relevant_docs.append(neighbor)
        
        test_data["relevance"][query] = relevant_docs
    
    return test_data
