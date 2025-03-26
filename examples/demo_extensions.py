"""
Demonstration of SeededSphere Extensions
Shows how to use incremental learning and domain-specific configurations together.
"""

import json
from seededsphere import SeededSphereSearch
from seededsphere.extensions import IncrementalLearning, DomainConfigurator

def load_initial_documents():
    """Load initial set of academic papers"""
    return [
        {
            "id": "paper1",
            "title": "Introduction to Machine Learning",
            "content": "This paper provides an overview of machine learning concepts..."
        },
        {
            "id": "paper2",
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning has revolutionized artificial intelligence..."
        }
    ]

def load_new_documents():
    """Load new documents to add incrementally"""
    return [
        {
            "id": "code1",
            "title": "Python Implementation",
            "content": "def train_model(data):\n    model = Sequential()\n    model.add(Dense(64))"
        },
        {
            "id": "paper3",
            "title": "Recent Advances in NLP",
            "content": "Natural language processing has seen significant progress..."
        }
    ]

def main():
    # Initialize domain configurator
    domain_config = DomainConfigurator()
    
    # Load initial academic papers and get appropriate configuration
    initial_docs = load_initial_documents()
    academic_config = domain_config.get_configuration("academic")
    
    # Initialize search system with academic configuration
    print("\nInitializing search system with academic configuration...")
    sss = SeededSphereSearch(academic_config)
    sss.initialize(initial_docs)
    
    # Perform initial search
    print("\nPerforming initial search...")
    results = sss.search("machine learning concepts", top_k=2)
    print("\nInitial search results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
    
    # Initialize incremental learning
    print("\nInitializing incremental learning...")
    incremental = IncrementalLearning(sss)
    
    # Load new documents (mixed domain)
    new_docs = load_new_documents()
    
    # Detect domains for new documents
    print("\nDetecting domains for new documents...")
    for doc in new_docs:
        domain = domain_config.detect_domain(doc["content"])
        print(f"Document '{doc['title']}' detected as domain: {domain}")
        
        # Get domain-specific configuration
        doc_config = domain_config.get_configuration(domain)
        print(f"Using configuration: {doc_config}")
    
    # Incrementally add new documents
    print("\nAdding new documents incrementally...")
    n_new, n_affected = incremental.incremental_update(new_docs)
    print(f"Updated {n_new} new documents and affected {n_affected} existing documents")
    
    # Save checkpoint
    print("\nSaving checkpoint...")
    incremental.save_checkpoint()
    
    # Perform search again
    print("\nPerforming search with updated index...")
    results = sss.search("machine learning implementation", top_k=3)
    print("\nUpdated search results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (Score: {result['score']:.4f})")
    
    # Show domain information
    print("\nAvailable domain configurations:")
    for domain in domain_config.list_domains():
        info = domain_config.get_domain_info(domain)
        print(f"\n{info['name']} ({info['source']}):")
        print(f"Description: {info['description']}")
        print("Key settings:")
        print(f"  - Echo layers: {info['config']['min_echo_layers']}-{info['config']['max_echo_layers']}")
        print(f"  - Delta: {info['config']['delta']}")
        print(f"  - Alpha: {info['config']['alpha']}")

if __name__ == "__main__":
    main()
