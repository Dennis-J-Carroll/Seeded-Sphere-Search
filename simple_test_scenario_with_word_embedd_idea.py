"""simple test scenario with word embeddings"""
    # Create initial random embeddings for 5 words
    words = ["cat", "dog", "tiger", "house", "building"]
    
    # Initial random embeddings (normalized)
    dim = 64  # Embedding dimension
    embeddings = F.normalize(torch.randn(1, len(words), dim), p=2, dim=2)
    
    # Define relationship matrix (higher value = stronger relationship)
    relationships = torch.tensor([
        [  # Relationships for batch item 0
            [0.0, 0.8, 0.7, 0.2, 0.1],  # cat's relationships
            [0.8, 0.0, 0.6, 0.3, 0.2],  # dog's relationships
            [0.7, 0.6, 0.0, 0.1, 0.1],  # tiger's relationships
            [0.2, 0.3, 0.1, 0.0, 0.9],  # house's relationships
            [0.1, 0.2, 0.1, 0.9, 0.0],  # building's relationships
        ]
    ]).float()
    
    # Define token frequencies (normalized to [0,1])
    frequencies = torch.tensor([[0.8, 0.9, 0.4, 0.7, 0.5]]).float()
    
    return words, embeddings, relationships, frequencies

def run_echo_experiment():
    """Run experiment with echo refinement"""
    words, embeddings, relationships, frequencies = create_test_scenario()
    
    # Create echo layer
    echo = EchoLayer(delta=0.1625, freq_scale=0.325, min_layers=1, max_layers=5)
    
    # Apply echo refinement
    refined_embeddings, evolution = echo(embeddings, relationships, frequencies)
    
    # Calculate similarity matrices before and after
    initial_sim = cosine_similarity(
        embeddings[0].detach().numpy()
    )
    
    refined_sim = cosine_similarity(
        refined_embeddings[0].detach().numpy()
    )
    
    return {
        'words': words,
        'initial_embeddings': embeddings,
        'refined_embeddings': refined_embeddings,
        'initial_similarity': initial_sim,
        'refined_similarity': refined_sim,
        'evolution': evolution
    })