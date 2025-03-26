import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel
import random 

class SeededSphereSearch:
    def __init__(self, config=None):
        """
        Initialize the Seeded Sphere Search system.
        
        Args:
            config: Configuration dictionary with parameters
        """
        # Default configuration
        self.config = {
            "dimensions": 384,  # Embedding dimensions
            "min_echo_layers": 1,
            "max_echo_layers": 5,
            "frequency_scale": 0.325,
            "delta": 0.1625,  # Step size for echo refinement
            "alpha": 0.5,      # Temperature for scoring
            "base_anchor_weight": 0.2,
            "use_pretrained": True,  # Whether to use pretrained embeddings or random
            "pretrained_model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # Internal state
        self.vocabulary = {}  # Maps document IDs to their metadata
        self.word_frequencies = {}  # Maps words to their frequencies
        self.embeddings = {}  # Initial document embeddings
        self.refined_embeddings = {}  # Refined embeddings after echo
        self.relationships = defaultdict(dict)  # Document relationship matrix
        
        # Initialize transformer model if using pretrained embeddings
        if self.config["use_pretrained"]:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config["pretrained_model"])
            self.model = AutoModel.from_pretrained(self.config["pretrained_model"])
            
        self.initialized = False
        
    def initialize(self, corpus):
        """
        Initialize the system with the document corpus
        
        Args:
            corpus: List of document dictionaries with id, title, and content
        """
        print("Initializing Seeded Sphere Search...")
        
        # Store document vocabulary
        for doc in corpus:
            self.vocabulary[doc["id"]] = {
                "title": doc["title"],
                "content": doc["content"],
                "metadata": doc.get("metadata", {})
            }
        
        # Calculate document frequencies using TF-IDF
        print("Calculating document frequencies...")
        self._calculate_document_frequencies(corpus)
        
        # Calculate relationship matrix
        print("Building relationship matrix...")
        self._build_relationship_matrix(corpus)
        
        # Generate initial embeddings
        print("Generating initial embeddings...")
        self._generate_initial_embeddings(corpus)
        
        # Perform echo refinement
        print("Performing echo refinement...")
        self._perform_echo_refinement()
        
        self.initialized = True
        print("Initialization complete.")
        return self
        
    def _calculate_document_frequencies(self, corpus):
        """Calculate document frequencies and importance scores"""
        # Extract document content
        documents = [doc["content"] for doc in corpus]
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=50000)
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Store word frequencies as sum of TF-IDF weights
        feature_names = vectorizer.get_feature_names_out()
        for i, doc in enumerate(corpus):
            doc_id = doc["id"]
            self.word_frequencies[doc_id] = np.mean(tfidf_matrix[i].toarray())
    
    def _build_relationship_matrix(self, corpus):
        """Build document relationship matrix based on content similarity"""
        import scipy.sparse as sp
        from sklearn.metrics.pairwise import cosine_similarity
        
        documents = [doc["content"] for doc in corpus]
        doc_ids = [doc["id"] for doc in corpus]
        
        print(f"Building relationship matrix for {len(documents)} documents...")
        
        # Create TF-IDF matrix with sparse representation
        vectorizer = TfidfVectorizer(max_features=10000)
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Process in batches to avoid memory issues
        batch_size = 1000  # Adjust based on available memory
        for i in range(0, len(doc_ids), batch_size):
            batch_end = min(i + batch_size, len(doc_ids))
            batch_ids = doc_ids[i:batch_end]
            batch_matrix = tfidf_matrix[i:batch_end]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(doc_ids)-1)//batch_size + 1}...")
            
            # Calculate similarity with all documents
            batch_similarity = cosine_similarity(batch_matrix, tfidf_matrix)
            
            # Store significant relationships (above threshold)
            threshold = 0.05  # Ignore very weak relationships
            for batch_idx, doc_id1 in enumerate(batch_ids):
                doc_idx = i + batch_idx
                # Get indices of top relationships (excluding self)
                row = batch_similarity[batch_idx]
                # Set self-similarity to 0
                if doc_idx < len(row):
                    row[doc_idx] = 0
                
                # Get top relationships
                top_indices = np.argsort(row)[-50:]  # Top 50 relationships
                
                for idx in top_indices:
                    if row[idx] > threshold and idx < len(doc_ids):
                        doc_id2 = doc_ids[idx]
                        if doc_id1 != doc_id2:  # Redundant check
                            self.relationships[doc_id1][doc_id2] = row[idx]
        
        print(f"Relationship matrix built successfully.")
    
    def _generate_initial_embeddings(self, corpus):
        """Generate initial embeddings for documents"""
        if self.config["use_pretrained"]:
            self._generate_transformer_embeddings(corpus)
        else:
            self._generate_random_embeddings(corpus)
    
    def _generate_transformer_embeddings(self, corpus):
        """Generate embeddings using transformer models"""
        print("Generating transformer embeddings...")
        
        # Process in batches for efficiency
        batch_size = 16  # Adjust based on GPU memory
        total_batches = (len(corpus) + batch_size - 1) // batch_size
        
        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        print(f"Using device: {device}")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(corpus))
            batch = corpus[start_idx:end_idx]
            
            if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
                print(f"Processing batch {batch_idx+1}/{total_batches}...")
            
            # Process each document in the batch
            batch_texts = [doc["content"] for doc in batch]
            batch_ids = [doc["id"] for doc in batch]
            
            # Encode all texts in the batch
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                return_tensors="pt", 
                max_length=512
            ).to(device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use mean pooling instead of CLS token for better representation
            attention_mask = inputs["attention_mask"]
            embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            
            # Process each embedding in the batch
            for i, doc_id in enumerate(batch_ids):
                # Move to CPU and convert to numpy
                embedding = embeddings[i].cpu().numpy()
                # Normalize to unit sphere
                normalized_embedding = self._normalize_vector(embedding)
                # Store the embedding
                self.embeddings[doc_id] = normalized_embedding
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling operation to get sentence embeddings"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _generate_random_embeddings(self, corpus):
        """Generate random embeddings for documents (for testing)"""
        print("Generating random embeddings...")
        
        for doc in corpus:
            doc_id = doc["id"]
            
            # Create random embedding
            random_embedding = np.random.randn(self.config["dimensions"])
            
            # Normalize to unit sphere
            normalized_embedding = self._normalize_vector(random_embedding)
            
            # Store the embedding
            self.embeddings[doc_id] = normalized_embedding
    
    def _normalize_vector(self, vector):
        """Normalize a vector to unit length (project to unit sphere)"""
        norm = np.sqrt(np.sum(vector ** 2))
        if norm < 1e-8:
            return np.ones(len(vector)) / np.sqrt(len(vector))
        return vector / norm
    
    def _calculate_echo_layers(self, doc_id):
        """Calculate the number of echo layers based on document frequency"""
        frequency = self.word_frequencies.get(doc_id, 0.5)
        return min(
            self.config["max_echo_layers"],
            self.config["min_echo_layers"] + 
            int(frequency * self.config["frequency_scale"] * self.config["max_echo_layers"])
        )
    
    def _perform_echo_refinement(self):
        """Perform echo refinement for all documents"""
        # Initialize with current embeddings
        for doc_id in self.vocabulary:
            self.refined_embeddings[doc_id] = np.copy(self.embeddings[doc_id])
        
        # Calculate total number of echo operations for progress reporting
        total_operations = sum(self._calculate_echo_layers(doc_id) for doc_id in self.vocabulary)
        completed_operations = 0
        
        # For each document, calculate echo refinement
        for doc_id in self.vocabulary:
            layers = self._calculate_echo_layers(doc_id)
            current_embedding = np.copy(self.embeddings[doc_id])
            
            # Perform echo refinement for each layer
            for k in range(1, layers + 1):
                # Calculate shift from neighbors
                shift = np.zeros(self.config["dimensions"])
                neighbor_count = 0
                
                # Sort neighbors by relationship strength and take top N
                sorted_neighbors = sorted(
                    self.relationships[doc_id].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:50]  # Limit to top 50 neighbors for efficiency
                
                for neighbor_id, relationship in sorted_neighbors:
                    if neighbor_id in self.embeddings:
                        neighbor_embedding = self.embeddings[neighbor_id]
                        
                        # Calculate difference vector
                        diff = neighbor_embedding - current_embedding
                        
                        # Add weighted difference to shift
                        shift += relationship * diff * self.config["delta"]
                        neighbor_count += 1
                
                # Apply shift (with early stopping if shift is negligible)
                if neighbor_count > 0 and np.linalg.norm(shift) > 1e-6:
                    updated_embedding = current_embedding + shift
                    # Normalize to unit sphere
                    current_embedding = self._normalize_vector(updated_embedding)
                else:
                    # Skip further iterations if no significant shift
                    break
                    
                completed_operations += 1
                if completed_operations % 1000 == 0:
                    print(f"Echo refinement: {completed_operations}/{total_operations} operations completed")
            
            # Store the final refined embedding
            self.refined_embeddings[doc_id] = current_embedding
    
    def search(self, query, top_k=10, use_ellipsoidal=False, ellipsoidal_transformer=None):
        """
        Search for relevant documents given a query
        
        Args:
            query: Text query
            top_k: Number of top results to return
            use_ellipsoidal: Whether to use ellipsoidal transformation for scoring
            ellipsoidal_transformer: EllipsoidalTransformation instance
            
        Returns:
            list: Top k relevant documents
        """
        if not self.initialized:
            raise ValueError("SeededSphereSearch not initialized. Call initialize() first.")
        
        # Encode the query
        query_embedding = self._encode_query(query)
        
        # Apply ellipsoidal transformation if requested
        if use_ellipsoidal and ellipsoidal_transformer is not None:
            query_embedding = ellipsoidal_transformer.transform_query(query_embedding)
        
        # Use approximate nearest neighbors for large corpora
        use_ann = len(self.refined_embeddings) > 10000
        
        # Ensure numpy is available for both branches
        import numpy as np
        
        if use_ann:
            # For large corpora, use approximate nearest neighbors
            import faiss
            
            # Build index if it doesn't exist
            if not hasattr(self, 'index'):
                print("Building FAISS index for fast search...")
                self.doc_ids = list(self.refined_embeddings.keys())
                embeddings_array = np.array([self.refined_embeddings[doc_id] for doc_id in self.doc_ids])
                
                # Use L2 distance (equivalent to angular distance for normalized vectors)
                self.index = faiss.IndexFlatL2(self.config["dimensions"])
                self.index.add(embeddings_array)
                print("Index built successfully.")
            
            # Search using the index
            distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
            
            scores = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.doc_ids):  # Ensure valid index
                    doc_id = self.doc_ids[idx]
                    # Convert L2 distance to angular distance and score
                    distance = np.sqrt(distances[0][i]) / 2  # Approximate conversion
                    score = np.exp(-self.config["alpha"] * distance)
                    
                    scores.append({
                        "id": doc_id,
                        "title": self.vocabulary[doc_id]["title"],
                        "score": score,
                        "distance": distance
                    })
        else:
            # For smaller corpora, compute scores directly
            scores = []
            for doc_id, embedding in self.refined_embeddings.items():
                # Apply ellipsoidal transformation if requested
                if use_ellipsoidal and ellipsoidal_transformer is not None:
                    embedding = ellipsoidal_transformer.transform_document(embedding)
                
                # Calculate angular distance
                dot_product = np.dot(query_embedding, embedding)
                # Clamp to avoid numerical issues
                dot_product = max(-1.0, min(1.0, dot_product))
                distance = np.arccos(dot_product)
                
                # Calculate score
                score = np.exp(-self.config["alpha"] * distance)
                
                scores.append({
                    "id": doc_id,
                    "title": self.vocabulary[doc_id]["title"],
                    "score": score,
                    "distance": distance
                })
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k results
        return scores[:top_k]
    
    def _encode_query(self, query):
        """Encode the query text into an embedding"""
        if self.config["use_pretrained"]:
            # Encode using transformer
            inputs = self.tokenizer(query, padding=True, truncation=True, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            embeddings = outputs.last_hidden_state[:, 0, :]
            embedding = embeddings[0].numpy()
        else:
            # Simple random embedding for testing
            embedding = np.random.randn(self.config["dimensions"])
        
        # Normalize to unit sphere
        return self._normalize_vector(embedding)
    
    def get_document_info(self, doc_id):
        """Get detailed information about a document"""
        if doc_id not in self.vocabulary:
            return None
        
        return {
            "id": doc_id,
            "title": self.vocabulary[doc_id]["title"],
            "content": self.vocabulary[doc_id]["content"],
            "metadata": self.vocabulary[doc_id]["metadata"],
            "layers": self._calculate_echo_layers(doc_id),
            "embedding": self.embeddings[doc_id],
            "refined_embedding": self.refined_embeddings[doc_id]
        }