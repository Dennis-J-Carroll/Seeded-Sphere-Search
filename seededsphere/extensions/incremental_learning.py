"""
Incremental Learning Extension for SeededSphere
Implements efficient document updates without full retraining.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
from cachetools import TTLCache
from tqdm import tqdm

class IncrementalLearning:
    def __init__(self, sss_instance, config=None):
        """
        Initialize incremental learning extension.
        
        Args:
            sss_instance: SeededSphereSearch instance
            config: Configuration dictionary with optional parameters
        """
        self.sss = sss_instance
        self.config = {
            "cache_ttl": 3600,  # Cache lifetime in seconds
            "cache_maxsize": 10000,  # Maximum number of cached items
            "similarity_threshold": 0.3,  # Threshold for considering documents as related
            "max_affected_neighbors": 50,  # Maximum number of neighbors to update
            "checkpoint_frequency": 100,  # Save checkpoint every N updates
        }
        if config:
            self.config.update(config)
            
        # Initialize caches
        self.relationship_cache = TTLCache(
            maxsize=self.config["cache_maxsize"],
            ttl=self.config["cache_ttl"]
        )
        
        # Update counter for checkpointing
        self.update_counter = 0
    
    def detect_hotspots(self, new_documents: List[dict]) -> Set[str]:
        """
        Identify existing documents most affected by new additions.
        
        Args:
            new_documents: List of new document dictionaries
            
        Returns:
            Set of document IDs that need updating
        """
        affected_ids = set()
        
        # Generate embeddings for new documents
        new_embeddings = self.sss._generate_initial_embeddings(new_documents)
        
        # Find similar existing documents
        for doc in new_documents:
            doc_id = doc["id"]
            if doc_id not in new_embeddings:
                continue
                
            # Get embedding for new document
            new_embedding = new_embeddings[doc_id]
            
            # Calculate similarity with existing documents
            similarities = []
            for existing_id, existing_embedding in self.sss.refined_embeddings.items():
                # Check cache first
                cache_key = f"{doc_id}_{existing_id}"
                if cache_key in self.relationship_cache:
                    similarity = self.relationship_cache[cache_key]
                else:
                    # Calculate cosine similarity
                    similarity = np.dot(new_embedding, existing_embedding)
                    self.relationship_cache[cache_key] = similarity
                
                if similarity > self.config["similarity_threshold"]:
                    similarities.append((existing_id, similarity))
            
            # Sort by similarity and take top K neighbors
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = similarities[:self.config["max_affected_neighbors"]]
            
            # Add to affected IDs
            affected_ids.update(doc_id for doc_id, _ in top_neighbors)
        
        return affected_ids
    
    def incremental_update(self, new_documents: List[dict]) -> Tuple[int, int]:
        """
        Update embeddings for new documents and their affected neighbors.
        
        Args:
            new_documents: List of new document dictionaries
            
        Returns:
            Tuple of (number of new documents, number of affected documents)
        """
        # Find affected documents
        affected_ids = self.detect_hotspots(new_documents)
        
        # Generate embeddings for new documents
        new_embeddings = self.sss._generate_initial_embeddings(new_documents)
        
        # Update relationship matrix
        self._update_relationships(new_documents, list(affected_ids))
        
        # Perform partial echo refinement
        self._partial_echo_refinement(new_documents, list(affected_ids))
        
        # Update counters and save checkpoint if needed
        self.update_counter += len(new_documents)
        if self.update_counter >= self.config["checkpoint_frequency"]:
            self.save_checkpoint()
            self.update_counter = 0
        
        return len(new_documents), len(affected_ids)
    
    def _update_relationships(self, new_documents: List[dict], affected_ids: List[str]):
        """Update relationship matrix for new and affected documents"""
        # Get all document IDs to update
        doc_ids = [doc["id"] for doc in new_documents] + affected_ids
        
        # Update relationship matrix entries
        for i, doc_id1 in enumerate(doc_ids):
            for j, doc_id2 in enumerate(doc_ids[i+1:], i+1):
                # Skip if relationship is cached and still valid
                cache_key = f"{doc_id1}_{doc_id2}"
                if cache_key in self.relationship_cache:
                    relationship = self.relationship_cache[cache_key]
                else:
                    # Calculate new relationship strength
                    emb1 = self.sss.refined_embeddings.get(doc_id1)
                    emb2 = self.sss.refined_embeddings.get(doc_id2)
                    if emb1 is not None and emb2 is not None:
                        relationship = np.dot(emb1, emb2)
                        self.relationship_cache[cache_key] = relationship
                    else:
                        continue
                
                # Update relationship matrix
                self.sss.relationships[doc_id1][doc_id2] = relationship
                self.sss.relationships[doc_id2][doc_id1] = relationship
    
    def _partial_echo_refinement(self, new_documents: List[dict], affected_ids: List[str]):
        """Perform echo refinement only on new and affected documents"""
        doc_ids = [doc["id"] for doc in new_documents] + affected_ids
        
        # Get current embeddings
        current_embeddings = {
            doc_id: self.sss.refined_embeddings.get(doc_id, self.sss.embeddings.get(doc_id))
            for doc_id in doc_ids
        }
        
        # Perform echo refinement iterations
        for doc_id in tqdm(doc_ids, desc="Echo refinement"):
            if doc_id not in current_embeddings:
                continue
                
            embedding = current_embeddings[doc_id]
            layers = self.sss._calculate_echo_layers(doc_id)
            
            for _ in range(layers):
                # Calculate shift from neighbors
                shift = np.zeros_like(embedding)
                neighbor_count = 0
                
                # Consider only relationships with other documents in the update set
                for neighbor_id in doc_ids:
                    if neighbor_id == doc_id:
                        continue
                        
                    relationship = self.sss.relationships[doc_id].get(neighbor_id, 0)
                    if relationship > 0 and neighbor_id in current_embeddings:
                        neighbor_embedding = current_embeddings[neighbor_id]
                        diff = neighbor_embedding - embedding
                        shift += relationship * diff * self.sss.config["delta"]
                        neighbor_count += 1
                
                if neighbor_count > 0:
                    # Apply shift and normalize
                    embedding = embedding + shift
                    embedding = self.sss._normalize_vector(embedding)
                    current_embeddings[doc_id] = embedding
            
            # Update refined embeddings
            self.sss.refined_embeddings[doc_id] = embedding
    
    def save_checkpoint(self):
        """Save current state as a checkpoint"""
        import pickle
        import os
        from datetime import datetime
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)
        
        # Save checkpoint with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"checkpoints/sss_checkpoint_{timestamp}.pkl"
        
        with open(checkpoint_path, "wb") as f:
            pickle.dump({
                "embeddings": self.sss.embeddings,
                "refined_embeddings": self.sss.refined_embeddings,
                "relationships": self.sss.relationships,
                "vocabulary": self.sss.vocabulary,
                "config": self.sss.config
            }, f)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load state from a checkpoint"""
        import pickle
        
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
            
        self.sss.embeddings = checkpoint["embeddings"]
        self.sss.refined_embeddings = checkpoint["refined_embeddings"]
        self.sss.relationships = checkpoint["relationships"]
        self.sss.vocabulary = checkpoint["vocabulary"]
        self.sss.config.update(checkpoint["config"])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
