import numpy as np
from typing import Dict, Optional, Tuple, List
import faiss

class FaceDatabase:
    """In-memory database for matching face embeddings using FAISS."""

    def __init__(self, embeddings: Dict[str, np.ndarray]):
        """
        Initializes the database and normalizes all stored embeddings.
        
        Args:
            embeddings: Dictionary mapping identity IDs to their 512-d embeddings.
        """
        self.ids = []
        self.index = faiss.IndexFlatIP(512)  # Inner product = cosine on normalized vecs
        
        if embeddings:
            all_embeddings = []
            all_ids = []

            for person_id, emb_list in embeddings.items():
                for emb in emb_list:
                    all_embeddings.append(emb)
                    all_ids.append(person_id)

            raw_embeddings = np.stack(all_embeddings)
            self.ids = all_ids

            norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
            self.stored_embeddings = raw_embeddings / (norms + 1e-6)
            
            self.index.add(self.stored_embeddings.astype(np.float32))

    def add_identity(self, person_id: str, embedding: np.ndarray):
        """Adds a new person without full rebuild."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        self.index.add(embedding.astype(np.float32)[np.newaxis, :])
        self.ids.append(person_id)

    def match(self, query_embedding: np.ndarray, threshold: float, match_margin: float = 0.05, match_top_k: int = 10) -> Tuple[Optional[str], float]:
        """
        Finds the best matching identity for a query embedding using FAISS.
        
        Args:
            query_embedding: The query 512-d embedding.
            threshold: Minimum cosine similarity score to qualify as a match.
            match_margin: Minimum score difference between top-1 and top-2 identities.
            match_top_k: Number of top candidates to retrieve for margin test.
            
        Returns:
            Tuple of (best_id, best_score). best_id is None if below threshold or margin test fails.
        """
        if self.index.ntotal == 0:
            return None, 0.0

        # Normalize query embedding
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        query = query_embedding.astype(np.float32)[np.newaxis, :]
        D, I = self.index.search(query, k=min(match_top_k, self.index.ntotal))
        
        scores = D[0]
        indices = I[0]
        
        # Group by identity and get best score per identity
        identity_best_scores = {}
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            identity = self.ids[idx]
            if identity not in identity_best_scores or score > identity_best_scores[identity]:
                identity_best_scores[identity] = float(score)
        
        if not identity_best_scores:
            return None, 0.0
        
        # Sort identities by best score (descending)
        sorted_identities = sorted(identity_best_scores.items(), key=lambda x: x[1], reverse=True)
        
        best_id, best_score = sorted_identities[0]
        
        if best_score < threshold:
            return None, best_score
        
        # Margin test: if there's a second identity, check the margin
        if len(sorted_identities) >= 2:
            second_best_score = sorted_identities[1][1]
            if best_score - second_best_score < match_margin:
                return None, best_score
        
        return best_id, best_score
