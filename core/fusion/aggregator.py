import numpy as np
from typing import Dict, Optional, Deque
from collections import deque
from ..detection.face import Face

class EmbeddingAggregator:
    """Aggregates multiple face embeddings from a single track into a robust consensus embedding."""
    
    def __init__(self, buffer_size: int = 10, min_frames: int = 8):
        self.buffer_size = buffer_size
        self.min_frames = min_frames
        # Each track buffer now stores a deque of (embedding, norm) with a fixed maxlen
        self.track_buffers: Dict[int, Deque[tuple[np.ndarray, float]]] = {}

    def add_face(self, face: Face):
        """Adds a face to the track buffer for aggregation."""
        if face.track_id is None or face.embedding is None:
            return

        if face.track_id not in self.track_buffers:
            self.track_buffers[face.track_id] = deque(maxlen=self.buffer_size)
        
        # Store both normalized embedding and its raw feature norm
        # deque automatically handles the maxlen by discarding the oldest item
        self.track_buffers[face.track_id].append((face.embedding, face.feature_norm))

    def get_aggregated_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Returns the quality-weighted consensus embedding for the track."""
        if track_id not in self.track_buffers or len(self.track_buffers[track_id]) < self.min_frames:
            return None
        
        # Unpack embeddings and norms (weights)
        buffer = self.track_buffers[track_id]
        embeddings = np.stack([item[0] for item in buffer])
        norms = np.array([item[1] for item in buffer], dtype=np.float32)
        
        # Zero-Weight Edge Case: If all frames are so degraded that norms are 0,
        # fallback to equal weighting (simple mean) to avoid ZeroDivisionError.
        total_norm = np.sum(norms)
        if total_norm <= 0:
            mean_embedding = np.mean(embeddings, axis=0).astype(np.float32)
        else:
            # AdaFace Feature Norm Weighting: np.average with feature norms as weights.
            # We explicitly use float32 to maintain consistency with the ONNX model output.
            mean_embedding = np.average(embeddings, axis=0, weights=norms).astype(np.float32)
        
        # Final normalization to ensure a unit vector
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            return mean_embedding / norm
        return mean_embedding

    def clear_track(self, track_id: int):
        """Removes track data from memory."""
        if track_id in self.track_buffers:
            del self.track_buffers[track_id]
