# Face Recognition Fusion Strategy

This document outlines the current state and planned upgrades for the "Fusion Strategy" within the Ha-Meem AI Surveillance pipeline.

## 1. What is a "Fusion Strategy"?
In facial recognition, when a person walks in front of the camera, the system tracks them over multiple frames (e.g., 6 to 10 frames). For each of these frames, the AdaFace recognition model extracts a 512-dimensional vector (an "embedding") representing their face. 

Instead of picking just one frame and matching it to the database (which might be an unfavorable angle or blurred image), the system **fuses** all vectors from the track buffer into a single, highly robust "consensus vector".

## 2. Implementation: "AdaFace Feature Norm Weighting"
The pipeline now employs **AdaFace Feature Norm Weighting** to fuse embeddings. This is a quality-aware fusion strategy where the weight of each frame is determined by the L2-norm of its unnormalized feature vector.

In AdaFace, the model is trained such that the feature norm correlates with image quality. Clear, front-facing images produce high norms, while blurry or occluded images produce low norms.

### How it works:
1.  **Extraction**: The `AdaFaceRecognizer` extracts the 512-d feature and its raw L2-norm.
2.  **Buffering**: Both the normalized embedding and the raw norm are stored in the `EmbeddingAggregator` buffer.
3.  **Weighted Average**: When a decision is needed, the system calculates a weighted average using the norms:
    ```python
    consensus_embedding = np.average(embeddings, axis=0, weights=norms)
    ```
4.  **Final Normalization**: The resulting consensus vector is re-normalized to a unit vector for cosine similarity matching.

### Why this is better:
*   **Automatic Quality Filtering**: Blurry or low-quality frames naturally have less influence on the final identity decision.
*   **Robustness**: It prevents "pollution" of the track buffer by transient occlusions or fast motion blur.
*   **Accuracy**: This is the state-of-the-art approach for temporal aggregation with AdaFace models.

## 3. Alternative Metrics (Optional)
While feature norms are the most direct proxy for quality in AdaFace, the system is also capable of using:
*   **Blur Score**: Laplacian variance-based sharpenss detection.
*   **Detection Confidence**: The confidence score from the SCRFD detector.

Currently, these are used as **hard gates** (dropping frames below a threshold) rather than soft weights, which simplifies the pipeline while maintaining high precision.

## 4. Track Purity & Outlier Rejection
Even with robust tracking, ID switches or severe occlusions can occur, introducing foreign embeddings into the buffer. To protect the consensus vector, we implement an **Outlier Rejection** step before fusion.

### Implementation:
Before a new frame's embedding is added to the `EmbeddingAggregator`, its cosine similarity is measured against the current running consensus (or the first highly confident frame). 
*   If the similarity falls below a strict "purity threshold" (e.g., `< 0.4`), the frame is flagged as an anomaly and rejected.
*   This prevents a single mis-tracked face from violently skewing the weighted average.

## 5. Low-Resolution and Pose Adaptations
Due to physical deployment constraints, such as steep camera angles and varying distances, bounding boxes often fall below ideal pixel resolutions. The fusion strategy incorporates safeguards for these sub-optimal captures:

*   **Resolution-Aware Hard Gating:** While we accept smaller bounding boxes to maximize capture rates, extreme cases trigger a hard gate. If the bounding box falls below the minimum operable threshold, the frame is discarded before feature extraction to save compute cycles.
*   **Pose Estimation Penalties (Planned):** In the future, we can introduce a soft penalty based on pitch and yaw. While AdaFace norms handle general quality well, explicitly penalizing extreme overhead angles ensures that frontal captures dominate the consensus weighting.

## 6. Buffer Management & Lifecycle
To maintain real-time performance and prevent memory bloat, the track buffer operates with strict lifecycle rules:

*   **Maximum Capacity (Rolling Window):** The buffer holds a maximum of `N` frames (e.g., 15). Once full, it operates as a First-In-First-Out (FIFO) queue, dropping the oldest embeddings. This ensures the consensus represents the most recent visual evidence.
*   **Triggering Recognition:** The fusion and subsequent database matching are triggered either when the buffer reaches a "minimum confidence mass" (e.g., 5 frames) or when the tracker signals that the target has exited the frame.
*   **Buffer Flush:** Once a definitive "Authorized / Unknown" classification is logged for a specific track ID, the buffer is locked to prevent redundant database queries, and eventually flushed when the track terminates.
