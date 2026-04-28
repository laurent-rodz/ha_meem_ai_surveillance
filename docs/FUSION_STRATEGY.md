# Face Recognition Fusion Strategy

This document outlines the current state and planned upgrades for the "Fusion Strategy" within the Ha-Meem AI Surveillance pipeline.

## 1. What is a "Fusion Strategy"?
In facial recognition, when a person walks in front of the camera, the system tracks them over multiple frames (e.g., 6 to 10 frames). For each of these frames, the AdaFace recognition model extracts a 512-dimensional vector (an "embedding") representing their face. 

Instead of picking just one frame and matching it to the database (which might be an unfavorable angle or blurred image), the system **fuses** all vectors from the track buffer into a single, highly robust "consensus vector".

## 2. Current Setup: "Equal Mean Pool"
Currently, the pipeline employs a simple mathematical average to fuse embeddings. 

As implemented in `core/fusion/aggregator.py`:
```python
mean_embedding = np.mean(embeddings, axis=0)
```
This is an **Equal Mean Pool**. It treats every extracted embedding identically. If 8 frames are perfectly clear and 2 frames suffer from heavy motion blur, those 2 blurry frames will equally pull down the average quality of the final consensus vector. This can occasionally lead to failed recognition (false negatives) for fast-moving targets.

## 3. Recommended Upgrade: "Quality-Weighted Mean"
The recommended upgrade (listed as a "1h Effort | Pending" task in `PROJECT_ANALYSIS.md`) suggests giving proportional weight to high-quality frames.

Because the system already calculates a `blur_score` (Laplacian variance) and possesses a detection `confidence` score for every face, these metrics can be used as mathematical weights:
*   **High Weight**: A perfectly clear, front-facing frame (high blur score/high confidence) exerts a strong influence on the final average.
*   **Low Weight**: A slightly blurred or side-profile frame exerts very little influence on the final average.

### Mathematical Implementation Concept
Instead of `np.mean(embeddings)`, the new approach would resemble:
```python
# 'weights' is an array of quality scores for each embedding in the buffer
weighted_embeddings = np.average(embeddings, axis=0, weights=weights)
```

## Why Upgrade?
Implementing a **Quality-Weighted Mean** naturally filters out noise. It prevents sub-optimal frames from polluting the high-quality frames within the same track. This significantly increases overall facial recognition accuracy, especially in challenging lighting conditions or with uncooperative subjects moving quickly past the camera.
