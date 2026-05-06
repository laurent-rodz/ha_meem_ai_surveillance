# Task: Implement Head Pose Estimation (Yaw, Pitch, Roll) via 2D Landmarks

**Context:**
We are working on the "Ha-Meem AI Surveillance" pipeline. Currently, our `SCRFDDetector` extracts 5-point 2D facial landmarks (`kps`), but we do not calculate head pose. This document outlines the implementation of a lightweight head pose estimator using OpenCV's `cv2.solvePnP` to reject faces with extreme angles.

---

## Step 1: Create the Pose Estimator Utility
Create a new file `core/utils/pose_estimator.py`. Use the following standard 3D generic face model for 5 points and calculate the Euler angles.

```python
import cv2
import numpy as np

class PoseEstimator:
    def __init__(self):
        # Generic 3D face model for 5 keypoints (Left Eye, Right Eye, Nose, Left Mouth, Right Mouth)
        self.model_points = np.array([
            [-39.756, 38.125, 0.0],    # Left eye
            [39.756, 38.125, 0.0],     # Right eye
            [0.0, -4.0, 35.0],         # Nose tip
            [-27.186, -42.87, 0.0],    # Left mouth corner
            [27.186, -42.87, 0.0]      # Right mouth corner
        ], dtype=np.float64)

    def estimate_pose(self, kps: np.ndarray, frame_shape: tuple) -> tuple:
        """
        Estimates yaw, pitch, and roll from 5-point landmarks.
        kps: (5, 2) numpy array from SCRFD
        frame_shape: (height, width, channels)
        Returns: (pitch, yaw, roll) in degrees
        """
        if kps is None or len(kps) != 5:
            return 0.0, 0.0, 0.0

        h, w = frame_shape[:2]
        
        # Approximate camera intrinsics
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self.model_points, 
            kps.astype(np.float64), 
            camera_matrix, 
            dist_coeffs, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to rotation matrix
        rmat, _ = cv2.Rodrigues(rvec)
        
        # Decompose matrix to Euler angles
        pose_mat = cv2.hconcat((rmat, tvec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch, yaw, roll = euler_angles.flatten()
        return float(pitch), float(yaw), float(roll)
```

---

## Step 2: Update the Face Dataclass
Locate the `Face` dataclass (likely in `core/detection/face.py` or similar). Add fields for the pose angles so they can be carried through the pipeline.

```python
# Add these attributes to the Face dataclass
pitch: float = 0.0
yaw: float = 0.0
roll: float = 0.0
```

---

## Step 3: Integrate into the Pipeline
In `apps/entry_pipeline/main.py` (or the relevant camera worker), instantiate the `PoseEstimator`. Immediately after the detector returns the faces, calculate the pose.

```python
# 1. Initialization (outside the loop)
from core.utils.pose_estimator import PoseEstimator
pose_estimator = PoseEstimator()

# 2. Integration (inside processing loop, after detection)
for face in detected_faces:
    pitch, yaw, roll = pose_estimator.estimate_pose(face.kps, frame.shape)
    face.pitch = pitch
    face.yaw = yaw
    face.roll = roll
```

---

## Step 4: Update the QualityGate
In `core/filtering/quality_gate.py`, add logic to reject faces that exceed a configurable angle threshold.

1.  Add thresholds to your config YAML (e.g., `max_yaw: 30.0`, `max_pitch: 30.0`).
2.  Update the `QualityGate` to check these values:

```python
# In QualityGate class:
def is_qualified(self, face: Face) -> bool:
    # Existing checks (blur, width, etc.)...
    
    # New pose checks
    if abs(face.yaw) > self.config['max_yaw'] or abs(face.pitch) > self.config['max_pitch']:
        return False
        
    return True
```

---

**Instructions to Agent:** Please write the necessary code modifications for the files mentioned above. Ensure the imports are correct and handle any potential `None` values gracefully.
