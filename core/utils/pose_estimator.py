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
