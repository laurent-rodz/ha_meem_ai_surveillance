import numpy as np


class PoseEstimator:
    """Estimates head pose from 5 SCRFD keypoints using landmark geometry.

    Avoids solvePnP which requires accurate camera intrinsics (unavailable for
    arbitrary CCTV streams). Instead, uses direct geometric ratios that are
    camera-agnostic and reliable for filtering obviously non-frontal faces.

    Keypoint order (SCRFD): left_eye, right_eye, nose, left_mouth, right_mouth
    """

    def estimate_pose(self, kps: np.ndarray, frame_shape: tuple) -> tuple:
        """
        Estimates yaw, pitch, and roll from 5-point landmarks.

        Args:
            kps: (5, 2) numpy array from SCRFD
                 [left_eye, right_eye, nose, left_mouth, right_mouth]
            frame_shape: (height, width, channels) — used for normalisation

        Returns:
            (pitch, yaw, roll) in degrees (approximate)
        """
        if kps is None or len(kps) != 5:
            return 0.0, 0.0, 0.0

        left_eye, right_eye, nose, left_mouth, right_mouth = kps

        # --- Yaw: horizontal asymmetry of nose between the eyes ---
        # When facing forward, nose_x ≈ midpoint of the two eyes.
        eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
        eye_span = right_eye[0] - left_eye[0]  # positive when L eye is left of R eye

        if abs(eye_span) < 1e-6:
            return 0.0, 0.0, 0.0

        # Normalised offset: 0=frontal, ±0.5=profile
        nose_offset = (nose[0] - eye_mid_x) / abs(eye_span)
        # Scale to degrees: ±0.5 offset ≈ ±45° (beyond that is profile)
        yaw = float(np.clip(nose_offset * 90.0, -90.0, 90.0))

        # --- Pitch: vertical position of nose relative to eye-mouth band ---
        eye_mid_y = (left_eye[1] + right_eye[1]) / 2.0
        mouth_mid_y = (left_mouth[1] + right_mouth[1]) / 2.0
        face_height = mouth_mid_y - eye_mid_y

        if abs(face_height) < 1e-6:
            return 0.0, float(yaw), 0.0

        # Nose should sit ~40% down the eye-to-mouth band when frontal
        FRONTAL_RATIO = 0.40
        nose_ratio = (nose[1] - eye_mid_y) / face_height
        pitch_raw = (nose_ratio - FRONTAL_RATIO) / FRONTAL_RATIO
        # Scale to degrees: ±1.0 raw ≈ ±90° (full range)
        pitch = float(np.clip(pitch_raw * 90.0, -90.0, 90.0))

        # --- Roll: tilt of the inter-eye line ---
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        roll = float(np.degrees(np.arctan2(dy, dx)))

        return pitch, yaw, roll
