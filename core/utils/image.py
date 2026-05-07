import numpy as np


def pose_weight(kps: np.ndarray) -> float:
    """Weight in [0.1, 1.0] reflecting how frontal the face is.

    Uses the horizontal offset of the nose tip from the eye midpoint,
    normalised by the inter-eye distance.  A perfectly frontal face scores
    1.0; a ~60° profile scores ~0.1.  The face is never discarded — the
    weight just reduces the profile frame's contribution to the temporal
    consensus so sharper, frontal frames dominate.
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = abs(float(right_eye[0]) - float(left_eye[0]))
    if eye_dist < 1.0:
        return 0.5
    eye_center_x = (float(left_eye[0]) + float(right_eye[0])) / 2.0
    nose_offset = abs(float(nose[0]) - eye_center_x) / eye_dist
    # 0.0 offset (frontal) → 1.0 weight; 0.5+ offset (profile) → 0.1 weight
    return float(max(0.1, 1.0 - nose_offset * 1.8))
