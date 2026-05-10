import numpy as np
from typing import List
import supervision as sv
from ..detection.face import Face

class ByteTracker:
    """ByteTrack implementation using supervision for robust face tracking."""
    
    def __init__(self, track_activation_threshold: float = 0.25, lost_track_buffer: int = 30, minimum_matching_threshold: float = 0.8, frame_rate: int = 30):
        # You can tune these or set them via config.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )

    def _calculate_iou(self, bbox1, bbox2):
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        iou = inter_area / float(bbox1_area + bbox2_area - inter_area + 1e-6)
        return iou

    def update(self, detected_faces: List[Face]) -> List[Face]:
        """Updates tracks with new detections using ByteTrack."""
        if not detected_faces:
            # Empty update progresses the track age without new detections.
            self.tracker.update_with_detections(sv.Detections.empty())
            return []
            
        # Convert Face objects to sv.Detections
        # Note: some bounding boxes might be out of range, supervision handles generic formatting
        xyxy = np.array([f.bbox[:4] for f in detected_faces])
        confidence = np.array([f.confidence for f in detected_faces])
        
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=np.zeros(len(detected_faces), dtype=int)
        )
        
        tracked_detections = self.tracker.update_with_detections(detections)
        
        updated_faces = []
        
        # We need to map the output tracks back to our original Face objects
        # because tracked_detections might be reordered or filtered.
        for i in range(len(tracked_detections)):
            tx1, ty1, tx2, ty2 = tracked_detections.xyxy[i]
            tracker_id = tracked_detections.tracker_id[i]
            
            best_iou = -1
            best_face = None
            
            for face in detected_faces:
                # ByteTrack returns smoothed bounding boxes. The IoU should be high
                # compared to the original detection. Use >= so that ties go to the
                # last-seen face (deterministic ordering), matching SORTTracker stability.
                iou = self._calculate_iou((tx1, ty1, tx2, ty2), face.bbox[:4])
                if iou >= best_iou:
                    best_iou = iou
                    best_face = face
                    
            # If we've found a solid match, attach the track ID.
            # 0.3 threshold provides leeway as kalman filter can predict smoothed bboxes
            # slightly off the raw detection.
            if best_face is not None and best_iou > 0.3:
                best_face.track_id = tracker_id
                # Keep original raw coordinates to maintain accurate landmarks (kps)
                updated_faces.append(best_face)
                
        return updated_faces
