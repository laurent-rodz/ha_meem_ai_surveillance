import cv2
import time
import os
from datetime import datetime
import numpy as np

from core.events import EventEmitter, SnapshotWriter
from core.detection import SCRFDDetector
from core.tracking import ByteTracker
from core.recognition import AdaFaceRecognizer
from core.fusion import EmbeddingAggregator
from core.quality import calculate_blur_score
from core.database import FaceDatabase
from core.config import load_config, load_merged_configs

def run_pipeline():
    # Load and merge configs
    config = load_merged_configs([
        'configs/default.yaml',
        'configs/thresholds.yaml',
        'configs/tensorrt.yaml'
    ])
    
    camera_cfg = load_config('configs/cameras.yaml')
    
    # Initialize components
    # Note: Paths are placeholders as weights aren't committed
    detector = SCRFDDetector(config, config['models']['scrfd_onnx'])
    tracker = ByteTracker(track_activation_threshold=0.25, lost_track_buffer=30)
    recognizer = AdaFaceRecognizer(config, config['models']['adaface_onnx'])
    aggregator = EmbeddingAggregator(
        buffer_size=10, 
        min_frames=config['recognition']['min_frames_for_decision']
    )
    
    gallery_path = config['dataset']['gallery_embeddings']
    gallery_embeddings = np.load(gallery_path, allow_pickle=True).item()

    face_db = FaceDatabase(gallery_embeddings)
    
    # Open camera (using camera_01 from config)
    cap = cv2.VideoCapture(camera_cfg['cameras'][0]['url'])
    
    event_emitter = EventEmitter(
        camera_id=camera_cfg['cameras'][0]['id'],
        log_file=os.getenv("LOG_FILE", "logs/events.jsonl")
    )
    
    snapshot_writer = SnapshotWriter(
        base_dir="snapshots",
        camera_id=camera_cfg['cameras'][0]['id']
    )
    
    decided_tracks = set()
    
    # Cooldown mechanism for authorized identities
    identity_last_seen = {}
    identity_cooldown_seconds = 6
    
    print("Starting AI Surveillance Pipeline...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        start_time = time.time()
        
        # 1. Detection
        faces = detector.detect(frame)
        
        # 2. Tracking
        tracked_faces = tracker.update(faces)
        
        active_track_ids = set(face.track_id for face in tracked_faces)
        for track_id in list(aggregator.track_buffers.keys()):
            if track_id not in active_track_ids:
                aggregator.clear_track(track_id)
                decided_tracks.discard(track_id)
                
        valid_faces = []
        valid_face_imgs = []
        
        for face in tracked_faces:
            # Operational Constraints: Resolution Gate
            if face.width < config['recognition']['min_face_size']:
                continue
                
            # Blur Rejection
            x1, y1, x2, y2 = face.bbox[:4].astype(int)
            bbox_crop = frame[max(0, y1):y2, max(0, x1):x2]
            
            face.blur_score = calculate_blur_score(bbox_crop)
            if face.blur_score < config['recognition']['blur_threshold']:
                continue
                
            # Face Alignment
            if face.kps is not None:
                from insightface.utils import face_align
                face_img = face_align.norm_crop(frame, face.kps)
            else:
                face_img = bbox_crop
                
            valid_faces.append(face)
            valid_face_imgs.append(face_img)

        # 3. Recognition (Feature Extraction) - Batched
        if valid_face_imgs:
            embeddings = recognizer.extract_embeddings_batch(valid_face_imgs)
            
            for face, embedding in zip(valid_faces, embeddings):
                face.embedding = embedding
                
                # 4. Fusion (Aggregation)
                aggregator.add_face(face)
            
                # Get consensus
                consensus_emb = aggregator.get_aggregated_embedding(face.track_id)
                
                if consensus_emb is not None and face.track_id not in decided_tracks:
                    identity, score = face_db.match(
                        consensus_emb, 
                        config['recognition']['similarity_threshold']
                    )
                    
                    current_time = time.time()
                    
                    event_emitted = False
                    
                    # Check for cooldown if authorized
                    can_emit = True
                    if identity is not None:
                        last_seen = identity_last_seen.get(identity, 0)
                        if current_time - last_seen < identity_cooldown_seconds:
                            can_emit = False
                    
                    if can_emit:
                        # 1. Create a single source of truth for time
                        event_time = datetime.now()
                        
                        # 2. Create event object
                        event_data = {
                            "timestamp": event_time.isoformat(),
                            "camera_id": camera_cfg['cameras'][0]['id'],
                            "track_id": face.track_id,
                            "identity": identity,
                            "score": float(score),
                            "event": "AUTHORIZED" if identity else "UNKNOWN"
                        }
                        
                        # 3. Capture and save snapshot
                        x1, y1, x2, y2 = face.bbox[:4].astype(int)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, identity or "UNKNOWN", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        snapshot_path = snapshot_writer.save(frame, identity, timestamp=event_time)
                        event_data["snapshot"] = snapshot_path
                        
                        # 4. Emit event
                        print(f"{event_data['event']}: {identity if identity else 'Unknown'} ({score:.3f})")
                        event_emitter.emit(event_data)
                        
                        if identity is not None:
                            identity_last_seen[identity] = current_time
                        event_emitted = True
                    
                    if event_emitted:
                        decided_tracks.add(face.track_id)
            
        for face in tracked_faces:
            # 5. Visualization (Simplified)
            x1, y1, x2, y2 = face.bbox[:4].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {face.track_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Performance Logging
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        cv2.imshow('Ha-Meem AI Surveillance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline()