import cv2
import time
import threading
import queue
from datetime import datetime
import numpy as np

from core.events import EventEmitter, SnapshotWriter
from core.tracking import ByteTracker
from core.fusion import EmbeddingAggregator
from core.quality import calculate_blur_score

class CameraWorker(threading.Thread):
    def __init__(self, camera_id, camera_url, detector, recognizer, face_db, config, resolution=None):
        super().__init__()
        self.camera_id = camera_id
        self.camera_url = camera_url
        self.resolution = resolution
        
        # Shared Models
        self.detector = detector
        self.recognizer = recognizer
        self.face_db = face_db
        self.config = config
        
        # Per-camera State
        self.tracker = ByteTracker(track_activation_threshold=0.25, lost_track_buffer=30)
        self.aggregator = EmbeddingAggregator(
            buffer_size=10, 
            min_frames=config['recognition']['min_frames_for_decision']
        )
        self.event_emitter = EventEmitter(
            camera_id=self.camera_id,
            log_file="logs/events.jsonl"
        )
        self.snapshot_writer = SnapshotWriter(
            base_dir="snapshots",
            camera_id=self.camera_id
        )
        self.decided_tracks = set()
        self.identity_last_seen = {}
        self.identity_cooldown_seconds = 6
        
        # Output Queue for Display
        self.frame_queue = queue.Queue(maxsize=2)
        
        self.stop_event = threading.Event()
        self.cap = None
        self._initialize_camera()

    def _initialize_camera(self):
        print(f"[{self.camera_id}] Initializing stream: {self.camera_url}")
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_url)
        
        if self.resolution:
            print(f"[{self.camera_id}] Setting resolution to {self.resolution['width']}x{self.resolution['height']}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution['height'])
        
    def _handle_reconnect(self):
        print(f"[{self.camera_id}] Stream read failed. Attempting reconnect in 2 seconds...")
        time.sleep(2)
        self._initialize_camera()

    def stop(self):
        self.stop_event.set()

    def run(self):
        print(f"[{self.camera_id}] Worker thread started.")
        while not self.stop_event.is_set():
            if not self.cap.isOpened():
                self._handle_reconnect()
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._handle_reconnect()
                continue
                
            start_time = time.time()
            
            # 1. Detection
            faces = self.detector.detect(frame)
            
            # 2. Tracking
            tracked_faces = self.tracker.update(faces)
            
            active_track_ids = set(face.track_id for face in tracked_faces)
            for track_id in list(self.aggregator.track_buffers.keys()):
                if track_id not in active_track_ids:
                    self.aggregator.clear_track(track_id)
                    self.decided_tracks.discard(track_id)
                    
            valid_faces = []
            valid_face_imgs = []
            
            for face in tracked_faces:
                # Operational Constraints: Resolution Gate
                if face.width < self.config['recognition']['min_face_size']:
                    continue
                    
                # Blur Rejection
                x1, y1, x2, y2 = face.bbox[:4].astype(int)
                bbox_crop = frame[max(0, y1):y2, max(0, x1):x2]
                
                face.blur_score = calculate_blur_score(bbox_crop)
                if face.blur_score < self.config['recognition']['blur_threshold']:
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
                embeddings = self.recognizer.extract_embeddings_batch(valid_face_imgs)
                
                for face, embedding in zip(valid_faces, embeddings):
                    face.embedding = embedding
                    
                    # 4. Fusion (Aggregation)
                    self.aggregator.add_face(face)
                    
                    # Get consensus
                    consensus_emb = self.aggregator.get_aggregated_embedding(face.track_id)
                    
                    if consensus_emb is not None and face.track_id not in self.decided_tracks:
                        identity, score = self.face_db.match(
                            consensus_emb, 
                            self.config['recognition']['similarity_threshold']
                        )
                        
                        current_time = time.time()
                        
                        event_emitted = False
                        
                        # Check for cooldown if authorized
                        can_emit = True
                        if identity is not None:
                            last_seen = self.identity_last_seen.get(identity, 0)
                            if current_time - last_seen < self.identity_cooldown_seconds:
                                can_emit = False
                        
                        if can_emit:
                            # 1. Create a single source of truth for time
                            event_time = datetime.now()
                            
                            # 2. Create event object
                            event_data = {
                                "timestamp": event_time.isoformat(),
                                "camera_id": self.camera_id,
                                "track_id": face.track_id,
                                "identity": identity,
                                "score": float(score),
                                "event": "AUTHORIZED" if identity else "UNKNOWN"
                            }
                            
                            # 3. Capture and save snapshot
                            # x1, y1, x2, y2 from face.bbox for snapshot bounding box
                            x1, y1, x2, y2 = face.bbox[:4].astype(int)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, identity or "UNKNOWN", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            snapshot_path = self.snapshot_writer.save(frame, identity, timestamp=event_time)
                            event_data["snapshot"] = snapshot_path
                            
                            # 4. Emit event
                            print(f"[{self.camera_id}] {event_data['event']}: {identity if identity else 'Unknown'} ({score:.3f})")
                            self.event_emitter.emit(event_data)
                            
                            if identity is not None:
                                self.identity_last_seen[identity] = current_time
                            event_emitted = True
                        
                        if event_emitted:
                            self.decided_tracks.add(face.track_id)
                
            for face in tracked_faces:
                # 5. Visualization (Simplified)
                x1, y1, x2, y2 = face.bbox[:4].astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {face.track_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Performance Logging
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(frame, f"[{self.camera_id}] FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Put the latest processed frame in the queue (drop oldest if full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(frame)

        print(f"[{self.camera_id}] Worker thread stopping. Releasing resources.")
        if self.cap:
            self.cap.release()
