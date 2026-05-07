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
from core.io_worker import AsyncIOWorker
from core.pipeline_state import PipelineState

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
        self.io_worker = AsyncIOWorker(self.event_emitter, self.snapshot_writer)
        self.pipeline_state = PipelineState(
            camera_id=self.camera_id,
            cooldown_seconds=self.config.get('recognition', {}).get('cooldown_seconds', 6)
        )
        
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
        if hasattr(self, 'io_worker'):
            self.io_worker.stop()

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
                    self.pipeline_state.release_track(track_id)
                    
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
                embeddings, norms = self.recognizer.extract_embeddings_batch(valid_face_imgs)
                
                for face, embedding, norm in zip(valid_faces, embeddings, norms):
                    face.embedding = embedding
                    face.feature_norm = norm
                    
                    # 4. Fusion (Aggregation)
                    self.aggregator.add_face(face)
                    
                    # Get consensus
                    consensus_emb = self.aggregator.get_aggregated_embedding(face.track_id)
                    
                    # Process if: not decided yet, or upgradeable (UNKNOWN → AUTHORIZED)
                    if consensus_emb is not None and (
                        not self.pipeline_state.is_decided(face.track_id) or
                        self.pipeline_state.is_upgradeable(face.track_id)
                    ):
                        identity, score = self.face_db.match(
                            consensus_emb, 
                            self.config['recognition']['similarity_threshold']
                        )
                        
                        # Skip if we already know this track as AUTHORIZED
                        if identity and self.pipeline_state.is_decided(face.track_id) and \
                           not self.pipeline_state.is_upgradeable(face.track_id):
                            continue
                        
                        event_emitted = False
                        
                        # Check for cooldown
                        if self.pipeline_state.can_alert(identity, face.track_id):
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
                            # Store bbox [x1,y1,x2,y2] so report.py can crop without re-running SCRFD
                            x1, y1, x2, y2 = face.bbox[:4].astype(int)
                            event_data["bbox"] = [x1, y1, x2, y2]
                            
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, identity or "UNKNOWN", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
                            # 4. Emit event asynchronously
                            print(f"[{self.camera_id}] {event_data['event']}: {identity if identity else 'Unknown'} ({score:.3f})")
                            self.io_worker.submit(frame, event_data, identity, event_time)
                            
                            # Upgrade track if previously UNKNOWN, else mark as decided
                            if identity and self.pipeline_state.is_upgradeable(face.track_id):
                                self.pipeline_state.upgrade_track(face.track_id, identity)
                            else:
                                self.pipeline_state.mark_decided(face.track_id, identity)
                            event_emitted = True
                
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
