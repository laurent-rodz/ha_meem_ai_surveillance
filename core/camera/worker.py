import logging
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
from core.quality.blur import AdaptiveBlurThreshold
from core.io_worker import AsyncIOWorker
from core.pipeline_state import PipelineState
from core.utils.image import pose_weight
from core.utils.pose_estimator import PoseEstimator

log = logging.getLogger(__name__)

_DIAG_INTERVAL = 5.0  # seconds between periodic diagnostic summaries


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
        tracking_config = self.config.get('tracking', {})
        self.tracker = ByteTracker(
            track_activation_threshold=tracking_config.get('track_activation_threshold', 0.25),
            lost_track_buffer=tracking_config.get('lost_track_buffer', 30),
            minimum_matching_threshold=tracking_config.get('minimum_matching_threshold', 0.8)
        )
        
        # Pose estimator
        self.pose_estimator = PoseEstimator()
        
        # Fusion config
        fusion_config = self.config.get('fusion', {})
        self.aggregator = EmbeddingAggregator(
            buffer_size=fusion_config.get('buffer_size', 10),
            min_frames=fusion_config.get('min_frames', 6),
            min_decision_seconds=fusion_config.get('min_decision_seconds', 0.3),
            recency_decay=fusion_config.get('recency_decay', 0.95),
            expire_after_seconds=fusion_config.get('expire_after_seconds', 5.0)
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
        
        # Adaptive blur threshold per camera
        quality_config = self.config.get('quality', {}).get('adaptive_blur', {})
        if quality_config.get('enabled', True):
            self.adaptive_blur = AdaptiveBlurThreshold(
                window_size=quality_config.get('window_size', 500),
                percentile=quality_config.get('percentile', 20.0),
                fallback=quality_config.get('fallback', self.config['recognition'].get('blur_threshold', 100)),
                min_samples=quality_config.get('min_samples', 10)
            )
        else:
            self.adaptive_blur = None
        
        # Output Queue for Display
        self.frame_queue = queue.Queue(maxsize=2)
        
        self.stop_event = threading.Event()
        self.cap = None
        self._last_diag_time = time.time()
        self._diag_frames = 0
        self._diag_detections = 0
        self._diag_tracked = 0
        self._diag_size_rejected = 0
        self._diag_pose_rejected = 0
        self._diag_blur_rejected = 0
        self._diag_fusion_pending = 0
        self._diag_events = 0
        self._diag_pose_yaw_samples: list = []
        self._initialize_camera()

    def _initialize_camera(self):
        log.info(f"[{self.camera_id}] Initializing stream: {self.camera_url}")
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_url)
        
        if self.resolution:
            log.info(f"[{self.camera_id}] Setting resolution to {self.resolution['width']}x{self.resolution['height']}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution['height'])
        
    def _handle_reconnect(self):
        log.warning(f"[{self.camera_id}] Stream read failed. Attempting reconnect in 2 seconds...")
        time.sleep(2)
        self._initialize_camera()

    def stop(self):
        self.stop_event.set()
        if hasattr(self, 'io_worker'):
            self.io_worker.stop()

    def run(self):
        log.info(f"[{self.camera_id}] Worker thread started.")
        while not self.stop_event.is_set():
            if not self.cap.isOpened():
                self._handle_reconnect()
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._handle_reconnect()
                continue
                
            start_time = time.time()
            self._diag_frames += 1
            
            # 1. Detection
            faces = self.detector.detect(frame)
            self._diag_detections += len(faces)
            
            # 2. Tracking
            tracked_faces = self.tracker.update(faces)
            self._diag_tracked += len(tracked_faces)
            log.debug(f"[{self.camera_id}] Detected={len(faces)} Tracked={len(tracked_faces)}")
            
            # 2.5 Pose Estimation
            for face in tracked_faces:
                if face.kps is not None and len(face.kps) == 5:
                    face.pitch, face.yaw, face.roll = self.pose_estimator.estimate_pose(face.kps, frame.shape)
            
            # Expire stale tracks based on time (replaces manual tracker-based cleanup)
            expired_tracks = self.aggregator.expire_stale_tracks()
            for track_id in expired_tracks:
                self.pipeline_state.release_track(track_id)
                    
            valid_faces = []
            valid_face_imgs = []
            
            max_yaw = self.config.get('recognition', {}).get('max_yaw', 30.0)
            max_pitch = self.config.get('recognition', {}).get('max_pitch', 30.0)
            min_face_size = self.config['recognition']['min_face_size']

            for face in tracked_faces:
                # Operational Constraints: Resolution Gate
                if face.width < min_face_size:
                    log.debug(f"[{self.camera_id}] track={face.track_id} REJECTED size={face.width:.0f}px < {min_face_size}px")
                    self._diag_size_rejected += 1
                    continue
                
                # Pose Rejection
                if abs(face.yaw) > max_yaw or abs(face.pitch) > max_pitch:
                    log.debug(f"[{self.camera_id}] track={face.track_id} REJECTED pose yaw={face.yaw:.1f} pitch={face.pitch:.1f}")
                    self._diag_pose_rejected += 1
                    self._diag_pose_yaw_samples.append(abs(face.yaw))
                    continue
                     
                # Blur Rejection
                x1, y1, x2, y2 = face.bbox[:4].astype(int)
                bbox_crop = frame[max(0, y1):y2, max(0, x1):x2]
                
                face.blur_score = calculate_blur_score(bbox_crop)
                
                # Compute composite quality score: blur × confidence × pose × size
                confidence = getattr(face, 'confidence', 0.5)
                pose_w = pose_weight(face.kps) if (face.kps is not None and len(face.kps) >= 3) else 0.5
                size_factor = min(face.width / min_face_size, 1.0)
                face.quality_score = face.blur_score * confidence * pose_w * size_factor
                
                # Update adaptive blur threshold if enabled
                if self.adaptive_blur is not None:
                    self.adaptive_blur.update(face.blur_score)
                    blur_threshold = self.adaptive_blur.threshold()
                else:
                    blur_threshold = self.config['recognition']['blur_threshold']
                
                if face.blur_score < blur_threshold:
                    log.debug(f"[{self.camera_id}] track={face.track_id} REJECTED blur={face.blur_score:.1f} < threshold={blur_threshold:.1f}")
                    self._diag_blur_rejected += 1
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
                    
                    if consensus_emb is None:
                        buf = self.aggregator.track_buffers.get(face.track_id, {})
                        n_frames = len(buf.get('entries', []))
                        elapsed = time.time() - buf.get('first_seen', time.time())
                        log.debug(f"[{self.camera_id}] track={face.track_id} fusion pending: frames={n_frames}/{self.aggregator.min_frames} elapsed={elapsed:.2f}s/{self.aggregator.min_decision_seconds}s")
                        self._diag_fusion_pending += 1

                    # Process if: not decided yet, or upgradeable (UNKNOWN → AUTHORIZED)
                    if consensus_emb is not None and (
                        not self.pipeline_state.is_decided(face.track_id) or
                        self.pipeline_state.is_upgradeable(face.track_id)
                    ):
                        identity, score = self.face_db.match(
                            consensus_emb, 
                            self.config['recognition']['similarity_threshold'],
                            match_margin=self.config['recognition'].get('match_margin', 0.05),
                            match_top_k=self.config['recognition'].get('match_top_k', 10)
                        )
                        log.debug(f"[{self.camera_id}] track={face.track_id} match: identity={identity} score={score:.3f}")
                        
                        # Skip if we already know this track as AUTHORIZED
                        if identity and self.pipeline_state.is_decided(face.track_id) and \
                           not self.pipeline_state.is_upgradeable(face.track_id):
                            continue
                        
                        event_emitted = False
                        
                        # Check for cooldown
                        if self.pipeline_state.can_alert(identity, face.track_id):
                            # Determine if this is an upgrade scenario
                            upgradeable = identity and self.pipeline_state.is_upgradeable(face.track_id)
                            threshold = self.config['recognition']['similarity_threshold']
                            upgrade_margin = self.config['recognition'].get('upgrade_margin', 0.05)
                            
                            # For upgrade, require higher confidence (threshold + margin)
                            if upgradeable and score < threshold + upgrade_margin:
                                continue  # Not confident enough to upgrade
                            
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
                            log.info(f"[{self.camera_id}] {event_data['event']}: {identity if identity else 'Unknown'} ({score:.3f})")
                            self.io_worker.submit(frame, event_data, identity, event_time)
                            self._diag_events += 1
                            
                            # Upgrade track if previously UNKNOWN with sufficient confidence
                            if upgradeable and score >= threshold + upgrade_margin:
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

            # Periodic diagnostic summary (INFO level, every _DIAG_INTERVAL seconds)
            now = time.time()
            if now - self._last_diag_time >= _DIAG_INTERVAL:
                log.info(
                    f"[{self.camera_id}] DIAG — frames={self._diag_frames} "
                    f"det={self._diag_detections} tracked={self._diag_tracked} "
                    f"rejected(size={self._diag_size_rejected} pose={self._diag_pose_rejected} "
                    f"blur={self._diag_blur_rejected} fusion={self._diag_fusion_pending}) "
                    f"events={self._diag_events}"
                    + (f" | pose_yaw_avg={sum(self._diag_pose_yaw_samples)/len(self._diag_pose_yaw_samples):.1f}° max={max(self._diag_pose_yaw_samples):.1f}°"
                       if self._diag_pose_yaw_samples else "")
                )
                # Reset counters
                self._diag_frames = self._diag_detections = self._diag_tracked = 0
                self._diag_size_rejected = self._diag_pose_rejected = 0
                self._diag_blur_rejected = self._diag_fusion_pending = self._diag_events = 0
                self._diag_pose_yaw_samples.clear()
                self._last_diag_time = now
            
            # Put the latest processed frame in the queue (drop oldest if full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put_nowait(frame)

        log.info(f"[{self.camera_id}] Worker thread stopping. Releasing resources.")
        if self.cap:
            self.cap.release()
