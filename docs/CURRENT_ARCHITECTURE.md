# Ha-Meem AI Surveillance Architecture Pipeline

## Project Overview
Ha-Meem AI Surveillance is a production-grade, real-time face recognition system designed for factory environments. It processes video streams from multiple CCTV cameras to:
- Detect and identify authorized personnel at entry gates
- Track individuals across video frames
- Generate alerts for unknown persons
- Produce structured logs and visual reports

## Directory Structure
```
ha_meem_ai_surveillance/
├── apps/                    # High-level applications
│   ├── entry_pipeline/      # Main single/multi-camera pipeline
│   ├── api_server/          # FastAPI REST API with SSE streaming
│   ├── alert_bot/           # WhatsApp alert bot (Selenium-based)
│   └── dataset_tools/       # Face extraction and gallery building
├── core/                    # Core AI modules
│   ├── camera/              # Camera worker thread
│   ├── detection/           # SCRFD face detector
│   ├── recognition/         # AdaFace recognizer
│   ├── tracking/            # ByteTrack tracker (via supervision)
│   ├── fusion/              # Multi-frame embedding aggregation
│   ├── database/            # FAISS vector database
│   ├── events/              # Event emission and snapshot writing
│   ├── quality/             # Blur detection and adaptive thresholding
│   └── utils/               # Pose estimation and image utilities
├── configs/                 # YAML configuration files
├── models/                  # ONNX/TensorRT model storage
├── data/                    # Raw frames and face datasets
├── scripts/                 # Smoke tests and report generation
├── docs/                    # Documentation and implementation plans
├── logs/                    # Event logs (JSONL format)
├── snapshots/               # Saved face snapshots by date
└── trt_cache/              # TensorRT engine cache
```

## Core Components
| Module | File(s) | Purpose |
|--------|---------|---------|
| **CameraWorker** | `core/camera/worker.py` | Per-camera thread handling the full pipeline (detection → tracking → recognition → events) |
| **SCRFDDetector** | `core/detection/scrfd_detector.py` | Face detection using SCRFD model with ONNX Runtime + TensorRT |
| **AdaFaceRecognizer** | `core/recognition/adaface_recognizer.py` | Face recognition producing 512-d embeddings |
| **ByteTracker** | `core/tracking/byte_tracker.py` | Multi-object tracking using supervision's ByteTrack |
| **EmbeddingAggregator** | `core/fusion/aggregator.py` | Temporal fusion of embeddings with recency weighting |
| **FaceDatabase** | `core/database/face_database.py` | FAISS-based vector database for identity matching |
| **PoseEstimator** | `core/utils/pose_estimator.py` | Head pose estimation (yaw, pitch, roll) via cv2.solvePnP |
| **AsyncIOWorker** | `core/io_worker.py` | Background thread for non-blocking snapshot saving and event logging |
| **EventEmitter** | `core/events/event_emitter.py` | Appends JSON events to log file |
| **SnapshotWriter** | `core/events/snapshot_writer.py` | Saves cropped face images organized by date |
| **PipelineState** | `core/pipeline_state.py` | Tracks decision state per track (AUTHORIZED/UNKNOWN) with cooldowns |
| **AdaptiveBlurThreshold** | `core/quality/blur.py` | Per-camera adaptive blur threshold using rolling percentile |

### Component Interactions
```
CameraWorker (per camera thread)
    ├── SCRFDDetector.detect(frame) → List[Face]
    ├── ByteTracker.update(faces) → tracked Faces
    ├── PoseEstimator.estimate_pose(kps, frame) → (pitch, yaw, roll)
    ├── AdaptiveBlurThreshold + calculate_blur_score()
    ├── AdaFaceRecognizer.extract_embeddings_batch(faces) → embeddings
    ├── EmbeddingAggregator.add_face(face) + get_aggregated_embedding()
    ├── FaceDatabase.match(embedding) → (identity, score)
    ├── PipelineState (decision tracking, cooldowns)
    └── AsyncIOWorker.submit(frame, event_data) → EventEmitter + SnapshotWriter
```

## Data Flow/Pipeline Architecture

### Real-Time Inference Pipeline
```
                    ┌─────────────────────────────────────────────────┐
                    │           CameraWorker (Thread)                 │
                    │                                                 │
RTSP/Video ───────►│  1. cv2.VideoCapture.read() → frame           │
                    │                                                 │
                    │  2. SCRFDDetector.detect(frame)                │
                    │     ↓                                           │
                    │     List[Face] (bbox, kps, confidence)         │
                    │                                                 │
                    │  3. ByteTracker.update(faces)                  │
                    │     ↓                                           │
                    │     tracked_faces (with track_id)               │
                    │                                                 │
                    │  4. PoseEstimator.estimate_pose(kps)           │
                    │     + quality gates (blur, size, yaw/pitch)    │
                    │     ↓                                           │
                    │     valid_faces + valid_face_imgs               │
                    │                                                 │
                    │  5. AdaFaceRecognizer.extract_embeddings_batch()│
                    │     ↓                                           │
                    │     embeddings (512-d normalized)               │
                    │                                                 │
                    │  6. EmbeddingAggregator (per track)            │
                    │     - Recency-weighted fusion                  │
                    │     - Time-based decision gate                 │
                    │     ↓                                           │
                    │     consensus_embedding                         │
                    │                                                 │
                    │  7. FaceDatabase.match(embedding)              │
                    │     ↓                                           │
                    │     (identity, score) or (None, score)         │
                    │                                                 │
                    │  8. PipelineState (cooldown, upgrade logic)    │
                    │     ↓                                           │
                    │     Event: AUTHORIZED / UNKNOWN                │
                    │                                                 │
                    │  9. AsyncIOWorker.submit()                     │
                    │     ├── SnapshotWriter.save() → JPEG           │
                    │     └── EventEmitter.emit() → events.jsonl     │
                    │                                                 │
                    └─────────────────────────────────────────────────┘
                            ↓
                    ┌─────────────────────────────────────────────────┐
                    │              FastAPI Server                      │
                    │  - GET /events (filtered queries)              │
                    │  - GET /events/stream (SSE real-time)          │
                    │  - GET /health, /events/latest                │
                    └─────────────────────────────────────────────────┘
                            ↓
                    ┌─────────────────────────────────────────────────┐
                    │            WhatsApp Bot (Polling)                │
                    │  Polls API → sends alerts via WhatsApp Web     │
                    └─────────────────────────────────────────────────┘
```

### Multi-Frame Fusion Logic
The system requires multiple frames before making a recognition decision:
1. **Buffer per track**: Stores up to `buffer_size` (10) embedding+quality+scores
2. **Time gate**: Minimum `min_decision_seconds` (0.3s) of observation
3. **Frame gate**: Minimum `min_frames` (6) before decision
4. **Recency weighting**: Newer frames weighted higher (decay=0.95)
5. **Quality weighting**: Blur × confidence × pose × size
6. **Upgrade path**: UNKNOWN tracks can upgrade to AUTHORIZED with higher confidence

## Technology Stack
| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.10.11 | Main development language |
| **Vision/AI** | OpenCV 4.13 | Image processing, video I/O, Laplacian blur |
| **Inference** | ONNX Runtime 1.23 (GPU) | Model inference with TensorRT backend |
| **Optimization** | TensorRT 10.12 (FP16) | GPU acceleration, engine caching |
| **Detection** | SCRFD (insightface 0.7.3) | Face detection with landmarks |
| **Recognition** | AdaFace | 512-d face embeddings |
| **Tracking** | supervision 0.27 + ByteTrack | Multi-object tracking with Kalman filter |
| **Vector DB** | FAISS 1.13 (CPU) | Cosine similarity search (IndexFlatIP) |
| **API** | FastAPI 0.135 + Uvicorn | REST API, SSE streaming |
| **Validation** | Pydantic 2.13 | Request/response models |
| **Alerts** | Selenium + pyautogui | WhatsApp Web automation |
| **Config** | PyYAML 6.0 + python-dotenv | YAML configs with env var substitution |
| **Numerics** | NumPy 2.2 | Array operations, embedding math |
| **Metrics** | scikit-learn 1.7 | (Available for extensions) |

## Entry Points & Execution Flow

### Primary Entry Points
| Command | Module | Purpose |
|---------|--------|---------|
| `py -m apps.entry_pipeline.main` | Single/Multi-camera pipeline | Main real-time surveillance |
| `py -m apps.api_server.main` | FastAPI server | REST API and SSE streaming |
| `py -m apps.alert_bot.whatsapp_bot` | WhatsApp bot | Polls API, sends alerts |
| `py -m apps.dataset_tools.extract_faces` | Face extraction | Build gallery from raw images |
| `py -m apps.dataset_tools.build_gallery` | Gallery builder | Create FAISS embeddings DB |

### Initialization Flow (`apps.entry_pipeline.main`)
1. `load_merged_configs()`
   - `configs/default.yaml` (device, models, dataset paths)
   - `configs/thresholds.yaml` (detection/recognition thresholds)
   - `configs/tensorrt.yaml` (TRT settings)
2. `load_config('configs/cameras.yaml')` → Parse enabled cameras with RTSP URLs
3. Initialize shared components (thread-safe):
   - `SCRFDDetector(config, model_path)`
   - `AdaFaceRecognizer(config, model_path)`
   - `FaceDatabase(gallery_embeddings.npy)`
4. For each enabled camera, create `CameraWorker`:
   - Per-camera: ByteTracker, PoseEstimator, EmbeddingAggregator
   - Per-camera: PipelineState, AdaptiveBlurThreshold
   - Shared: EventEmitter, SnapshotWriter, AsyncIOWorker
   - Start worker thread (`daemon=True`)
5. Main display loop:
   - Poll `frame_queue` from each worker
   - Build grid layout for multiple cameras
   - `cv2.imshow()` with camera labels
   - Press `q` to exit → stop all workers

### CameraWorker Run Loop (Per Camera)
```python
while not stop_event.is_set():
    1. ret, frame = cap.read()
    2. faces = detector.detect(frame)                    # SCRFD
    3. tracked_faces = tracker.update(faces)             # ByteTrack
    4. pose_estimator.estimate_pose(face.kps, frame)     # yaw/pitch/roll
    5. Quality gates:
       - face.width >= min_face_size (80px)
       - abs(yaw) <= max_yaw (30°), abs(pitch) <= max_pitch
       - blur_score >= adaptive_threshold
    6. recognizer.extract_embeddings_batch(faces)        # AdaFace batch
    7. aggregator.add_face(face)                        # Multi-frame buffer
    8. consensus_emb = aggregator.get_aggregated_embedding(track_id)
    9. if consensus_emb:
       identity, score = face_db.match(emb, threshold)
       → PipelineState: cooldown, upgrade logic
       → AsyncIOWorker.submit(frame, event) → save + emit
    10. Visualization: cv2.rectangle, cv2.putText
    11. frame_queue.put(frame) for display
```

## Key Design Decisions
- **Production Stability**: Async I/O for non-blocking snapshot/event operations, adaptive quality thresholds, time-based fusion gates
- **Multi-Frame Fusion**: Requires multiple frames to reduce false positives, with recency and quality weighting
- **Modular Architecture**: Separation of core AI modules, applications, and configuration for easy maintenance
- **GPU Acceleration**: TensorRT + ONNX Runtime for optimized inference on supported hardware
