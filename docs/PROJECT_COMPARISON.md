# Project Comparison Report
## `ha_meem_ai_surveillance` (A) vs `ha_meem_ai_surveillance-main` (B)

> **Legend:**
> **Project A** → `e:\ha_meem_ai_surveillance` (current workspace, production branch)
> **Project B** → `e:\ha_meem_ai_surveillance-main` (reference / upstream snapshot)

---

## 1. Top-Level Structure

| Item | Project A | Project B |
|---|---|---|
| Root files | `.env`, `.env.example`, `AGENT_INSTRUCTIONS.md`, `README.md`, `requirements.txt` | `AGENT_CONTEXT.md`, `__init__.py`, `install_log.txt`, `touch ARCHITECTURE_LOCK.md` |
| Extra top-level dirs | `docker/`, `docs/`, `logs/`, `scripts/`, `snapshots/`, `trt_cache/` | `experiments/`, `tests/`, `requirements/` (folder) |
| Virtual env committed | `.venv/` present | Not present |
| Environment config | `.env` + `python-dotenv` | No `.env`; config via YAML only |
| Test suite | `pytest` in `requirements.txt`, no test files found | `tests/` directory exists (stub README only) |
| Docker support | `docker/` directory | `docker/` directory |

> Project A has a live `.venv` and a `.env` file indicating it is a running, deployed instance. Project B is a cleaner reference snapshot designed for development and handoff.

---

## 2. Architecture Overview

### Project A — Threaded Worker + Shared Queue

```
run_pipeline()
  ├── load_merged_configs([default, thresholds, tensorrt])
  ├── SCRFDDetector  ──┐
  ├── AdaFaceRecognizer─┤  (shared, thread-safe)
  ├── FaceDatabase    ──┘
  └── [CameraWorker(Thread) × N]
        ├── ByteTracker          (per-camera)
        ├── EmbeddingAggregator  (per-camera)
        ├── EventEmitter         (per-camera, synchronous)
        ├── SnapshotWriter       (per-camera, synchronous)
        └── frame_queue (maxsize=2) → main display grid
```

- `CameraWorker` **is** a `threading.Thread` subclass.
- I/O (snapshot + event) is **synchronous on the hot path** — disk writes happen inline during inference.
- Display is a single **grid window** composited in the main thread polling all queues.
- Camera reconnect logic is built into the worker.

### Project B — Threaded Worker + Dedicated Async I/O Thread

```
run_pipeline()
  ├── load_config([default, thresholds])
  ├── SharedModels
  │     ├── SCRFDDetector
  │     ├── AdaFaceRecognizer
  │     └── FaceDatabase
  └── [CameraWorker × N]  (plain objects, not Thread subclasses)
        ├── SORTTracker          (per-camera, custom Kalman)
        ├── EmbeddingAggregator  (per-camera, v2 with recency decay)
        ├── AdaptiveBlurThreshold(per-camera, rolling percentile)
        ├── PipelineState        (per-camera, encapsulated state machine)
        └── AsyncIOWorker        (per-camera, dedicated background thread)
              ├── EventEmitter
              └── SnapshotWriter

  Each worker runs in a named Thread (cam-<id>)
  Main thread owns all cv2.imshow() calls (per-camera windows)
  ROI: interactive mouse-draw with persist to cameras.yaml
```

> **Key difference:** Project B introduces `AsyncIOWorker` — a dedicated daemon thread with a 64-slot queue that handles snapshot saving and JSONL logging **completely off the inference hot path**. Project A does this synchronously, potentially stalling the pipeline on every event.

---

## 3. Model Pipeline — Step-by-Step

| Step | Project A | Project B |
|---|---|---|
| **1. Frame capture** | `CameraWorker.run()` — inline `cap.read()` with reconnect | `CameraWorker.run()` — inline `cap.read()`, no reconnect logic |
| **2. Detection** | `SCRFDDetector.detect(frame)` | `SCRFDDetector.detect(frame)` |
| **3. Tracking** | `ByteTracker` (wraps `supervision.ByteTrack`) | `SORTTracker` (custom Kalman + Hungarian assignment) |
| **4. ROI filtering** | ❌ None | ✅ Per-camera configurable ROI; drawn interactively |
| **5. Track expiry** | Manual: checks active IDs vs aggregator keys | Automatic: `aggregator.expire_stale_tracks()` returns expired IDs |
| **6. Quality gate** | Fixed `blur_threshold` (config constant) | Adaptive: `AdaptiveBlurThreshold` rolling 20th-percentile |
| **7. Face alignment** | `insightface.utils.face_align.norm_crop()` | Custom `align_face()` from `core/utils/image.py` |
| **8. Pose weighting** | ❌ Not applied | ✅ `pose_weight(face.kps)` × `size_factor` → `face.quality_score` |
| **9. Recognition** | Batched → returns `(embeddings, norms)` | Batched → returns `embeddings` only |
| **10. Aggregation** | Norm-weighted mean (AdaFace feature norms as weights) | Quality + recency weighted mean (`quality_score × recency_decay^age`) |
| **11. Matching** | `FaceDatabase.match(emb, threshold)` — top-1 FAISS | `FaceDatabase.match(emb, threshold, margin, top_k)` — top-K + margin test |
| **12. Decision / cooldown** | `decided_tracks` set + `identity_last_seen` dict (inline in worker) | `PipelineState` object (encapsulated, testable state machine) |
| **13. UNKNOWN upgrade** | ❌ Not supported | ✅ UNKNOWN tracks can be upgraded to AUTHORIZED on a later better frame |
| **14. Event emit** | Synchronous: `EventEmitter.emit()` + `SnapshotWriter.save()` on hot path | Async: `AsyncIOWorker.submit()` (non-blocking, queue-based) |
| **15. Display** | Single composited grid window | Separate named window per camera |

---

## 4. Tracking Module

| Aspect | Project A — `ByteTracker` | Project B — `SORTTracker` |
|---|---|---|
| Algorithm | ByteTrack via `supervision` library | Custom SORT (Simple Online Realtime Tracking) |
| State estimation | Kalman filter inside `supervision` | Custom `KalmanBoxTracker` (7-dim state: cx, cy, area, ratio, velocities) |
| Assignment | ByteTrack two-stage matching | Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) |
| External dependency | `supervision==0.27.0` required | Zero external tracking dependencies (pure NumPy + SciPy) |
| Configuration | Hard-coded in worker constructor | Configurable via `configs/thresholds.yaml` (`iou_threshold`, `max_age`) |
| Track coasting | `lost_track_buffer=30` frames | `max_age=15` frames (configurable, ~1s at 15fps) |

---

## 5. Embedding Aggregation (Fusion)

| Aspect | Project A — v1 | Project B — v2 |
|---|---|---|
| Buffer type | `deque(maxlen=buffer_size)` | `list` with manual pop(0), timestamped entries |
| Weight source | **AdaFace feature norm** (raw pre-normalisation magnitude) | **quality_score** = `blur × confidence × pose_weight × size_factor` |
| Recency weighting | ❌ None — all frames equal | ✅ Exponential decay: `decay^(n-1-i)` oldest→newest |
| Decision gate | Frame count only: `len(buffer) >= min_frames` | Time + frame count: `elapsed >= min_decision_seconds AND len >= min_frames` |
| Stale track cleanup | Manual: main loop checks active tracker IDs | Automatic: `expire_stale_tracks()` polls `last_updated` timestamps |
| Config exposure | `buffer_size`, `min_frames` | `buffer_size`, `min_frames`, `min_decision_seconds`, `recency_decay`, `expire_after_seconds` |

---

## 6. Quality Control

| Aspect | Project A | Project B |
|---|---|---|
| Blur scoring | `cv2.Laplacian().var()` | `cv2.Laplacian().var()` (identical) |
| Blur threshold | **Fixed** constant from config (`blur_threshold: 100`) | **Adaptive** `AdaptiveBlurThreshold` — rolling 20th-percentile |
| Adaptive window | ❌ | 500-frame rolling window per camera |
| Fallback | N/A | Falls back to static threshold until ≥10 samples collected |
| Blur feeds from | Only valid (sized) faces | **All** detected faces (threshold warms up faster) |
| Size gate | `min_face_size: 80px` | `min_face_size: 70px` (slightly more permissive) |
| Pose weighting | ❌ | ✅ `pose_weight()` penalises extreme head angles |
| Quality score | Not computed | Composite: `blur × detection_confidence × pose_weight × size_factor` |

---

## 7. Face Database & Matching

| Aspect | Project A | Project B |
|---|---|---|
| Backend | FAISS `IndexFlatIP` (required) | FAISS with **NumPy fallback** if faiss not installed |
| Search | `k=1` (top-1 only) | `top_k=10` (configurable), then identity-level aggregation |
| Identity grouping | Single embedding per query | Groups top-K hits by identity, takes max score per person |
| Margin test | ❌ None — first past threshold wins | ✅ Rejects if `top1_score - top2_score < match_margin` |
| Dynamic add | `add_identity()` method available | ❌ No dynamic add |
| Score logging | Only on match | **Always** logs raw score for near-miss threshold tuning |

---

## 8. Detection (SCRFD)

| Aspect | Project A | Project B |
|---|---|---|
| Backend | TensorRT via ORT `TensorrtExecutionProvider` | CUDA via ORT `CUDAExecutionProvider` |
| TRT FP16 | ✅ Enabled by default | ❌ Not used |
| Engine caching | ✅ `trt_cache/` directory with persistent engine | ❌ No caching |
| Provider injection | Manually injects ORT session into insightface wrapper | Uses insightface default session |
| Windows TRT fix | ✅ Prepends `tensorrt_libs` to `PATH` | ❌ Not needed |
| First-run penalty | ~120s TRT compile, then ~10ms cached | Instant (CUDA JIT only) |

> Project A trades first-run compile time for significantly higher sustained throughput via TRT FP16. Project B is faster to set up but has lower peak GPU utilization.

---

## 9. API Server

| Aspect | Project A | Project B |
|---|---|---|
| Event source | Reads JSONL file on every request | In-memory `deque(maxlen=1000)` pre-loaded at startup |
| Live updates | ❌ File re-read each call | ✅ Background `_tail_log_file()` thread pushes new events to cache |
| Server-Sent Events | ❌ Not supported | ✅ `/events/stream` SSE endpoint with keepalive |
| Filtering | ❌ None (limit only) | ✅ `camera_id`, `identity`, `event_type`, `since` query params |
| Environment config | `python-dotenv` for host/port | Hard-coded `0.0.0.0:8000` |
| Health endpoint | `{"status": "ok"}` | `{"status": "ok", "cached_events": N}` |
| Endpoints | `/`, `/health`, `/events/latest`, `/events` | `/`, `/health`, `/events/latest`, `/events`, `/events/stream` |

---

## 10. Configuration

| Config file | Project A | Project B |
|---|---|---|
| `default.yaml` | device, precision, batch_size, logging, models, dataset paths | device, logging, models only |
| `thresholds.yaml` | detection confidence+IoU, recognition threshold+size+blur+min_frames | Full: detection, recognition (+upgrade/match margins, top_k), fusion, tracking, cooldown, quality |
| `tensorrt.yaml` | ✅ Full TRT config: fp16, cache, workspace, DLA | ❌ Not present |
| `cameras.yaml` | id, url, enabled, resolution per camera | id, url, name, roi per camera |
| `dataset.yaml` | Embedded in `default.yaml` | ✅ Separate file |

---

## 11. Display & UX

| Aspect | Project A | Project B |
|---|---|---|
| Window layout | Single grid window — all cameras tiled | Separate named window per camera |
| ROI drawing | ❌ Not supported | ✅ Interactive mouse click-drag to define Gate ROI |
| ROI persistence | ❌ | ✅ Saved back to `configs/cameras.yaml` on release |
| ROI clear | ❌ | ✅ Press `r` to clear all ROIs |
| Log output | `print()` statements | Structured `logging` with timestamps and severity levels |

---

## 12. Full Feature Matrix

| Feature | Project A | Project B |
|---|---|---|
| TensorRT FP16 acceleration | ✅ | ❌ |
| CUDA-only mode | ✅ (fallback) | ✅ |
| CPU mode | ✅ | ✅ |
| ByteTrack | ✅ | ❌ |
| SORT (Kalman + Hungarian) | ❌ | ✅ |
| Configurable tracker params | ❌ (hard-coded) | ✅ |
| Interactive ROI gate | ❌ | ✅ |
| Adaptive blur threshold | ❌ | ✅ |
| Pose-weighted quality score | ❌ | ✅ |
| Recency-weighted aggregation | ❌ | ✅ |
| Time-gated decision | ❌ | ✅ |
| Self-expiring track buffers | ❌ | ✅ |
| UNKNOWN → AUTHORIZED upgrade | ❌ | ✅ |
| Top-K match + margin test | ❌ | ✅ |
| FAISS fallback (no-faiss env) | ❌ | ✅ |
| Async I/O (off hot path) | ❌ | ✅ |
| Camera auto-reconnect | ✅ | ❌ |
| SSE live event stream API | ❌ | ✅ |
| API event filtering | ❌ | ✅ |
| In-memory event cache | ❌ | ✅ |
| Structured logging | ❌ (print) | ✅ |
| Per-camera named windows | ❌ | ✅ |
| `python-dotenv` env config | ✅ | ❌ |
| Docker support | ✅ | ✅ |

---

## 13. Recommendation

### Where Project A is Stronger
1. **TensorRT FP16** — both SCRFD and AdaFace run with TRT engine caching; significantly higher GPU throughput.
2. **Camera reconnect** — the worker retries stream open on failure, making it resilient for real IP cameras.
3. **Environment management** — `.env` + `python-dotenv` keeps secrets out of YAML.
4. **AdaFace norm preservation** — returns raw norms alongside embeddings, enabling norm-weighted aggregation.

### Where Project B is Stronger
1. **Decoupled I/O** — `AsyncIOWorker` keeps disk writes off the inference loop; Project A stalls on every event.
2. **Adaptive quality** — `AdaptiveBlurThreshold` self-tunes to each camera's conditions with no manual calibration.
3. **Richer matching** — top-K + margin test prevents false accepts from near-tie gallery scores.
4. **UNKNOWN upgrade** — a face initially unknown can be upgraded to authorized on a better-quality frame.
5. **PipelineState encapsulation** — decided/cooldown state is testable in isolation.
6. **Interactive ROI** — click-drag gate zones with persistence; very useful for entry-point surveillance.
7. **Richer API** — SSE live stream + filtering makes the API useful for real-time dashboards.
8. **Structured logging** — timestamps and severity levels throughout.
9. **Zero-dependency tracker** — SORT is self-contained; Project A requires the `supervision` library.

### Ideal Merge Strategy
Combine the best of both:
- **From Project A:** TensorRT pipeline, camera reconnect, norm-weighted aggregation.
- **From Project B:** AsyncIOWorker, AdaptiveBlurThreshold, PipelineState, ROI gate, top-K margin matching, SSE API.
