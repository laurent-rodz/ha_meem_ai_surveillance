# Ha-Meem AI Surveillance — Full Project Analysis & Recommendations

> Analysed: All source modules across `core/`, `apps/`, `configs/`, `docker/`, and project metadata.

---

## 1. Current Architecture Snapshot

```
RTSP / Video File
      │
      ▼
SCRFDDetector (ONNX via insightface)
      │ face bboxes + landmarks
      ▼
IOUTracker          ← assigns track IDs
      │ tracked faces
      ▼
QualityGate         ← width gate + blur score (Laplacian var)
      │ qualified faces
      ▼
AdaFaceRecognizer   ← ONNX, extracts 512-d embedding per frame
      │ embeddings
      ▼
EmbeddingAggregator ← mean pool over N frames (min 8)
      │ consensus embedding
      ▼
FaceDatabase        ← cosine sim against gallery (.npy, in-memory)
      │ identity / UNKNOWN
      ▼
EventEmitter → logs/events.jsonl
      │
      ├── SnapshotWriter → snapshots/
      │
      └── FastAPI Server → WhatsAppBot (Selenium polling)
```

---

## 2. What Is Already Good ✅

| Area | Strength |
|---|---|
| Architecture | Clean modular separation (`core/` vs `apps/`). No spaghetti. |
| Config system | All thresholds & paths in YAML — no hardcoded magic numbers. |
| Multi-frame fusion | `EmbeddingAggregator` enforces min-frame decision. Correct design. |
| Quality gating | Blur rejection + min face size gate — reduces false positives. |
| Face dataclass | Typed, clean `Face` dataclass with useful properties. |
| ONNX runtime | Both models use ONNX + CUDA provider — good portability. |
| Event log | JSONL flat-file — simple, appendable, readable. |
| API layer | FastAPI is the right choice — async, typed, fast. |

---

## 3. Critical Issues & Weaknesses ⚠️

### 3.1 Tracker — `IOUTracker` is fragile
**File:** `core/tracking/iou_tracker.py`

- **O(N×M) brute-force matching** — no Hungarian algorithm. Fine for 2 cameras / <10 faces. Will break at 20 cameras.
- **No Kalman filter** — track positions are not predicted between frames. A person walking fast or briefly occluded will lose their track and get a new ID.
- **max_age=5** is very low — one missed detection kills the track.
- The `track_buffers` cleanup in `main.py` is done **manual and inline** instead of being a tracker responsibility.

### 3.2 Recognition — single request per frame, no batching
**File:** `core/recognition/adaface_recognizer.py`

- `extract_embedding()` runs one face at a time. When 5–10 faces are in frame simultaneously, you make 5–10 sequential ONNX calls. GPU is barely utilized.
- No **face alignment** using the landmarks (`kps`) that SCRFD already provides. You crop raw bboxes and resize. This significantly hurts accuracy for tilted/angled faces (which your AGENT_CONTEXT says is *the* primary operating condition: 15°–25° tilt).

### 3.3 Database — flat `.npy`, no indexing
**File:** `core/database/face_database.py`

- Linear cosine scan `np.dot(stored_embeddings, query)` is fine for <100 people. At 500+ workers, this becomes noticeable.
- The gallery is a plain `.npy` dictionary loaded entirely into RAM on startup. Adding a new person requires rebuilding the full gallery and restarting the pipeline.

### 3.4 Fusion — simple mean pooling, no quality weighting
**File:** `core/fusion/aggregator.py`

- All embeddings in the buffer are treated equally. A blurry frame embed has the same weight as a sharp one.
- Using `list.pop(0)` to trim the buffer is O(N) — use `collections.deque(maxlen=N)` instead.

### 3.5 Pipeline — single-threaded, blocking main loop
**File:** `apps/entry_pipeline/main.py`

- `cap.read()` → `detect()` → `recognize()` → `match()` all run **serially on one thread**.
- If recognition takes 30ms and detection takes 20ms, you're capped at ~20 FPS total. At 20 cameras, this is unsustainable.
- The embedding aggregator's `track_buffers` dict is accessed and mutated in the same loop with no thread safety.
- Config is loaded 3 times separately (`load_config` repeated in each tool) using duplicated boilerplate code.

### 3.6 Alert System — Selenium WhatsApp bot is fragile
**File:** `apps/alert_bot/whatsapp_bot.py`

- Selenium scraping WhatsApp Web is the most fragile part of the entire system. WhatsApp regularly changes their DOM selectors and you already have 7 brittle XPath/CSS selectors hardcoded.
- Hard-coded phone number in the source file.
- Uses `pyautogui.press("escape")` to dismiss a Windows file dialog — this will fail in headless/server mode.
- Polling with 2-second intervals via HTTP is not real-time. Events can be up to 2s late.
- Queue `maxsize=3` means events are silently dropped under load.

### 3.7 Event log — file read on every API request
**File:** `apps/api_server/main.py`

- `read_events()` opens and reads the entire JSONL file on every GET. If the file grows to 100k events, this is very slow.
- No pagination cursor, no filtering by camera, no time-range filter.

### 3.8 Incomplete utilities
**File:** `core/utils/similarity.py`

- `cosine_similarity()` and `euclidean_distance()` are stubs — both just `pass`. The real cosine logic lives duplicated inside `face_database.py`.

### 3.9 Docker — broken Dockerfile
**File:** `docker/Dockerfile`

- Copies `requirements/base.txt` — this path **does not exist** in the repo. The actual file is `requirements.txt`.
- Uses `nvidia/cuda:12.1.0-runtime` but AGENT_CONTEXT says CUDA 12.6 is deployed. Version mismatch.

---

## 4. Prioritized Recommendations

### 🔴 Priority 1 — Correctness & Accuracy (Do Now)

#### 4.1 Add face alignment before recognition
The landmarks (`kps`) from SCRFD are already available on the `Face` object but **never used**.
Use `insightface.utils.face_align.norm_crop()` (already a dependency) to align the 112×112 face crop using the 5-point landmarks before passing to AdaFace.

**Expected gain:** +5–15% recognition accuracy on angled faces. This is your single biggest accuracy win.

```python
# In adaface_recognizer.py — replace raw crop with:
from insightface.utils import face_align
aligned = face_align.norm_crop(frame, landmark=face.kps)
```

#### 4.2 Implement the similarity utils (fix stubs)
Move cosine logic from `face_database.py` into `core/utils/similarity.py` and import it. Also implement the Euclidean distance for future use.

---

### 🟠 Priority 2 — Reliability & Robustness (Do Soon)

#### 4.3 Upgrade tracker: IOU → ByteTrack or BoT-SORT

| Tracker | Pros | Cons |
|---|---|---|
| **IOUTracker** (current) | Simple | No prediction, no re-ID, breaks on occlusion |
| **ByteTrack** ⭐ Recommended | Fast, handles occlusion, uses Kalman, very lightweight | Slightly more complex setup |
| **BoT-SORT** | Best accuracy, re-ID aware | Heavier |
| **DeepSORT** | Widely documented | Slower, re-ID model needed |

**Recommendation:** Use **ByteTrack** via the `supervision` library (Roboflow).
```bash
pip install supervision
```
```python
import supervision as sv
tracker = sv.ByteTracker()
```
This replaces your entire `iou_tracker.py` with 2 lines of code and gives you Kalman filtering, occlusion handling, and proper Hungarian matching.

#### 4.4 Replace WhatsApp Selenium bot → Twilio / WhatsApp Business API

The Selenium bot is the single highest-risk component. Any WhatsApp Web UI update will break it silently.

| Option | Reliability | Cost | Setup |
|---|---|---|---|
| **Selenium (current)** | ❌ Fragile | Free | Already done |
| **Twilio WhatsApp API** ⭐ Recommended | ✅ Stable | ~$0.005/msg | 30 min |
| **WhatsApp Business API (Meta)** | ✅ Official | Varies | Requires approval |
| **Telegram Bot API** | ✅ Very stable | Free | 15 min |

**Quickest fix:** Switch to **Telegram Bot API**. Takes 15 minutes, completely free, no approval needed.
```bash
pip install python-telegram-bot
```

#### 4.5 Fix pipeline threading — decouple capture from inference
Current: `capture → detect → recognize → match` all sequential.
Recommended: use a **producer/consumer** pattern with threads.

```
Thread 1: cap.read() → frame_queue
Thread 2: detect + track → face_queue
Thread 3: recognize + match → event_queue
Thread 4: emit events + snapshots
```
This alone can nearly double effective throughput.

---

### 🟡 Priority 3 — Scalability (When Moving to 20 Cameras)

#### 4.6 Enable recognition batching
Group all faces from a frame into a batch and run a single `session.run()` call.

```python
# In adaface_recognizer.py
def extract_embeddings_batch(self, face_imgs: list) -> np.ndarray:
    batch = np.stack([self._preprocess(img) for img in face_imgs])
    return self.session.run(None, {self.input_name: batch})[0]
```

**Expected gain:** 3–5× GPU throughput with 5+ simultaneous faces.

#### 4.7 Replace `.npy` gallery → FAISS vector index

```bash
pip install faiss-gpu
```

```python
import faiss
index = faiss.IndexFlatIP(512)  # Inner product = cosine on normalized vecs
index.add(stored_embeddings)
D, I = index.search(query[np.newaxis], k=1)
```

Benefits:
- Sub-millisecond search for 10,000+ identities
- Can add new people without full rebuild (`index.add()`)
- GPU-accelerated (`faiss-gpu`)

#### 4.8 Switch EventEmitter to a message broker for multi-camera

When running 20 cameras, all writing to the same `events.jsonl` will cause file lock contention.

| Option | Recommendation |
|---|---|
| **Redis Pub/Sub** ⭐ | Lightweight, no schema, easy Python integration |
| **RabbitMQ** | Better routing, more complex |
| **Kafka** | Overkill for 20 cameras |

#### 4.9 Replace flat JSONL API → proper database

Replace the "read entire file on every request" pattern in `api_server/main.py`:

| Option | Use Case |
|---|---|
| **SQLite + SQLAlchemy** ⭐ for PoC→Production | Zero-ops, persistent, queryable |
| **PostgreSQL** | When you need multi-instance API |

---

## 5. Technology Replacement Summary

| Component | Current | Recommended Upgrade | Effort | Status |
|---|---|---|---|---|
| Tracker | Custom IOU | **ByteTrack** (via `supervision`) | 1–2h | ✅ Done |
| Alert Delivery | Selenium WhatsApp | **Telegram Bot API** or Twilio | 2–4h | ❌ Pending |
| Vector Search | NumPy linear scan | **FAISS** (GPU) | 2–3h | ✅ Done |
| Pipeline threading | Single thread | Producer/consumer threads | 4–8h | ✅ Done |
| Event storage | JSONL flat file | **SQLite** + SQLAlchemy | 4–6h | ❌ Pending |
| Face alignment | Raw bbox crop | `insightface.norm_crop` (already installed) | 1h | ✅ Done |
| Buffer structure | `list.pop(0)` | `collections.deque` | 30min | ✅ Done |
| Fusion strategy | Equal mean pool | Quality-weighted mean | 1h | ✅ Done |
| Docker base image | CUDA 12.1 | CUDA 12.6 (match deployment) | 15min | ✅ Done |

---

## 6. Recommended Implementation Order

```text
Phase 1 — Accuracy (Week 1)
  ├── [x] 4.1  Face alignment with kps            ← biggest accuracy gain
  ├── [ ] 4.2  Fix similarity stubs
  ├── [x] 4.10 deque buffer fix

Phase 2 — Reliability (Week 2)
  ├── [x] 4.3  ByteTrack integration
  ├── [ ] 4.4  Telegram/Twilio alert bot
  └── [x] 4.12 Fix Dockerfile

Phase 3 — Performance (Week 3)
  ├── [x] 4.5  Threaded pipeline
  ├── [x] 4.6  Batch recognition
  ├── [x] 4.11 Quality-weighted fusion
  └── [ ] 4.13 Centralized config loader

Phase 4 — Scale (Before 20-camera rollout)
  ├── [x] 4.7  FAISS index
  ├── [ ] 4.8  Redis event bus
  └── [ ] 4.9  SQLite API backend
```

---

> **Bottom line:** The architecture and design philosophy are solid. The main gaps are (1) no face alignment despite having landmarks, (2) a fragile Selenium-based alerter, (3) a single-threaded blocking pipeline, and (4) a primitive tracker with no motion prediction. The face alignment fix alone will give you the most accuracy gain for the least effort.
