# Implementation Priority List
## Porting Project B Features → Project A

> **Goal:** Incrementally upgrade `ha_meem_ai_surveillance` (A) with the superior features
> from `ha_meem_ai_surveillance-main` (B), without breaking the existing TensorRT pipeline.
>
> Priorities are ordered by **impact × risk**:
> - 🔴 **P1 — Critical** (correctness / reliability bugs in current system)
> - 🟠 **P2 — High** (significant accuracy or performance gains)
> - 🟡 **P3 — Medium** (quality-of-life, observability, robustness)
> - 🟢 **P4 — Low** (nice-to-have, polish)

---

## 🔴 P1 — Critical (Do First)

### 1. Async I/O Worker
**Problem:** Project A writes snapshots and JSONL events **synchronously on the inference hot path**,
stalling every camera thread on every recognition event.

**Fix:** Port `core/io_worker.py` from Project B — a dedicated background thread with a 64-slot queue.

**Files to create/modify:**
- Create `core/io_worker.py` (copy from B)
- Modify `core/camera/worker.py` — replace inline `emit()` + `save()` with `AsyncIOWorker.submit()`

**Effort:** ~1–2 hours | **Risk:** Low

---

### 2. Encapsulate Pipeline State (`PipelineState`)
**Problem:** Project A scatters `decided_tracks` (a raw `set`) and `identity_last_seen` (a raw `dict`)
directly inside `CameraWorker`. This is fragile, untestable, and has a subtle bug: UNKNOWN tracks
are never re-attempted after the first failed match.

**Fix:** Port `core/pipeline_state.py` from Project B. It:
- Separates decided vs. cooldown state cleanly
- Exposes `is_upgradeable()` for UNKNOWN → AUTHORIZED upgrade logic
- Makes state independently unit-testable

**Files to create/modify:**
- Create `core/pipeline_state.py` (copy from B)
- Modify `core/camera/worker.py` — replace raw dicts with `PipelineState` instance

**Effort:** ~2–3 hours | **Risk:** Low-Medium

---

### 3. UNKNOWN → AUTHORIZED Track Upgrade
**Problem:** Project A marks a track as `decided` after the first consensus match — even if that
match was UNKNOWN (no identity found). There is no second chance, so a subject who turns to face
the camera on frame 10 after being seen sideways on frames 1–6 will never be identified.

**Fix:** After porting `PipelineState` (item 2 above), add the upgrade logic from
`apps/entry_pipeline/main.py` in Project B (lines 249–257):

```python
if upgradeable:
    if identity and score >= threshold + upgrade_margin:
        state.upgrade_track(track_id, identity)
        emit_event = "AUTHORIZED"
```

**Files to modify:**
- `core/camera/worker.py`
- `configs/thresholds.yaml` — add `upgrade_margin: 0.05`

**Effort:** ~1 hour (requires P1.2 first) | **Risk:** Low

---

## 🟠 P2 — High Impact

### 4. Adaptive Blur Threshold (`AdaptiveBlurThreshold`)
**Problem:** Project A uses a single fixed `blur_threshold: 100` for all cameras. A PTZ camera
in a dim corridor has very different image statistics from a 4K entrance cam — one value cannot
be optimal for both, leading to either over-rejection or noise infiltration.

**Fix:** Port `AdaptiveBlurThreshold` from `core/quality/blur.py` in Project B. It maintains a
rolling 500-frame window of blur scores per camera and sets the threshold at the 20th percentile.

**Files to modify:**
- `core/quality/blur.py` — add the `AdaptiveBlurThreshold` class
- `core/camera/worker.py` — replace fixed threshold with per-camera instance
- `configs/thresholds.yaml` — add `quality:` block

**Effort:** ~2 hours | **Risk:** Low

---

### 5. Top-K Matching + Margin Test
**Problem:** Project A's `FaceDatabase.match()` searches only `k=1` and accepts the result if it
exceeds the threshold. When two gallery identities score close together (e.g., 0.58 vs 0.56),
the top-1 pick is unreliable — a near-tie should be rejected, not blindly accepted.

**Fix:** Upgrade `FaceDatabase.match()` to retrieve `top_k=10` candidates, group by identity,
then apply a margin test: if `score_1 - score_2 < match_margin`, return UNKNOWN.

**Files to modify:**
- `core/database/face_database.py` — upgrade `match()` signature and logic
- `configs/thresholds.yaml` — add `match_margin: 0.05` and `match_top_k: 10`
- `core/camera/worker.py` — pass margin/top_k to `face_db.match()`

**Effort:** ~2 hours | **Risk:** Low

---

### 6. Recency-Weighted Aggregation + Time-Gated Decision
**Problem:** Project A's `EmbeddingAggregator` weights all frames equally and gates on frame count
only. This means 8 old blurry frames can outvote 2 recent sharp ones. On hardware with variable
FPS, a count-based gate fires at very different real-world time intervals.

**Fix:** Port Project B's v2 aggregator:
- Exponential recency decay: `decay^(n-1-i)` (newest frame has weight 1.0)
- Time-based gate: wait `min_decision_seconds` of wall-clock time before deciding

**Files to modify:**
- `core/fusion/aggregator.py` — upgrade to v2 logic
- `configs/thresholds.yaml` — add `fusion:` block (`recency_decay`, `min_decision_seconds`, `expire_after_seconds`)

**Effort:** ~3 hours | **Risk:** Medium (changes aggregation math — test on a recorded video first)

---

### 7. Self-Expiring Track Buffers
**Problem:** Project A manually prunes the aggregator by checking which track IDs are still active
in the tracker. This logic is in the main loop of `CameraWorker.run()` and is fragile — if a track
is dropped by ByteTracker between frames, the aggregator buffer leaks.

**Fix:** Already included in the v2 aggregator (item 6). After the upgrade, replace the manual
pruning loop with a single call to `aggregator.expire_stale_tracks()` and use the returned IDs
to clean up `PipelineState`.

**Files to modify:** `core/camera/worker.py` (depends on P2.6)

**Effort:** ~30 min (part of item 6) | **Risk:** Low

---

## 🟡 P3 — Medium Priority

### 8. Pose-Weighted Quality Score
**Problem:** Project A does not penalize profile or heavily-angled face crops. A 90° profile crop
contributes equally to the consensus as a frontal crop, degrading aggregation quality.

**Fix:** Port `pose_weight()` from `core/utils/image.py` in Project B and compute a composite
`quality_score = blur × confidence × pose_weight × size_factor` per face.

**Files to create/modify:**
- Create `core/utils/image.py` (or add `pose_weight` to existing utils)
- Modify `core/camera/worker.py` — compute and assign `face.quality_score`

**Effort:** ~2 hours | **Risk:** Low

---

### 9. Interactive ROI Gate per Camera
**Problem:** Project A has no way to restrict face detection to a specific zone (e.g., a doorway).
All faces in the full frame — including people in the background — are processed.

**Fix:** Port the ROI filtering logic, mouse-draw callback, and `_save_roi()` persistence from
Project B's `apps/entry_pipeline/main.py`. Switch from grid display to per-camera named windows.

**Files to modify:**
- `apps/entry_pipeline/main.py` — full rewrite of display/event loop
- `configs/cameras.yaml` — add `roi:` field per camera

**Effort:** ~3–4 hours | **Risk:** Medium (changes the display architecture)

---

### 10. Structured Logging (replace `print`)
**Problem:** Project A uses bare `print()` statements throughout. No timestamps, no severity
levels, no way to redirect to a log file, no way to suppress debug noise in production.

**Fix:** Replace all `print()` calls with `logging.getLogger(__name__)` using the same
format Project B uses: `%(asctime)s [%(levelname)s] %(name)s — %(message)s`.

**Files to modify:** All files under `core/` and `apps/`

**Effort:** ~2 hours | **Risk:** Very low

---

### 11. Upgraded API Server (In-Memory Cache + SSE)
**Problem:** Project A's API re-reads the entire JSONL file on every request. Under high event
volume this is slow and causes file-lock contention with the writer threads.

**Fix:** Port the in-memory `deque` cache, `_tail_log_file()` background thread, and
`/events/stream` SSE endpoint from Project B's `apps/api_server/main.py`.
Also add `camera_id`, `identity`, `event_type`, `since` query filters.

**Files to modify:**
- `apps/api_server/main.py` — full upgrade

**Effort:** ~3 hours | **Risk:** Low

---

### 12. FAISS Fallback (NumPy linear scan)
**Problem:** If `faiss` fails to install (e.g., on a CPU-only dev machine), Project A crashes at
import time. This blocks testing and onboarding.

**Fix:** Wrap the FAISS import in a `try/except` and implement a NumPy dot-product fallback,
exactly as Project B does.

**Files to modify:**
- `core/database/face_database.py`

**Effort:** ~30 min | **Risk:** Very low

---

## 🟢 P4 — Low Priority (Polish)

### 13. Configurable Tracker Parameters
Move `ByteTracker` construction parameters (`track_activation_threshold`, `lost_track_buffer`,
`minimum_matching_threshold`) out of the hard-coded `CameraWorker.__init__()` and into
`configs/thresholds.yaml` under a `tracking:` block.

**Effort:** ~1 hour | **Risk:** Very low

---

### 14. Per-Camera Named Display Windows
Switch from the current single composited grid window to per-camera named windows
(required anyway for ROI mouse-draw in item 9). Can be done independently as a UX improvement.

**Effort:** ~1–2 hours | **Risk:** Low

---

### 15. `core/utils/` Module
Create the `core/utils/` package from Project B containing:
- `bbox.py` — `iou_matrix`, `bbox_to_xywh`, `xywh_to_bbox`
- `image.py` — `align_face`, `pose_weight`
- `similarity.py` — cosine similarity helpers
- `config.py` — `load_config` utility

These are prerequisites for items 6, 8, and a future SORT tracker swap.

**Effort:** ~1 hour | **Risk:** Very low

---

## Suggested Sprint Order

| Sprint | Items | Goal |
|---|---|---|
| **Sprint 1** | 1, 2, 3, 12 | Fix correctness bugs; decouple I/O; stop memory leaks |
| **Sprint 2** | 4, 5, 6+7, 10 | Accuracy improvements; structured logging |
| **Sprint 3** | 8, 9, 11 | UX (ROI), observability (API), quality score |
| **Sprint 4** | 13, 14, 15 | Config polish, code hygiene, utils module |

---

## Risk Notes

> **Do NOT replace ByteTracker with SORTTracker** unless you have labelled test video.
> ByteTrack handles occlusion better in crowded scenes; SORT handles fast single-person motion
> better. Benchmark both on your actual camera footage before switching.

> **Keep TensorRT** — Project B's CUDA-only path is slower. The TRT pipeline in Project A is
> a key differentiator and must not be removed during this merge.
