# Ha-Meem AI Surveillance — Feature Analysis & Suggestions

## Current State Summary

### ✅ Already Implemented (Core + Advanced)
| Feature | Status | Notes |
|---------|--------|-------|
| SCRFD Face Detection (TensorRT FP16) | ✅ | Cached TRT engine |
| AdaFace Recognition (Batch) | ✅ | With quality-weighted fusion |
| ByteTrack (via supervision) | ✅ | Kalman filter, occlusion handling |
| Pose Estimation | ✅ | Just implemented (cv2.solvePnP) |
| FAISS Vector Database | ✅ | Top-K + margin test |
| Async I/O Worker | ✅ | Off hot path |
| API Server (FastAPI) | ✅ | SSE streaming, filtering |
| In-memory Event Cache | ✅ | File tailing background thread |
| Adaptive Blur Threshold | ✅ | Per-camera rolling percentile |
| Recency-weighted Fusion | ✅ | Exponential decay |
| UNKNOWN→AUTHORIZED Upgrade | ✅ | Configurable margin |
| Report Generation | ✅ | Markdown with face crops |
| Structured Logging | ✅ | Python logging module |

### ❌ Still Pending (from Project B Merge)
1. **Interactive ROI Gate** — Click-drag zones with persistence to YAML
2. **Telegram/Twilio Alert Bot** — Replace fragile Selenium WhatsApp
3. **SQLite API Backend** — Replace JSONL + in-memory cache
4. **Centralized Config Loader** — Eliminate duplicate load_config() calls
5. **Fix Similarity Utils** — Implement stubs in `core/utils/similarity.py`

---

## Feature Suggestions by Priority

### 🔴 Priority 1 — Security & Compliance (Critical for Production)

#### 1.1 API Authentication & Authorization
**Problem:** Current API has no authentication — anyone can access events/logs.
**Solution:**
- Add JWT token authentication to API endpoints
- Role-based access (admin, viewer, guard)
- API key support for third-party integrations

```yaml
# configs/security.yaml
api:
  auth_enabled: true
  jwt_secret: "${JWT_SECRET}"
  token_expiry_hours: 24
  roles:
    - admin: [all]
    - viewer: [read]
    - guard: [events:read, cameras:read]
```

#### 1.2 HTTPS/TLS Support
**Problem:** API runs on HTTP — credentials sent in plaintext.
**Solution:**
- Add SSL certificate support to uvicorn
- Auto-generate self-signed cert for internal deployments
- Let's Encrypt integration for public-facing instances

#### 1.3 Audit Logging
**Problem:** No record of who accessed what data or changed configurations.
**Solution:**
- Log all API access with user ID, timestamp, endpoint
- Track configuration changes
- Export audit logs for compliance

#### 1.4 Privacy Compliance (GDPR/CCPA)
**Problem:** No mechanism to handle "right to be forgotten" or data portability.
**Solution:**
- Auto-blur faces for non-consented individuals
- API endpoint to delete person data (cascade delete from FAISS + events)
- Data export endpoint (GDPR Article 20)
- Configurable data retention policies

---

### 🟠 Priority 2 — Factory-Specific Features (High Value)

#### 2.1 Zone-Based Presence Tracking
**Problem:** System only identifies faces but doesn't track which work zone people are in.
**Solution:**
- Define multiple zones per camera (entry, production floor, break room)
- Track movements between zones
- Generate zone-specific attendance reports

```yaml
# configs/zones.yaml
cameras:
  camera_01:
    zones:
      - id: entry_gate
        polygon: [[100,100], [300,100], [300,400], [100,400]]
      - id: production_floor
        polygon: [[400,100], [900,100], [900,500], [400,500]]
```

#### 2.2 Shift & Attendance Management
**Problem:** Factory needs automated attendance tracking with shift support.
**Solution:**
- Configure shift timings (morning, evening, night)
- Auto-mark attendance on entry/exit
- Overtime calculation
- Integration with payroll systems (API/CSV export)

#### 2.3 Dwell Time Analytics
**Problem:** No insight into how long people spend in restricted areas.
**Solution:**
- Track time spent per zone per person
- Alert on prolonged presence in hazardous/restricted zones
- Generate dwell time reports

#### 2.4 Safety Gear Detection (Optional)
**Problem:** Factory requires helmets, vests, etc.
**Solution:**
- Add YOLO-based PPE detection
- Alert when person enters without safety gear
- Link to face ID for accountability

---

### 🟡 Priority 3 — Alerting & Integration (Reliability)

#### 3.1 Replace Selenium WhatsApp → Telegram Bot
**Problem:** Selenium bot breaks on WhatsApp Web UI changes.
**Solution:**
```bash
pip install python-telegram-bot
```
- 15-minute setup, completely free
- Stable API, no scraping

#### 3.2 Add Email Alerts
**Problem:** Only WhatsApp/Telegram available.
**Solution:**
- SMTP integration for email notifications
- Configurable alert levels (INFO, WARNING, CRITICAL)
- Daily summary emails

#### 3.3 Webhook Support
**Problem:** No way to push events to third-party systems (SIEM, ERP).
**Solution:**
- Configure webhook URLs per event type
- Retry logic with exponential backoff
- Support for custom headers (API keys)

#### 3.4 Alert Throttling
**Problem:** Same person re-triggering alerts repeatedly.
**Solution:**
- Per-identity cooldown timers
- Batch alerts into digest notifications
- Priority-based alert routing

---

### 🟢 Priority 4 — Database & Storage (Scalability)

#### 4.1 SQLite/PostgreSQL Backend
**Problem:** In-memory cache + JSONL file doesn't scale; no querying.
**Solution:**
```python
# Replace JSONL with SQLAlchemy
from sqlalchemy import create_engine, Column, String, Float, DateTime
class Event(Base):
    __tablename__ = 'events'
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)
    camera_id = Column(String)
    track_id = Column(Integer)
    identity = Column(String, nullable=True)
    score = Column(Float)
    event_type = Column(String)
```

**Benefits:**
- Persistent storage with WAL mode (fast writes)
- SQL querying for complex reports
- No file locking issues with multiple cameras

#### 4.2 Face Gallery Management UI
**Problem:** Adding new people requires manually editing `.npy` files.
**Solution:**
- Web UI to register new persons (upload photo → auto-extract embedding)
- Bulk import from CSV
- Deactivate/activate persons without deleting

#### 4.3 Automatic Backups
**Problem:** No backup mechanism for gallery/events.
**Solution:**
- Daily incremental backups of FAISS index + events DB
- Configurable retention (keep last N backups)
- Cloud backup option (S3/GCS/Azure)

---

### 🔵 Priority 5 — Monitoring & Observability

#### 5.1 Prometheus Metrics Endpoint
**Problem:** No visibility into system performance.
**Solution:**
```python
from prometheus_client import Gauge, Counter, generate_latest

faces_detected = Counter('faces_detected_total', 'Total faces detected', ['camera_id'])
recognition_latency = Gauge('recognition_latency_ms', 'Recognition latency')

@app.get("/metrics")
def metrics():
    return generate_latest()
```

**Metrics to expose:**
- FPS per camera
- Detection/recognition latency
- FAISS search latency
- Queue sizes (frame, event, IO)
- GPU memory usage

#### 5.2 Grafana Dashboard
**Problem:** No visual monitoring.
**Solution:**
- Pre-built Grafana dashboard JSON
- Panels for: FPS, latency, event rates, identity distribution
- Alert thresholds visualized

#### 5.3 Health Check Enhancements
**Problem:** `/health` only returns `{"status": "ok"}`.
**Solution:**
```json
{
  "status": "ok",
  "components": {
    "detector": {"status": "ok", "last_inference_ms": 15},
    "recognizer": {"status": "ok", "last_inference_ms": 8},
    "database": {"status": "ok", "identities": 150},
    "camera_01": {"status": "ok", "fps": 28.5, "last_frame": "2026-05-08T00:30:00"}
  }
}
```

---

### 🟣 Priority 6 — UI/UX (Optional but Valuable)

#### 6.1 Web Dashboard (React/Vue)
**Features:**
- Live camera grid with face overlays
- Real-time event feed (SSE)
- Search historical events with filters
- Person management (add/remove/edit)
- Zone configuration with drag-drop

#### 6.2 Mobile App (Optional)
- Guard/manager mobile view
- Push notifications for alerts
- Quick search by person name
- Approve/reject UNKNOWN faces

---

### ⚪ Priority 7 — Advanced Analytics

#### 7.1 Heatmap Generation
- Visualize foot traffic patterns
- Identify congestion points
- Optimize camera placement

#### 7.2 Demographic Analytics
- Age/gender estimation (optional, privacy-controlled)
- Peak hours analysis
- Visitor vs. employee patterns

#### 7.3 Anomaly Detection
- Detect unusual patterns (e.g., person loitering at night)
- Baseline behavior learning
- Alert on deviations

---

## Implementation Priority Order

### Phase 1: Security & Compliance (Week 1-2)
1. API Authentication (JWT)
2. HTTPS/TLS Support
3. Audit Logging
4. Privacy compliance tools

### Phase 2: Complete Pending Items (Week 2-3)
5. Interactive ROI Gate (from Project B)
6. Telegram Alert Bot (replace Selenium)
7. SQLite Backend (replace JSONL)
8. Centralized Config Loader

### Phase 3: Factory Features (Week 3-4)
9. Zone-Based Presence Tracking
10. Shift & Attendance Management
11. Dwell Time Analytics

### Phase 4: Monitoring (Week 4-5)
12. Prometheus Metrics
13. Grafana Dashboard
14. Enhanced Health Checks

### Phase 5: UI (Optional, Week 6+)
15. Web Dashboard
16. Mobile App (if needed)

---

## Quick Wins (Low Effort, High Value)

| Feature | Effort | Value |
|---------|--------|-------|
| Telegram Bot (replace Selenium) | 1-2h | 🔥🔥🔥 High |
| Fix Similarity Utils | 30min | 🔥 Medium |
| Centralized Config Loader | 1h | 🔥 Medium |
| Alert Throttling | 1-2h | 🔥 Medium |
| Interactive ROI Gate | 3-4h | 🔥🔥 High |

---

## Recommendations

1. **Start with Telegram Bot** — Biggest reliability win for least effort
2. **Add SQLite Backend** — Enables proper querying + eliminates file lock issues
3. **API Authentication** — Must-have before any production deployment
4. **Zone Tracking** — Core differentiator for factory use case
5. **Prometheus Metrics** — Essential for operational monitoring

---

## Next Steps

Which features would you like to implement first? I recommend:
1. **Telegram Bot** (immediate reliability fix)
2. **SQLite Backend** (scalability foundation)
3. **API Authentication** (security requirement)
