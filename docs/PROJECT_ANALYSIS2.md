# Architectural Analysis: Ha-Meem AI Surveillance

## Project Overview
The Ha-Meem AI Surveillance project is an advanced, AI-powered multi-camera surveillance system designed to perform real-time face detection, tracking, and identity recognition. It serves as a comprehensive pipeline capable of processing live video feeds, extracting facial embeddings, maintaining a real-time event log, and exposing these events through a modern web API, with support for automated alerts via a WhatsApp bot.

## Feature Inventory
*   **Multi-Camera Video Processing:** Streams and processes multiple video feeds concurrently using multi-threading.
    *   *Entry Point:* `apps.entry_pipeline.main` & `core.camera.worker`
*   **Real-time Face Detection:** Utilizes the SCRFD model to accurately detect faces and extract keypoints.
    *   *Entry Point:* `core.detection.scrfd_detector`
*   **Face Tracking & Aggregation:** Uses ByteTracker for robust multi-object tracking and aggregates face embeddings over time to improve recognition accuracy.
    *   *Entry Point:* `core.tracking` & `core.fusion`
*   **Face Recognition & Alignment:** Aligns faces using keypoints and extracts 512-dimensional feature embeddings using the AdaFace model.
    *   *Entry Point:* `core.recognition.adaface_recognizer`
*   **High-Speed Identity Matching:** In-memory vector database matching using FAISS for cosine similarity searches against a gallery of known identities.
    *   *Entry Point:* `core.database.face_database`
*   **Event Logging & Snapshot Generation:** Automatically captures bounding boxes, generates snapshots, and logs access events to JSONL.
    *   *Entry Point:* `core.events`
*   **Web API & Integration:** A RESTful API server that exposes the latest events, statuses, and health checks.
    *   *Entry Point:* `apps.api_server.main`
*   **Alert Bot:** An automated WhatsApp alert system utilizing Selenium to notify users of critical events.
    *   *Entry Point:* `apps.alert_bot.whatsapp_bot`

## Architecture & Design
The system employs a highly modular and heavily Object-Oriented Programming (OOP) architecture:
*   **Abstract Base Classes:** Components like `BaseDetector` and `BaseRecognizer` establish clear contracts, making it trivial to swap out underlying AI models (e.g., swapping SCRFD with RetinaFace) without altering the core pipeline.
*   **Decoupled Multi-threading:** The `CameraWorker` class encapsulates the entire inference pipeline for a single camera on an isolated thread, emitting frames to a shared `queue.Queue` to prevent blocking the main display thread.
*   **Event-Driven Communication:** Detected events (authorized/unknown identities) are broadcasted via an `EventEmitter`, decoupling the inference pipeline from logging and downstream consumers (like the FastAPI server).
*   **Configuration Overlays:** Configuration is cleanly separated using a robust YAML + `.env` merge strategy, allowing runtime injection of sensitive variables (like `CAMERA_URL` or API ports).

## Technical Stack
*   **Language Requirements:** Python 3.9+ (implied by heavy typing usage and FastAPI).
*   **Vision & AI Inference:** `opencv-python` (video handling), `insightface` (alignment/models), `onnxruntime-gpu` (model execution).
*   **Hardware Acceleration:** Configured for `CUDAExecutionProvider` and `TensorrtExecutionProvider` (`tensorrt-cu12`). Caches TRT engines to optimize startup times after initial compilation.
*   **Similarity Search:** `faiss-cpu` for efficient inner-product (cosine) similarity indexing.
*   **Web Services & Automation:** `fastapi`, `uvicorn`, `pydantic` for the API; `selenium`, `pyautogui` for the WhatsApp bot.
*   **Data & Numerics:** `numpy`, `scikit-learn`.

## Performance & Scalability
*   **Concurrency:** The system scales linearly with the number of cameras up to the hardware limit, thanks to its thread-per-camera architecture (`CameraWorker`). 
*   **Bottlenecks & Mitigations:** 
    *   To mitigate Python's Global Interpreter Lock (GIL), inference is offloaded to C/C++ backends (ONNX/TensorRT and FAISS), which routinely release the GIL.
    *   The pipeline implements operational constraints like a `blur_threshold` and `min_face_size` to reject poor-quality frames early, saving expensive model computation.
    *   Employs batch processing (`extract_embeddings_batch`) to maximize GPU utilization during feature extraction.
*   **Queue Management:** `queue.Queue(maxsize=2)` prevents memory leaks and backpressure if the main rendering thread falls behind the camera processing speed.

## Comparison Meta-Data
*   **Code Quality:** Exceptionally high. Uses clean abstractions, extensive type-hinting, docstrings, and robust error handling (e.g., auto-reconnects for video streams).
*   **Documentation Coverage:** Good. Abstractions and core logic are well-commented, though external API docs rely primarily on FastAPI's auto-generated Swagger UI.
*   **Integration Ease:** High. The decoupled nature of the event system (`events.jsonl` + REST API) allows any third-party system (dashboards, mobile apps) to easily hook into the surveillance data without touching Python code.

## Feature Parity Checklist

| Feature | Implemented | Implementation Detail / Notes |
| :--- | :---: | :--- |
| **Real-time Face Detection** | ✅ | `insightface` via ONNX/TensorRT (SCRFD) |
| **Face Tracking** | ✅ | ByteTracker with track buffer |
| **Identity Recognition** | ✅ | AdaFace via ONNX/TensorRT |
| **Multi-Camera Support** | ✅ | Threaded `CameraWorker` with queueing |
| **High-Speed Matching DB** | ✅ | FAISS (Inner Product / Cosine Similarity) |
| **Hardware Acceleration** | ✅ | CUDA & TensorRT explicitly supported |
| **REST API Access** | ✅ | FastAPI with JSONL log tailing |
| **Third-Party Alerting** | ✅ | WhatsApp bot via Selenium |
| **Blur/Quality Filtering** | ✅ | Laplacian variance blur scoring |
| **Web Dashboard (UI)** | ❌ | Requires external front-end to consume the FastAPI |
