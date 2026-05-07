# NVIDIA DeepStream Integration Plan

## Context
The ha_meem_ai_surveillance project currently uses OpenCV-based video capture with Python threading for multi-camera processing. NVIDIA DeepStream SDK offers hardware-accelerated pipeline capabilities with GStreamer-based streaming analytics, TensorRT optimization, and built-in multi-stream management.

**Critical Constraint:** DeepStream SDK does NOT natively support Windows. It requires Ubuntu/Linux environment. The project must run in WSL2 (Windows Subsystem for Linux) or migrate to Linux entirely.

---

## Prerequisites Checklist

### 1. System Requirements
- **Windows 10 version 2004+ (Build 19041+) or Windows 11**
- **NVIDIA GPU** with updated drivers installed on Windows host
- **WSL2 enabled** with Ubuntu 22.04 or 24.04
- **NVIDIA Container Toolkit** for Docker GPU access in WSL2

### 2. Software to Install (Windows Side)
- NVIDIA Driver (latest Game Ready or Studio driver with WSL support)
- WSL2 with Ubuntu (via `wsl --install Ubuntu-22.04`)

### 3. Software to Install (WSL2/Ubuntu Side)
- Docker Engine
- NVIDIA Container Toolkit
- DeepStream SDK (via Docker container recommended: `nvcr.io/nvidia/deepstream:7.0-triton-multiarch`)

---

## Integration Approach: Hybrid DeepStream Pipeline

Given the existing codebase, the recommended approach is a **hybrid integration** where:
- **DeepStream handles**: Video ingestion, batched inference (detection), and multi-stream management
- **Existing Python code handles**: Face recognition (AdaFace), tracking (ByteTracker), fusion, and event emission

This minimizes rewrite while leveraging DeepStream's strengths.

---

## Implementation Steps

### Phase 1: Environment Setup (WSL2 + DeepStream)

1. **Install WSL2 with Ubuntu 22.04**
   ```powershell
   # Windows PowerShell (Admin)
   wsl --install Ubuntu-22.04
   wsl --set-default-version 2
   ```

2. **Install Docker in WSL2 Ubuntu**
   ```bash
   # Inside WSL2 Ubuntu
   sudo apt-get update
   sudo apt-get install -y apt-transport-https ca-certificates curl gnupg-agent software-properties-common
   curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
   sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
   sudo apt-get update
   sudo apt-get install -y docker-ce docker-ce-cli containerd.io
   ```

3. **Install NVIDIA Container Toolkit**
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. **Pull and Run DeepStream Container**
   ```bash
   sudo docker pull nvcr.io/nvidia/deepstream:7.0-triton-multiarch
   # Test
   sudo docker run --rm --gpus all nvcr.io/nvidia/deepstream:7.0-triton-multiarch deepstream-app --version
   ```

---

### Phase 2: Project Structure for DeepStream Integration

Create new directory structure to support hybrid pipeline:

```
apps/
  deepstream_pipeline/
    main.py              # DeepStream-based entry point
    deepstream_app_config.txt  # DeepStream SDK config
    deepstream_model_config.txt # PGIE/SGIE model configs
    nvinfer_configs/      # Individual model configs for SCRFD, AdaFace
    custom_probes.py      # Python callbacks for DeepStream pipeline
    face_processor.py     # Bridge between DeepStream and existing code
```

---

### Phase 3: DeepStream Pipeline Configuration

1. **Create DeepStream Application Config** (`deepstream_app_config.txt`):
   - Define input sources (RTSP/USB/File)
   - Configure stream muxer (`nvstreammux`)
   - Set up primary inference engine (PGIE) for face detection (SCRFD)
   - Configure secondary inference (SGIE) if needed
   - Define output sinks (display/filesink/message broker)

2. **Create Model Configs for NvInfer**:
   - Convert SCRFD ONNX to TensorRT engine compatible with DeepStream
   - Create `config_infer_primary_scrfd.txt` with parsing functions
   - Handle custom object detection parsing in Python

---

### Phase 4: Python Bindings Integration

1. **Install DeepStream Python Bindings** in WSL2:
   ```bash
   # Inside DeepStream container or native Ubuntu with DeepStream SDK
   git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git
   cd deepstream_python_apps/bindings
   # Follow build instructions for Python 3.10+
   ```

2. **Create Custom Probe Function** (`custom_probes.py`):
   - Extract frame data and bounding boxes from DeepStream metadata
   - Pass detections to existing `CameraWorker`-like processing
   - Bridge DeepStream's GStreamer buffers to OpenCV numpy arrays

3. **Implement Face Processor** (`face_processor.py`):
   - Reuse existing `AdaFaceRecognizer` for embedding extraction
   - Reuse `ByteTracker` for tracking (or use DeepStream's built-in tracker)
   - Reuse `EmbeddingAggregator` for fusion
   - Reuse `FaceDatabase` for matching
   - Emit events via existing `EventEmitter` and `AsyncIOWorker`

---

### Phase 5: Modify Existing Code for Compatibility

1. **Refactor `CameraWorker`** to support both modes:
   - Add `use_deepstream: bool` flag in config
   - Create `DeepStreamWorker` class that inherits or wraps CameraWorker logic
   - Separate frame capture from processing logic

2. **Update `core/detection/scrfd_detector.py`**:
   - Add TensorRT engine creation compatible with DeepStream's NvInfer
   - Or keep as secondary processing after DeepStream detection

3. **Create Adapter Classes**:
   - `DeepStreamFrameAdapter`: Convert GStreamer buffers to numpy arrays
   - `DeepStreamMetadataParser`: Extract detections from DeepStream metadata

---

### Phase 6: Configuration Updates

Update `configs/default.yaml` and `configs/thresholds.yaml`:

```yaml
pipeline:
  mode: "deepstream"  # or "opencv" for legacy mode
  deepstream:
    config_file: "apps/deepstream_pipeline/deepstream_app_config.txt"
    batch_size: 4
    gpu_id: 0
    nvstreammux:
      width: 1920
      height: 1080
      batch_size: 4
      batched_push_timeout: 40000
```

Update `configs/cameras.yaml` to work with DeepStream source format.

---

### Phase 7: Testing and Validation

1. **Test DeepStream Pipeline Standalone**:
   - Run sample DeepStream apps with test video
   - Verify SCRFD detection works in DeepStream
   - Validate frame extraction and metadata parsing

2. **Test Hybrid Pipeline**:
   - Process DeepStream detections through existing recognition pipeline
   - Verify tracking, fusion, and event emission work correctly
   - Compare performance vs OpenCV-based pipeline

3. **Multi-Camera Test**:
   - Test with 2+ camera streams
   - Monitor GPU utilization and FPS
   - Validate event logging and snapshots

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| DeepStream doesn't support Windows natively | High | Use WSL2; document WSL2 setup clearly |
| SCRFD model conversion to DeepStream NvInfer | Medium | Use ONNX→TensorRT conversion; may need custom parsing |
| Performance overhead in Python callbacks | Medium | Minimize data copying; use shared memory where possible |
| DeepStream Python bindings complexity | Medium | Start from existing samples; use official NVIDIA examples |
| WSL2 GPU access issues | High | Follow NVIDIA WSL2 FAQ; ensure driver compatibility |

---

## Success Criteria

1. ✅ DeepStream pipeline ingests video streams (RTSP/File)
2. ✅ SCRFD face detection runs via DeepStream's NvInfer
3. ✅ Detected faces are processed by existing AdaFace recognizer
4. ✅ Tracking, fusion, and event emission work as before
5. ✅ Multi-camera support with improved performance
6. ✅ Documentation updated with WSL2/DeepStream setup instructions

---

## Timeline Estimate

- **Phase 1 (Environment)**: 2-4 hours
- **Phase 2-3 (Config)**: 3-5 hours
- **Phase 4 (Python Integration)**: 6-10 hours
- **Phase 5 (Code Refactoring)**: 4-6 hours
- **Phase 6-7 (Testing)**: 3-5 hours

**Total**: ~18-30 hours depending on familiarity with DeepStream

---

## Next Steps

1. Confirm willingness to use WSL2 (required for Windows users)
2. Choose between DeepStream Docker container vs native Ubuntu installation
3. Prioritize which features to port first (detection only? full pipeline?)
4. Start with Phase 1 and validate WSL2 + DeepStream setup
