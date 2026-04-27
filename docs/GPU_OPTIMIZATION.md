# Pipeline Optimization Tasks

This document outlines potential optimizations to shift more workloads from the CPU to the GPU, aiming to increase overall throughput and reduce latency in the surveillance pipeline.

## 1. Hardware-Accelerated Video Decoding (NVDEC)
Currently, frame extraction from video streams is likely handled by the CPU.
- **Task**: Implement hardware-accelerated video decoding using NVIDIA NVDEC.
- **Implementation**: Compile OpenCV with CUDA/GStreamer support or use a dedicated library like NVIDIA Video Processing Framework (VPF) or PyAV with hardware acceleration. This ensures frames are decoded directly into GPU memory.

## 2. GPU Pre-processing
Image resizing, color conversion (BGR to RGB), and normalization ([-1, 1] scaling) are currently performed on the CPU using OpenCV and NumPy in `adaface_recognizer.py` and potentially elsewhere.
- **Task**: Move pre-processing steps to the GPU.
- **Implementation**: 
  - Use **CuPy** or **PyTorch** tensors to perform resizing and normalization directly on the device.
  - Alternatively, integrate **NVIDIA DALI** (Data Loading Library) to handle the entire pipeline from decoding to pre-processing entirely on the GPU, avoiding costly Host-to-Device (H2D) memory transfers.

## 3. GPU-Accelerated Vector Search (FAISS)
The current `FaceDatabase` uses `faiss.IndexFlatIP` which executes on the CPU. While fast for small galleries, it can become a bottleneck as the number of stored identities grows.
- **Task**: Migrate the FAISS index to the GPU.
- **Implementation**: 
  - Ensure `faiss-gpu` is installed.
  - Modify `core/database/face_database.py` to transfer the index to the GPU using `res = faiss.StandardGpuResources()` and `faiss.index_cpu_to_gpu(res, 0, cpu_index)`.

## 4. Tensor-Based GPU Tracking
The ByteTrack implementation utilizes `supervision`, which runs Kalman filters and Hungarian matching on the CPU.
- **Task**: Evaluate and implement a GPU-accelerated tracking algorithm if tracking becomes a CPU bottleneck under dense crowds.
- **Implementation**: Implement a batch-tensor-based tracker directly in PyTorch or CuPy, allowing bounding box IOUs and cost matrices to be computed on the GPU.

## 5. End-to-End GPU Memory Pipeline
Currently, the pipeline transfers data between the CPU and GPU multiple times (e.g., CPU frame -> GPU detection -> CPU bounding boxes -> CPU crop -> GPU recognition -> CPU embedding).
- **Task**: Keep tensors on the GPU as much as possible.
- **Implementation**: Use ONNX Runtime's `io_binding` feature to bind inputs and outputs directly to pre-allocated CUDA memory. Pass memory pointers between the detector, tracker, and recognizer instead of pulling data back to NumPy arrays on the host.