# Optimized AI Models (ONNX & TensorRT)

This directory serves as the repository for all inference-ready models used by the Ha-Meem AI Surveillance system. The architecture leverages **ONNX Runtime** with a **TensorRT** backend for maximum throughput on NVIDIA GPUs.

## 1. Model Catalog

The system utilizes two primary models for the recognition pipeline:

| Component | Backbone | Filename | Input Shape | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **Face Detection** | SCRFD-10G | `scrfd_10g_bnkps.onnx` | `640x640` | Real-time detection with 5-point landmarks. |
| **Face Recognition** | AdaFace | `adaface.onnx` | `112x112` | 512-d feature extraction optimized for factory occlusions. |

---

## 2. TensorRT Optimization & Caching

The system is configured to automatically compile ONNX models into optimized **TensorRT Engines** during the first run.

*   **Precision:** FP16 (Half-precision) is enabled by default for a 2x-3x speedup with negligible accuracy loss.
*   **Storage:** Compiled engines are stored in the `trt_cache/` directory.
*   **Performance Note:** The "Cold Start" (first run) may take 1-2 minutes while the engine is being optimized. Subsequent starts will load the cached engine in milliseconds.

---

## 3. Pre-processing & Normalization

To maintain consistent recognition accuracy, all inputs to the recognition model must follow these specifications:

- **Color Space:** RGB
- **Range:** `[-1, 1]` (Normalized: `(pixel - 127.5) / 128.0`)
- **Alignment:** Similarity transformation based on 5 landmarks (Nose, Eyes, Mouth corners) to 112x112 pixels.

---

## 4. Security & Weight Management

> [!IMPORTANT]
> Model weight files (`.onnx`, `.engine`) are excluded from Git via `.gitignore`. 
> Authorized developers must download the weights from the private Project Storage and place them manually in this folder before running the pipeline.