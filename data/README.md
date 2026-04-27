# Data Management & Specification

This project utilizes an externalized data strategy to maintain repository performance and strict version control over logic while keeping large binary assets separated.

## 1. External Dataset Architecture

Datasets must reside outside the core repository and are linked via the configuration system. This allows for flexible scaling and easier sharing of model-ready assets across different environments.

### Data Path Configuration
Update the absolute paths in **`configs/default.yaml`** to point to your local storage:

| Asset Type | Default Path | Description |
| :--- | :--- | :--- |
| **Raw Frames** | `data/raw_frames/` | Unprocessed input images organized by identity folders. |
| **Aligned Faces** | `data/aligned_faces/` | Faces cropped and normalized to **80px** using SCRFD landmarks. |
| **Gallery DB** | `data/gallery_embeddings.npy` | Compiled 512-d feature vectors used for real-time matching. |

---

## 2. Directory Structure

To ensure compatibility with the `apps/dataset_tools/` scripts, your external data should follow this hierarchy:

```text
/your/external/data/path/
├── raw_frames/
│   ├── person_01/
│   │   ├── img_001.jpg
│   │   └── img_002.png
│   └── person_02/
│       └── img_001.jpg
└── aligned_faces/
    ├── person_01/
    │   ├── crop_001.jpg
    │   └── crop_002.jpg
    └── person_02/
        └── crop_001.jpg
```

---

## 3. Gallery Database Specification

The `gallery_embeddings.npy` file is the core of the recognition system. It is generated using `py -m apps.dataset_tools.build_gallery` and contains:

- **Format:** NumPy `.npy` (Serialized Python Dictionary)
- **Structure:** `{ identity_id: List[np.ndarray] }`
- **Embedding:** 512-dimensional L2-normalized float32 vectors.
- **Constraints:** Supports up to 10 representative embeddings per identity for optimal balance between accuracy and search speed.

---

## 4. Resolution & Quality Requirements

For the **Proof of Concept (PoC)**, the following gates are enforced:

*   **Minimum Width:** 80 pixels (aligned).
*   **Alignment:** 112x112 norm-crop (ArcFace/AdaFace standard).
*   **Blur Rejection:** Automatic filtering during extraction to prevent low-quality features from poisoning the gallery.