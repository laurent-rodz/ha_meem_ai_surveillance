# Ha-Meem AI Surveillance

A professional-grade AI surveillance system for real-time inference and data management.

## Structure

- `apps/`: High-level applications (inference pipeline, dataset tools).
- `core/`: Core AI modules (detection, recognition, tracking, fusion, quality).
- `models/`: Model storage and exported ONNX/TensorRT engines.
- `configs/`: Configuration for models, cameras, and thresholds.
- `experiments/`: Research and experiment scripts.
- `tests/`: Testing suite.
- `docker/`: Deployment configurations.
- `requirements.txt`: Python dependencies.

## Prerequisites

### 1. External Software
- **Python 3.10.11**: [Download here](https://www.python.org/downloads/release/python-31011/)
- **Visual Studio Build Tools**: [Download here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
- **CUDA Toolkit 12.6**: [Download Archive](https://developer.nvidia.com/cuda-12-6-0-download-archive).
- **cuDNN 9.10.0**: [Direct Download](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.10.0.56_cuda12-archive.zip).

### 2. Models & Data
- **Models & Dataset**: Private Google Drive (Request access from project owner).
  - Place model weights in `models/`.
  - Place datasets in `data/`.

## Setup

1. **First installation with**:
   ```bash
   git clone https://github.com/laurent-rodz/ha_meem_ai_surveillance.git
   cd ha_meem_ai_surveillance
   py -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration (Mandatory)**:
   Update the absolute data paths in the following files to match your local setup:
   - **`configs/cameras.yaml`**: Update `url` for RTSP streams or local video files.
   - **`configs/default.yaml`**: Update the `models` paths to point to your `.onnx` files, and the `dataset` paths for raw frames and embeddings.

## Running the System

Follow these steps in order to set up your face recognition database and start the surveillance:

1. **Extract Faces**: Detects and crops faces from your raw input images to create a training/gallery dataset.
   ```bash
   py -m apps.dataset_tools.extract_faces
   ```

2. **Build Gallery**: Generates 512-d embeddings for the extracted faces and saves them into the `gallery_embeddings.npy` database.
   ```bash
   py -m apps.dataset_tools.build_gallery
   ```

3. **Inference Pipeline**: Starts the real-time AI surveillance system (Detector -> Tracker -> Recognizer).
   ```bash
   py -m apps.entry_pipeline.main
   ```

4. **Multi-Camera Pipeline**: Starts the surveillance system across all enabled cameras defined in `configs/cameras.yaml` in a grid view.
   ```bash
   py -m apps.multi_pipeline.main
   ```

5. **Other Components**:

   - **API Server**: `py -m apps.api_server.main`
   - **WhatsApp Bot**: `py -m apps.alert_bot.whatsapp_bot`