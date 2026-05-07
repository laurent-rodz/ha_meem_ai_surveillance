import os
import sys
import logging

# Windows TRT pip install fix: prepend tensorrt_libs to PATH before importing onnxruntime
if sys.platform == "win32":
    trt_lib_dir = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "lib", "site-packages", "tensorrt_libs"))
    if os.path.isdir(trt_lib_dir) and trt_lib_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = trt_lib_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import cv2
import onnxruntime as ort
from .base_recognizer import BaseRecognizer

log = logging.getLogger(__name__)


class AdaFaceRecognizer(BaseRecognizer):
    """AdaFace face recognition using ONNX Runtime with TensorRT backend.

    On the first run, TRT compiles a FP16 engine and caches it to disk.
    Subsequent runs load the cached engine instantly (~10ms vs ~120s cold compile).
    """

    def __init__(self, config: dict, model_path: str):
        super().__init__(config)

        trt_cfg = config.get("tensorrt", {})

        # Ensure TRT engine cache directory exists before first compile
        cache_path = trt_cfg.get("engine_cache_path", "./trt_cache")
        os.makedirs(cache_path, exist_ok=True)

        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            providers = []
            if trt_cfg.get("enabled", True):
                trt_provider_options = {
                    "trt_fp16_enable": trt_cfg.get("fp16", True),
                    "trt_engine_cache_enable": trt_cfg.get("engine_cache_enable", True),
                    "trt_engine_cache_path": cache_path,
                    "trt_max_workspace_size": trt_cfg.get("max_workspace_size", 1073741824),
                    "trt_dla_enable": trt_cfg.get("dla_enable", False),
                }
                providers.append(("TensorrtExecutionProvider", trt_provider_options))
            
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
            providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # 0:Verbose, 1:Info, 2:Warning, 3:Error, 4:Fatal

        self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (112, 112)  # Standard AdaFace / ArcFace input size

        active = self.session.get_providers()
        log.info(f"[AdaFaceRecognizer] Active providers: {active}")

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Resize, convert color space, normalize to [-1, 1], and add batch dim."""
        img = cv2.resize(face_img, self.input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # HWC → CHW
        img = img.transpose(2, 0, 1).astype(np.float32)
        # Normalize pixel range [0, 255] → [-1, 1]
        img = (img - 127.5) / 128.0
        return np.expand_dims(img, axis=0)

    def extract_embedding(self, face_img: np.ndarray) -> tuple[np.ndarray, float]:
        """Extract and L2-normalize a 512-d face embedding, returning both (embedding, norm)."""
        input_data = self._preprocess(face_img)
        outputs = self.session.run(None, {self.input_name: input_data})
        embedding = outputs[0][0]

        norm = float(np.linalg.norm(embedding))
        if norm > 0:
            embedding = embedding / norm
        return embedding, norm

    def extract_embeddings_batch(self, face_imgs: list) -> tuple[np.ndarray, np.ndarray]:
        """Extract and L2-normalize 512-d face embeddings for a batch. Returns (embeddings, norms)."""
        if not face_imgs:
            return np.array([]), np.array([])
            
        # _preprocess adds a batch dim, so we concatenate them along axis 0
        batch = np.concatenate([self._preprocess(img) for img in face_imgs], axis=0)
        outputs = self.session.run(None, {self.input_name: batch})
        embeddings = outputs[0]

        # Calculate raw feature norms before normalization
        norms = np.linalg.norm(embeddings, axis=1)
        
        # L2-normalize each embedding in the batch
        norms_expanded = norms[:, np.newaxis]
        normalized_embeddings = embeddings / np.where(norms_expanded == 0, 1e-10, norms_expanded)
        
        return normalized_embeddings, norms
