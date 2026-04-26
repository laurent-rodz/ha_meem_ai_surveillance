import os
import sys

# Windows TRT pip install fix: prepend tensorrt_libs to PATH before importing onnxruntime
if sys.platform == "win32":
    trt_lib_dir = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "lib", "site-packages", "tensorrt_libs"))
    if os.path.isdir(trt_lib_dir) and trt_lib_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = trt_lib_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import cv2
import onnxruntime as ort
from .base_recognizer import BaseRecognizer


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
            trt_provider_options = {
                "trt_fp16_enable": trt_cfg.get("fp16", True),
                "trt_engine_cache_enable": trt_cfg.get("engine_cache_enable", True),
                "trt_engine_cache_path": cache_path,
                "trt_max_workspace_size": trt_cfg.get("max_workspace_size", 1073741824),
                "trt_dla_enable": trt_cfg.get("dla_enable", False),
            }
            providers = [
                ("TensorrtExecutionProvider", trt_provider_options),
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = (112, 112)  # Standard AdaFace / ArcFace input size

        active = self.session.get_providers()
        print(f"[AdaFaceRecognizer] Active providers: {active}")

    def _preprocess(self, face_img: np.ndarray) -> np.ndarray:
        """Resize, convert color space, normalize to [-1, 1], and add batch dim."""
        img = cv2.resize(face_img, self.input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # HWC → CHW
        img = img.transpose(2, 0, 1).astype(np.float32)
        # Normalize pixel range [0, 255] → [-1, 1]
        img = (img - 127.5) / 128.0
        return np.expand_dims(img, axis=0)

    def extract_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """Extract and L2-normalize a 512-d face embedding."""
        input_data = self._preprocess(face_img)
        outputs = self.session.run(None, {self.input_name: input_data})
        embedding = outputs[0][0]

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
