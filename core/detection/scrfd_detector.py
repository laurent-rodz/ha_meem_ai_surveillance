import os
import sys
import logging

# Windows TRT pip install fix: prepend tensorrt_libs to PATH before importing onnxruntime
if sys.platform == "win32":
    trt_lib_dir = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "lib", "site-packages", "tensorrt_libs"))
    if os.path.isdir(trt_lib_dir) and trt_lib_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = trt_lib_dir + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import onnxruntime as ort
from typing import List
from .base_detector import BaseDetector
from .face import Face

log = logging.getLogger(__name__)


class SCRFDDetector(BaseDetector):
    """SCRFD Face Detector using ONNX Runtime with TensorRT backend.

    The insightface SCRFD model handles anchor decoding and NMS internally.
    We inject our own TRT-enabled ORT session to replace the default CUDA session,
    giving full TensorRT acceleration while keeping all post-processing intact.
    """

    def __init__(self, config: dict, model_path: str):
        super().__init__(config)
        from insightface.model_zoo import get_model

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

        # Load the insightface SCRFD wrapper (handles anchor decode, NMS, kps)
        self.detector = get_model(model_path)

        # Inject our TRT-enabled ORT session, replacing the default CUDA one.
        # This is done before prepare() so the anchor grid is built on the correct session.
        self.detector.session = ort.InferenceSession(model_path, providers=providers)

        self.detector.prepare(
            ctx_id=0 if self.device == "cuda" else -1,
            input_size=(640, 640),
        )

        active = self.detector.session.get_providers()
        log.info(f"[SCRFDDetector] Active providers: {active}")

    def detect(self, image: np.ndarray) -> List[Face]:
        """Runs inference using insightface and returns Face objects."""
        bboxes, kps = self.detector.detect(image)

        faces = []
        if bboxes is not None:
            for i in range(bboxes.shape[0]):
                # bbox layout: [x1, y1, x2, y2, score]
                face = Face(
                    bbox=bboxes[i],
                    kps=kps[i] if kps is not None else None,
                )
                faces.append(face)

        return faces
