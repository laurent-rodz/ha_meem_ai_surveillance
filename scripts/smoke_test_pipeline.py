import os
import sys

# 1. Force the DLL registration BEFORE importing ONNX Runtime via PATH
if sys.platform == "win32":
    trt_lib_dir = os.path.normpath(os.path.join(os.path.dirname(sys.executable), "..", "lib", "site-packages", "tensorrt_libs"))
    print(f"TRT Lib Dir: {trt_lib_dir}")
    if os.path.isdir(trt_lib_dir) and trt_lib_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = trt_lib_dir + os.pathsep + os.environ.get("PATH", "")
        print("[TRT] Prepended to os.environ['PATH']")
    else:
        if not os.path.isdir(trt_lib_dir):
            print("[TRT] WARNING: Directory not found!")
        else:
            print("[TRT] Already in PATH")

# 2. Now import ONNX runtime
import onnxruntime as ort
import numpy as np
import yaml

def load_config(p):
    with open(p) as f:
        return yaml.safe_load(f)

# Load config
cfg = {**load_config('configs/default.yaml'), **load_config('configs/thresholds.yaml'), **load_config('configs/tensorrt.yaml')}
trt = cfg.get('tensorrt', {})
cache = trt.get('engine_cache_path', './trt_cache')
os.makedirs(cache, exist_ok=True)

opts = {
    'trt_fp16_enable': trt.get('fp16', True),
    'trt_engine_cache_enable': trt.get('engine_cache_enable', True),
    'trt_engine_cache_path': cache,
    'trt_max_workspace_size': trt.get('max_workspace_size', 1073741824),
}

providers = [
    ('TensorrtExecutionProvider', opts),
    ('CUDAExecutionProvider', {'device_id': 0}),
    'CPUExecutionProvider'
]

print("\n=== AdaFace TRT Smoke Test ===")
try:
    sess = ort.InferenceSession(cfg['models']['adaface_onnx'], providers=providers)
    active = sess.get_providers()
    print(f"Active providers: {active}")
    if 'TensorrtExecutionProvider' not in active:
        print("ERROR: TensorRT provider failed to load for AdaFace! Falling back to CUDA.")
    
    inp = sess.get_inputs()[0]
    dummy = np.random.randn(1, 3, 112, 112).astype(np.float32)
    out = sess.run(None, {inp.name: dummy})
    print(f"Output shape: {out[0].shape}")
    print("AdaFace inference OK.")
except Exception as e:
    print(f"AdaFace ERROR: {e}")

print("\n=== SCRFD TRT Smoke Test ===")
try:
    sess2 = ort.InferenceSession(cfg['models']['scrfd_onnx'], providers=providers)
    active2 = sess2.get_providers()
    print(f"Active providers: {active2}")
    if 'TensorrtExecutionProvider' not in active2:
        print("ERROR: TensorRT provider failed to load for SCRFD! Falling back to CUDA.")
    
    inp2 = sess2.get_inputs()[0]
    dummy2 = np.random.randn(1, 3, 640, 640).astype(np.float32)
    outs2 = sess2.run(None, {inp2.name: dummy2})
    print(f"Outputs: {len(outs2)} tensors, first shape={outs2[0].shape}")
    print("SCRFD inference OK.")
except Exception as e:
    print(f"SCRFD ERROR: {e}")