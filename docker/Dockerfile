# TensorRT + CUDA 12 base image — includes TRT 10.x, cuDNN 9.x, CUDA 12.x
# Matches the TensorrtExecutionProvider requirements for onnxruntime-gpu
FROM nvcr.io/nvidia/tensorrt:24.12-py3

WORKDIR /app

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Ensure pip-installed TensorRT libraries are found by ONNX Runtime
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/tensorrt_libs

# Copy project source
COPY . .

# Pre-create TRT engine cache directory (engines are compiled on first run)
RUN mkdir -p trt_cache

CMD ["python", "apps/entry_pipeline/main.py"]