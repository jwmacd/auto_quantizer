# Use a base image with Python and CUDA support (PyTorch 2.3.0 / CUDA 12.1)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install git (needed for installing autoawq from GitHub)
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt /app/

# Set CUDA architectures for auto-gptq build (RTX 3090 is 8.6, RTX 4090 is 8.9)
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"

# Install Python dependencies from requirements.txt and AutoAWQ from GitHub
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir \
    git+https://github.com/casper-hansen/AutoAWQ.git && \
    pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Clean up git after installation (optional, reduces image size slightly)
# RUN apt-get purge -y --auto-remove git && apt-get clean

# Copy the quantization script into the container
COPY quantize.py /app/

# Set the default command to run the script with python
# Expects model path and optionally quant config via command line args or Docker CMD override
ENTRYPOINT ["python", "/app/quantize.py"]

# Remove redundant lines from previous version if they exist
# RUN apt-get update && apt-get install -y --no-install-recommends git && \
#     pip install --no-cache-dir \
#         git+https://github.com/casper-hansen/AutoAWQ.git \
#         transformers~=4.45.0 \
#         accelerate~=0.30.0 \
#         safetensors \
#         psutil \
#         datasets \
#         huggingface_hub[hf_xet] && \
#     apt-get purge -y --auto-remove git && \
#     rm -rf /var/lib/apt/lists/*
# RUN pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
