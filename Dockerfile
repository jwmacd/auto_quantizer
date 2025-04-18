# Use a base image with Python and CUDA support (PyTorch 2.3.0 / CUDA 12.1)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# -----------------------------------------------------------------------------
# Install build tools and git (needed for Triton JIT + AutoAWQ installation)
# -----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Workdir & requirements
# -----------------------------------------------------------------------------
WORKDIR /app

# Copy requirements early to leverage Docker layer caching
COPY requirements.txt /app/

# -----------------------------------------------------------------------------
# Python packages – AWQ only
# -----------------------------------------------------------------------------
# 1) AutoAWQ from GitHub main branch (latest fixes)
RUN pip install --no-cache-dir --upgrade \
        git+https://github.com/casper-hansen/AutoAWQ.git@main && \
    pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Ensure unbuffered output for logging
ENV PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Copy application code
# -----------------------------------------------------------------------------
COPY awq_quantize.py /app/

# -----------------------------------------------------------------------------
# Default command – run quantiser automatically
# -----------------------------------------------------------------------------
# Users only need to mount their model dir to /models (rw)
# Optional extra args can be appended via Docker run command or Unraid template
# -----------------------------------------------------------------------------
ENTRYPOINT ["python", "/app/awq_quantize.py", "--model_path", "/models"]
