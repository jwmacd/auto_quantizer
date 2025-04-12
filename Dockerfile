# Use a base image with Python and CUDA support (PyTorch 2.3.0 / CUDA 12.1)
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Install necessary packages
# - Install git to fetch from GitHub
# - Install the latest AutoAWQ directly from the main branch
# - Upgrade transformers and accelerate (torch is now from base image)
# - Install other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    pip install --no-cache-dir \
        git+https://github.com/casper-hansen/AutoAWQ.git \
        transformers~=4.45.0 \
        accelerate~=0.30.0 \
        safetensors && \
    apt-get purge -y --auto-remove git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

# Install Python dependencies
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r /app/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

# Copy the quantization script into the container
COPY quantize.py /app/

# Make the script executable (optional, as we run with python)
# RUN chmod +x /app/quantize.py

# Set the default command to run the script with python
# Expects model path and optionally quant config via command line args or Docker CMD override
ENTRYPOINT ["python", "/app/quantize.py"]
