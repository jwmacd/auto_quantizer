# Use a base image with Python and CUDA support (adjust CUDA version as needed)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the quantization script
COPY quantize.py .

# Default command (can be overridden)
ENTRYPOINT ["python", "quantize.py"]
