# Docker Container for AutoAWQ Quantization

This project provides a Docker container to quantize Hugging Face compatible language models using the [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) library. It simplifies the process of applying Activation-aware Weight Quantization (AWQ) to models, creating a smaller and potentially faster version suitable for inference.

## Features

- Quantizes models compatible with the Hugging Face `transformers` library.
- Uses the `AutoAWQ` library for efficient quantization.
- Saves quantized models alongside original files with an `-AWQ` suffix.
- Dockerized for easy environment setup and reproducibility.
- Supports GPU acceleration via NVIDIA Container Toolkit.

## Prerequisites

- Docker installed ([Docker Engine](https://docs.docker.com/engine/install/))
- NVIDIA Container Toolkit installed if using GPU acceleration ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- A Hugging Face compatible language model directory containing model weights (e.g., `.safetensors` files) and tokenizer files.

## Building the Docker Image

Build the image using the provided `Dockerfile`. You can tag it with a name of your choice.

```bash
docker build -t <image-name> . 
```

## Running Quantization

To run the quantization, mount the directory containing your model files with **Read/Write access** into the container at `/models`. The script reads the model from `/models` and saves the quantized output files (`*-AWQ.safetensors`, `quant_config-AWQ.json`) back into the same directory on your host.

```bash
# Example using GPU acceleration
# Ensure the host path /path/to/your/models is writable
docker run --rm --gpus all \
  -v /path/to/your/models:/models:rw \
  <image-name> \
  --model_path /models \
  --quant_config '{"zero_point": true, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}'
```

**Arguments:**
  - `--gpus all`: (Optional) Enables GPU acceleration. Remove this flag to run on CPU only.
  - `-v /path/to/your/models:/models:rw`: Mounts your local model directory (replace `/path/to/your/models`) to `/models` inside the container with Read/Write permissions.
  - `<image-name>`: The name you tagged the Docker image with during the build step.
  - `--model_path /models`: Specifies the path *inside the container* where the model is located and where output AWQ files will be saved.
  - `--quant_config '...'`: A JSON string specifying the AWQ quantization parameters. Common parameters include:
    - `w_bit`: Number of bits for weight quantization (e.g., 4).
    - `q_group_size`: Group size for quantization (e.g., 128).
    - `zero_point`: Whether to use zero-point quantization (e.g., `true`).
    - `version`: AWQ implementation version (e.g., `GEMM`, `W4A16`). Refer to AutoAWQ documentation for options.

**Output:**

The script will create `*-AWQ.safetensors` and `quant_config-AWQ.json` files within the directory you mounted (e.g., `/path/to/your/models` on your host).

## Optional: Deployment via Container Registry (e.g., GHCR)

If you want to store your built image in a container registry like GitHub Container Registry (GHCR) for easier distribution or use on other machines (like an Unraid server), follow these steps:

1. **Build the Image Locally** (if not already done):

    ```bash
    docker build -t <image-name> . 
    ```

2. **Tag for the Registry:**
    Tag the image with the full registry path, including your registry username/organization and a chosen image name/tag. Replace placeholders accordingly.

    ```bash
    # Example for GHCR:
    # docker tag <local-image-name> <registry-host>/<your-ghcr-username>/<image-name>:<tag>
    docker tag <image-name> ghcr.io/<your-ghcr-username>/<image-name>:<tag>
    ```

3. **Log in to the Registry:**
    Authenticate with the container registry. For GHCR, use your GitHub username and a Personal Access Token (PAT) with appropriate `read:packages` and `write:packages` scopes.

    ```bash
    # Example for GHCR:
    docker login ghcr.io -u <your-ghcr-username>
    ```
    *(Enter your PAT when prompted for the password)*

4. **Push to the Registry:**

    ```bash
    # Example for GHCR:
    docker push ghcr.io/<your-ghcr-username>/<image-name>:<tag>
    ```

5. **Using the Image (Example: Unraid Docker Template):**

    You can now pull and use this image on other systems. For example, in Unraid's Docker tab:
    - **Repository:** `ghcr.io/<your-ghcr-username>/<image-name>:<tag>` (or your specific image path)
    - **GPU Allocation:** Pass through NVIDIA GPUs if desired.
    - **Volume Mappings:**
        - Map a host path containing your model to `/models` with **Read/Write** access.
          (e.g., Host: `/mnt/user/models/my_model/`, Container: `/models`, Mode: Read/Write)
    - **Extra Parameters / Post Arguments:** Add the `quantize.py` arguments needed for your specific model:
        ```bash
        --model_path /models --quant_config '{"w_bit": 4, "q_group_size": 128, "zero_point": true, "version": "GEMM"}'
        ```

    *Note: The Unraid steps are provided as an example; adapt configuration for your specific environment.*
