# Project Context

## Project Overview

This project provides a Docker container to quantize machine learning models stored in `safetensors` format into the AutoAWQ (AWQ) format. It aims to provide a reproducible environment for model quantization.

## Architecture

The core component is a Docker container built using a `Dockerfile`. Inside the container, a Python script (`quantize.py`) handles the quantization process. The script utilizes libraries like `autoawq`, `torch`, and `safetensors`. Users mount a directory containing the input models into the container and specify an output directory.

## Key Components

- `Dockerfile`: Defines the Docker image build process, including dependencies and environment setup.
- `quantize.py`: The Python script that performs the AWQ quantization.
- `requirements.txt`: Lists Python dependencies.
- `README.md`: Provides instructions on building and running the Docker container.

## Conventions

- Use standard Python coding conventions (PEP 8).
- Dockerfile follows best practices for image size and build speed.

## Dependencies

- Python 3.x
- Docker
- PyTorch (with CUDA support if GPU is used)
- AutoAWQ (`autoawq`)
- Transformers (`transformers`)
- Safetensors (`safetensors`)
- Accelerate (`accelerate`)
