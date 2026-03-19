# Use the official NVIDIA L4T CUDA 12.6 image
FROM nvcr.io/nvidia/l4t-cuda:12.6.11-runtime

# 1. Install wget first, then download the NVIDIA public ARM64 keyring
RUN apt-get update && apt-get install -y wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb

# 2. NOW we can successfully install Python, audio tools, and cuDNN 9
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    libcudnn9-cuda-12 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install the safety pin for numpy
RUN pip3 install --no-cache-dir "numpy<2"

# Install the exact Jetson ONNX Runtime GPU wheel (Requires cu126)
RUN pip3 install --no-cache-dir https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

# Install Wyoming, audio processing, and tokenizer dependencies
RUN pip3 install --no-cache-dir \
    wyoming \
    soundfile \
    onnx \
    librosa \
    scipy \
    pypinyin \
    hangul-romanize \
    num2words \
    spacy \
    tokenizers

# Copy your entire folder into the container
COPY . /app

EXPOSE 10200

# Boot up the server when the container starts
CMD ["python3", "wyoming_server.py"]
