# HuggingFace Spaces — FastAPI (Docker SDK)
# This Dockerfile is used when you set SDK: docker in README.md

FROM python:3.10-slim

# System dependencies for OpenCV + Tesseract + detectron2
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install detectron2 for CPU (adjust URL for GPU Spaces)
RUN pip install --no-cache-dir \
    "detectron2 @ https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.3/detectron2-0.6-cp310-cp310-linux_x86_64.whl"

# Copy source code
COPY . .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
