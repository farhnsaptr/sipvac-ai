FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 git wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# detectron2 pre-built CPU wheel — MUST match torch==2.1.0
RUN pip install --no-cache-dir \
    "detectron2 @ https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.1/detectron2-0.6-cp310-cp310-linux_x86_64.whl"

COPY . .
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
