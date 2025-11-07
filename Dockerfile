FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch \
    DEMUCS_HOME=/cache/demucs \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    GRADIO_SERVER_NAME=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

EXPOSE 8686
CMD ["bash", "-lc", "GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT:-8686} python app.py"]


