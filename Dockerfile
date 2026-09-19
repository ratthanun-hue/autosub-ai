# ใช้ Base Image PyTorch CUDA 12.4
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ติดตั้ง System Packages, FFmpeg, และ aria2 สำหรับดาวน์โหลดความเร็วสูง
RUN apt-get update && apt-get install -y \
    ffmpeg \
    aria2 \
    zstd \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# สร้าง Virtual Environment
RUN python -m venv /venv/main
ENV PATH="/venv/main/bin:$PATH"

# ติดตั้งแพ็กเกจสำหรับถอดเสียงภาษาไทยและ Dual Engine
RUN pip install --no-cache-dir \
    faster-whisper \
    transformers \
    pythainlp \
    demucs \
    fastapi \
    uvicorn \
    python-multipart \
    soundfile

# คัดลอกโค้ดสคริปต์และพจนานุกรม
WORKDIR /root/whisper-server
COPY app.py .
COPY thai_corrections.json .
COPY thai_slang_words.json .
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

EXPOSE 10100

ENTRYPOINT ["/entrypoint.sh"]
