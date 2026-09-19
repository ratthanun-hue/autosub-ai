# 1. ใช้ Base Image PyTorch CUDA 12.4 ที่เสถียรที่สุด
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com

# 2. ติดตั้ง System Packages, FFmpeg, และ aria2
RUN apt-get update && apt-get install -y \
    ffmpeg \
    aria2 \
    zstd \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. ติดตั้งแพ็กเกจเสริมเข้ากับ Python ใน Base Image โดยตรง (ไม่สร้าง venv ใหม่เพื่อรักษา PyTorch 2.5.1+cu124 ไว้)
RUN pip install --no-cache-dir \
    faster-whisper \
    transformers \
    pythainlp \
    demucs \
    fastapi \
    uvicorn \
    python-multipart \
    soundfile

# 4. คัดลอกโค้ดสคริปต์และพจนานุกรม
WORKDIR /root/whisper-server
COPY app.py .
COPY thai_corrections.json .
COPY thai_slang_words.json .
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# 5. ทำสคริปต์ onstart.sh สำหรับ Vast.ai ให้บูตขึ้นมาแล้วรันเซิร์ฟเวอร์ทันที
RUN echo '#!/bin/bash' > /root/onstart.sh && \
    echo 'bash /entrypoint.sh &' >> /root/onstart.sh && \
    chmod +x /root/onstart.sh

# 6. เปิดพอร์ต 8080 (ตรงกับพอร์ต Proxy ของ Vast.ai)
EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
