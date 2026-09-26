# ใช้ Base Image PyTorch CUDA 12.4
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# ติดตั้ง System Packages, FFmpeg, และ aria2 สำหรับดาวน์โหลดความเร็วสูง
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    aria2 \
    zstd \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้งแพ็กเกจสำหรับถอดเสียงภาษาไทยและ Dual Engine
RUN pip install --no-cache-dir \
    faster-whisper \
    transformers \
    pythainlp \
    torchaudio \
    demucs \
    fastapi \
    uvicorn \
    python-multipart \
    soundfile

# โมเดล AI ทั้ง 4 ตัว (ขนาดรวม 7.7 GB) ถูกแยกไปเก็บบน Cloudflare R2 เพื่อให้ Docker Image เบา (~3 GB) 
# และบูตผ่าน aria2c มัลติสตรีม 16 ท่อใน entrypoint.sh เสร็จใน 20-30 วิ โดยไม่เปลืองพื้นที่ดิสก์ตอน Build

# คัดลอกโค้ดสคริปต์, พจนานุกรม และสคริปต์ลงทะเบียนอัตโนมัติ
WORKDIR /root/whisper-server
COPY app.py .
COPY thai_corrections.json .
COPY thai_slang_words.json .
COPY self_register.py .
COPY entrypoint.sh /usr/local/bin/entrypoint.sh

RUN chmod +x /usr/local/bin/entrypoint.sh && \
    ln -s /usr/local/bin/entrypoint.sh /entrypoint.sh && \
    ln -s /usr/local/bin/entrypoint.sh /usr/bin/entrypoint.sh

EXPOSE 10100 22

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
