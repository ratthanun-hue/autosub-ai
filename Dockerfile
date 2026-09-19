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

# อบโมเดล AI ทั้ง 4 ตัวไว้ใน Image ล่วงหน้า (บูตเครื่องแล้วทำงานได้ทันทีภายใน 15 วิ)
RUN python -c "\
from faster_whisper import WhisperModel; \
print('Pre-baking large-v3-turbo...'); \
WhisperModel('large-v3-turbo', device='cpu', compute_type='int8'); \
from huggingface_hub import snapshot_download; \
print('Pre-baking Typhoon CTC...'); \
snapshot_download('typhoon-ai/typhoon-whisper-large-v3-ctc'); \
print('Pre-baking Typhoon Base...'); \
snapshot_download('typhoon-ai/typhoon-whisper-large-v3'); \
import torchaudio; \
print('Pre-baking HDemucs model...'); \
torchaudio.pipelines.HDEMUCS_HIGH_MUSDB.get_model(); \
print('All models baked successfully!'); \
"

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
