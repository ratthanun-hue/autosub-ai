#!/bin/bash
set -e

# ตรวจสอบว่ามีโมเดลอยู่ในแคชหรือยัง ถ้ายังไม่มี ให้ดาวน์โหลดจาก Cloudflare R2
if [ ! -d "/root/.cache/huggingface/hub" ]; then
    echo "=========================================================="
    echo "=== 1. DOWNLOADING AI MODELS FROM CLOUDFLARE R2 ==="
    echo "=========================================================="
    mkdir -p /root/.cache
    
    # ดาวน์โหลดแบบ Multi-Connection 16 ท่อ ความเร็วเต็มสปีด
    # (เปลี่ยน URL ด้านล่างเป็น Public URL ของ R2 พี่นะครับ)
    R2_URL="${R2_MODEL_URL:-https://pub-models.yourdomain.com/autosub_models.tar.zst}"
    
    if curl --output /dev/null --silent --head --fail "$R2_URL"; then
        echo "Fetching models from: $R2_URL"
        aria2c -x 16 -s 16 -k 1M -j 16 -o /tmp/models.tar.zst "$R2_URL"
        echo "Extracting models..."
        tar -I zstd -xf /tmp/models.tar.zst -C /
        rm -f /tmp/models.tar.zst
        echo "Models loaded successfully!"
    else
        echo "[NOTICE] R2 URL not reachable or not configured yet."
        echo "Whisper will download standard models directly from HuggingFace on first run."
    fi
fi

echo "=========================================================="
echo "=== 2. STARTING AUTOSUB-AI FASTAPI / UVICORN SERVER ==="
echo "=========================================================="
exec /opt/conda/bin/uvicorn app:app --host 0.0.0.0 --port 8080 --workers 1
