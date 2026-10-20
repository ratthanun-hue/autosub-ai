#!/bin/bash
set -e

# 1. DNS Fix (ป้องกันปัญหา DNS resolution)
echo -e 'nameserver 8.8.8.8\nnameserver 1.1.1.1' > /etc/resolv.conf 2>/dev/null || true

# 2. ตรวจสอบและดาวน์โหลด AI Models จาก Cloudflare R2 ผ่าน aria2c (16 ท่อ) ถ้ายังไม่มีในเครื่อง
MODEL_URL="https://pub-84a6cf83663a474fa4b8eb84b5ae4bd5.r2.dev/autosub-models-v1.tar.zst"
if [ ! -f "/root/.cache/torch/hub/torchaudio/models/hdemucs_high_trained.pt" ]; then
    echo "=========================================================="
    echo "=== DOWNLOADING AUDIO AI MODELS FROM CLOUDFLARE R2 (16 STREAMS) ==="
    echo "=========================================================="
    mkdir -p /tmp/r2_models /root/.cache
    cd /tmp/r2_models
    
    if command -v aria2c >/dev/null 2>&1; then
        aria2c -x 16 -s 16 -k 10M -o models.tar.zst "$MODEL_URL"
    else
        curl -L -o models.tar.zst "$MODEL_URL"
    fi
    
    echo "Extracting models into /root/..."
    tar --zstd -xf models.tar.zst -C /root/
    
    rm -rf /tmp/r2_models
    echo "Audio models ready!"
else
    echo "=========================================================="
    echo "=== PRE-BAKED AUDIO MODELS DETECTED: Skipping R2 download! ==="
    echo "=========================================================="
fi

# ซิงค์ symlink ให้รองรับทั้ง /root/.cache และ /workspace/.hf_home
mkdir -p /workspace
if [ -d "/root/.cache/huggingface" ] && [ ! -e "/workspace/.hf_home" ]; then
    ln -sfn /root/.cache/huggingface /workspace/.hf_home
fi
export HF_HOME="/root/.cache/huggingface"

# 3. ทำงานในโฟลเดอร์ของ whisper-server
cd /root/whisper-server

PYTHON_BIN=$(which python3 || which python || echo "/opt/conda/bin/python")
UVICORN_BIN=$(which uvicorn || echo "/opt/conda/bin/uvicorn")

# 3.5 ป้องกันปัญหา Transformers 5.x ไม่รองรับ PyTorch < 2.6
if $PYTHON_BIN -c "import transformers; exit(0 if int(transformers.__version__.split('.')[0]) >= 5 else 1)" 2>/dev/null; then
    echo "[COMPAT] Transformers 5.x detected with PyTorch 2.5! Downgrading to transformers<5.0.0..."
    pip install --no-cache-dir "transformers<5.0.0"
fi

# 4. เริ่มต้นสคริปต์ Self-Registration สำหรับ Dynamic Worker ในเบื้องหลัง
if [ -f "/root/whisper-server/self_register.py" ]; then
    echo "Starting Dynamic Node Auto-Registration daemon in background..."
    nohup $PYTHON_BIN /root/whisper-server/self_register.py > /root/whisper-server/self_register.log 2>&1 &
fi

# 4.5 Background Watchdog for Self-Registration
(
    while true; do
        sleep 15
        if [ -f "/root/whisper-server/self_register.py" ] && ! pgrep -f "self_register.py" > /dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] self_register.py dead! Auto-restarting..." >> /root/whisper-server/watchdog.log 2>&1
            nohup $PYTHON_BIN /root/whisper-server/self_register.py >> /root/whisper-server/self_register.log 2>&1 &
        fi
    done
) &

# 5. เริ่มต้นรัน FastAPI / Uvicorn Server บนพอร์ต 10100 (Auto-Heal Enabled)
CONCURRENCY_JOBS="${MAX_JOBS_PER_NODE:-3}"
echo "=========================================================="
echo "=== STARTING AUTOSUB-AI DYNAMIC WORKER (AUTO-HEAL ENABLED) ==="
echo "=== Concurrency Limit: ${CONCURRENCY_JOBS} jobs ==="
echo "=========================================================="
while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Uvicorn app:app on port 10100..."
    $UVICORN_BIN app:app --host 0.0.0.0 --port 10100 --workers 1 || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AUTO-HEAL] Uvicorn worker exited. Auto-healing in 3 seconds..."
    sleep 3
done
