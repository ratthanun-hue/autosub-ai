#!/bin/bash
set -e

# 1. DNS Fix (ป้องกันปัญหา DNS resolution)
echo -e 'nameserver 8.8.8.8\nnameserver 1.1.1.1' > /etc/resolv.conf 2>/dev/null || true

# 2. ทำงานในโฟลเดอร์ของ whisper-server
cd /root/whisper-server

# 3. ค้นหา Python และ Uvicorn binary
PYTHON_BIN=$(which python3 || which python || echo "/opt/conda/bin/python")
UVICORN_BIN=$(which uvicorn || echo "/opt/conda/bin/uvicorn")

# 4. เริ่มต้นสคริปต์ Self-Registration ทำงานเบื้องหลัง (Background)
if [ -f "/root/whisper-server/self_register.py" ]; then
    echo "Starting Auto-Registration daemon in background..."
    nohup $PYTHON_BIN /root/whisper-server/self_register.py > /root/whisper-server/self_register.log 2>&1 &
fi

# 5. เริ่มต้นรัน FastAPI / Uvicorn Server บนพอร์ต 10100
echo "=========================================================="
echo "=== STARTING AUTOSUB-AI FASTAPI / UVICORN SERVER ==="
echo "=========================================================="
exec $UVICORN_BIN app:app --host 0.0.0.0 --port 10100 --workers 1
