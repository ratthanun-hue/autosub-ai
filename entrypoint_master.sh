#!/bin/bash
set -e

# 1. DNS Fix (ป้องกันปัญหา DNS resolution)
echo -e 'nameserver 8.8.8.8\nnameserver 1.1.1.1' > /etc/resolv.conf 2>/dev/null || true

# 2. ตั้งค่า Ollama Runtime เพื่อประสิทธิภาพสูงสุดบน 1x GPU 16GB
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_NUM_PARALLEL="${MAX_PARALLEL_BATCHES:-4}"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE="-1" # ให้โมเดลอยู่ใน VRAM ตลอดเวลา ไม่ unload

echo "=========================================================="
echo "=== STARTING CENTRAL MASTER LLM NODE (Qwen2.5-14B) ==="
echo "=== Parallel Batch Slots: ${OLLAMA_NUM_PARALLEL} ==="
echo "=========================================================="

# 3. รัน Ollama Service ในเบื้องหลัง
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama daemon..."
    ollama serve > /root/whisper-server/ollama.log 2>&1 &
fi

# 4. รอให้ Ollama พร้อมทำงาน
echo "Waiting for Ollama to respond on 127.0.0.1:11434..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    sleep 2
done

# 5. ตรวจสอบและดึงโมเดล Qwen2.5-14B
MODEL_NAME="${OLLAMA_MODEL:-qwen2.5:14b}"
if ! ollama list | grep -q "$MODEL_NAME"; then
    echo "Pulling $MODEL_NAME into Ollama (this will take 1-3 minutes on first boot)..."
    ollama pull "$MODEL_NAME"
fi
echo "Model $MODEL_NAME is verified and ready in Ollama!"

# 6. รันสคริปต์ Self-Registration & Heartbeat ในเบื้องหลัง
cd /root/whisper-server
PYTHON_BIN=$(which python3 || which python || echo "/usr/bin/python3")

if [ -f "/root/whisper-server/self_register_master.py" ]; then
    echo "Starting Master Node Auto-Registration daemon..."
    nohup $PYTHON_BIN /root/whisper-server/self_register_master.py > /root/whisper-server/self_register_master.log 2>&1 &
fi

# 6.5 รัน Port Bridge (10100 -> 10200) เพื่อให้ปุ่มเปิดของ Vast.ai console ใช้งานได้
if [ -f "/root/whisper-server/port_bridge.py" ]; then
    echo "Starting Port Bridge on 10100 -> 10200..."
    nohup $PYTHON_BIN /root/whisper-server/port_bridge.py > /root/whisper-server/port_bridge.log 2>&1 &
fi

# 7. เริ่มต้น FastAPI / Uvicorn Server บนพอร์ต 10200
UVICORN_BIN=$(which uvicorn || echo "$PYTHON_BIN -m uvicorn")
echo "=========================================================="
echo "=== STARTING FASTAPI MASTER LLM SERVICE ON PORT 10200 ==="
echo "=========================================================="
exec $UVICORN_BIN llm_server:app --host 0.0.0.0 --port 10200 --workers 1
