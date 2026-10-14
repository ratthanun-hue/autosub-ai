#!/bin/bash
set -e

# 1. DNS Fix (ป้องกันปัญหา DNS resolution)
echo -e 'nameserver 8.8.8.8\nnameserver 1.1.1.1' > /etc/resolv.conf 2>/dev/null || true

# 2. ตั้งค่า Ollama Runtime เพื่อประสิทธิภาพสูงสุด (ปรับตามบทบาท Node)
ROLE="${ROLE:-cascade}"
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_NUM_PARALLEL="${MAX_PARALLEL_BATCHES:-2}"
export OLLAMA_KEEP_ALIVE="-1" # ให้โมเดลอยู่ใน VRAM ตลอดเวลา ไม่ unload
export BATCH_SIZE="${BATCH_SIZE:-25}"

STAGE1_M="${STAGE1_MODEL:-scb10x/typhoon2.5-qwen3-30b-a3b:latest}"
STAGE2_M="${STAGE2_MODEL:-none}"

MODELS_TO_PULL=()
if [ "$STAGE2_M" = "none" ] || [ -z "$STAGE2_M" ] || [ "$ROLE" = "single_stage" ]; then
    echo "=========================================================="
    echo "=== STARTING CENTRAL MASTER LLM NODE (Single-Stage Step 6) ==="
    echo "=== Model: ${STAGE1_M} (Step 7 Decommissioned) ==="
    echo "=== Parallel Batch Slots: ${OLLAMA_NUM_PARALLEL} | Batch Size: ${BATCH_SIZE} ==="
    echo "=========================================================="
    export OLLAMA_MAX_LOADED_MODELS=1
    MODELS_TO_PULL=("$STAGE1_M")
elif [ "$ROLE" = "step6" ] || [ "$ROLE" = "stage1" ]; then
    echo "=========================================================="
    echo "=== STARTING DEDICATED STEP 6 NODE (Typhoon 2.5 30B-A3B) ==="
    echo "=== Model: ${STAGE1_M} ==="
    echo "=========================================================="
    export OLLAMA_MAX_LOADED_MODELS=1
    MODELS_TO_PULL=("$STAGE1_M")
elif [ "$ROLE" = "step7" ] || [ "$ROLE" = "stage2" ]; then
    echo "=========================================================="
    echo "=== STARTING DEDICATED STEP 7 NODE (Qwen 2.5 14B Q5_K_M) ==="
    echo "=== Model: ${STAGE2_M} ==="
    echo "=========================================================="
    export OLLAMA_MAX_LOADED_MODELS=1
    MODELS_TO_PULL=("$STAGE2_M")
else
    echo "=========================================================="
    echo "=== STARTING CENTRAL MASTER LLM NODE (2-Stage Cascade) ==="
    echo "=== Models: ${STAGE1_M} + ${STAGE2_M} ==="
    echo "=== Parallel Batch Slots: ${OLLAMA_NUM_PARALLEL} | Batch Size: ${BATCH_SIZE} ==="
    echo "=========================================================="
    export OLLAMA_MAX_LOADED_MODELS=2
    MODELS_TO_PULL=("$STAGE1_M" "$STAGE2_M")
fi

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

# 5. ตรวจสอบและดึงโมเดลที่กำหนด
for M in "${MODELS_TO_PULL[@]}"; do
    if ! ollama list | grep -q "$M"; then
        echo "Pulling $M into Ollama..."
        ollama pull "$M"
    fi
    echo "Model $M is verified and ready in Ollama!"
    # Warmup ping to ensure resident in VRAM
    curl -s -X POST http://127.0.0.1:11434/api/chat -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"keep_alive\":-1,\"options\":{\"num_predict\":1}}" > /dev/null 2>&1 || true
done

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

# 6.6 Background Supervisor Watchdog (Auto-Heals Ollama + Models + Self-Register)
echo "Starting Internal Auto-Healing Supervisor Watchdog..."
(
    while true; do
        sleep 10
        # Auto-heal Ollama daemon if terminated
        if ! pgrep -x "ollama" > /dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] Ollama daemon dead! Auto-restarting..." >> /root/whisper-server/watchdog.log 2>&1
            ollama serve >> /root/whisper-server/ollama.log 2>&1 &
            sleep 4
            for M in "${MODELS_TO_PULL[@]}"; do
                curl -s -X POST http://127.0.0.1:11434/api/chat -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"keep_alive\":-1,\"options\":{\"num_predict\":1}}" > /dev/null 2>&1 || true
            done
        fi

        # Auto-heal self_register_master.py if terminated
        if [ -f "/root/whisper-server/self_register_master.py" ] && ! pgrep -f "self_register_master.py" > /dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] self_register_master.py dead! Auto-restarting..." >> /root/whisper-server/watchdog.log 2>&1
            nohup $PYTHON_BIN /root/whisper-server/self_register_master.py >> /root/whisper-server/self_register_master.log 2>&1 &
        fi

        # Auto-heal port_bridge.py if terminated
        if [ -f "/root/whisper-server/port_bridge.py" ] && ! pgrep -f "port_bridge.py" > /dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] port_bridge.py dead! Auto-restarting..." >> /root/whisper-server/watchdog.log 2>&1
            nohup $PYTHON_BIN /root/whisper-server/port_bridge.py >> /root/whisper-server/port_bridge.log 2>&1 &
        fi
    done
) &

# 7. เริ่มต้น FastAPI / Uvicorn Server บนพอร์ต 10200 พร้อม Auto-Healing Restart Loop
UVICORN_BIN=$(which uvicorn || echo "$PYTHON_BIN -m uvicorn")
echo "=========================================================="
echo "=== STARTING FASTAPI MASTER LLM SERVICE (AUTO-HEAL ENABLED) ==="
echo "=========================================================="
while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Uvicorn llm_server on port 10200..."
    $UVICORN_BIN llm_server:app --host 0.0.0.0 --port 10200 --workers 1 || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AUTO-HEAL] Uvicorn exited. Restarting service in 3 seconds..."
    sleep 3
done
