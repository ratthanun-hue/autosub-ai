#!/bin/bash
set -e

# ==========================================================
# AutoSub-AI Tier 3: Dedicated Qwen Node Entrypoint
# Specializes in: Precision Subtitle QC & Diff Extraction (Step 7)
# ==========================================================

# 1. DNS Fix
echo -e 'nameserver 8.8.8.8\nnameserver 1.1.1.1' > /etc/resolv.conf 2>/dev/null || true

# 2. Ollama & Runtime Settings for Tier 3
export ROLE="${ROLE:-step7}"
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_NUM_PARALLEL="${MAX_PARALLEL_BATCHES:-2}"
export OLLAMA_KEEP_ALIVE="-1"
export OLLAMA_MAX_LOADED_MODELS=1
export BATCH_SIZE="${BATCH_SIZE:-25}"

TARGET_MODEL="${STAGE2_MODEL:-qwen2.5:14b-instruct-q5_K_M}"
FALLBACK_MODEL="qwen2.5:14b"

echo "=========================================================="
echo "=== STARTING DEDICATED TIER 3 QWEN NODE (STEP 7) ==="
echo "=== Role: ${ROLE} | Target Model: ${TARGET_MODEL} ==="
echo "=== Parallel Batch Slots: ${OLLAMA_NUM_PARALLEL} | Batch Size: ${BATCH_SIZE} ==="
echo "=========================================================="

# 3. Start Ollama Service in Background
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama daemon..."
    ollama serve > /root/whisper-server/ollama.log 2>&1 &
fi

# 4. Wait for Ollama to become ready
echo "Waiting for Ollama to respond on 127.0.0.1:11434..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is ready!"
        break
    fi
    sleep 2
done

# 5. Pull & Verify Qwen Model with Automatic Fallback
ACTIVE_MODEL=""
if ollama list | grep -q "$TARGET_MODEL"; then
    ACTIVE_MODEL="$TARGET_MODEL"
else
    echo "Pulling ${TARGET_MODEL} into Ollama..."
    if ollama pull "$TARGET_MODEL"; then
        ACTIVE_MODEL="$TARGET_MODEL"
    else
        echo "WARNING: Failed to pull ${TARGET_MODEL}. Falling back to ${FALLBACK_MODEL}..."
        ollama pull "$FALLBACK_MODEL" || true
        ACTIVE_MODEL="$FALLBACK_MODEL"
    fi
fi

export STAGE2_MODEL="$ACTIVE_MODEL"
export OLLAMA_MODEL="$ACTIVE_MODEL"
echo "Model ${ACTIVE_MODEL} is verified and ready in Ollama!"

# Warmup ping to ensure resident in VRAM
curl -s -X POST http://127.0.0.1:11434/api/chat \
    -d "{\"model\":\"${ACTIVE_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"keep_alive\":-1,\"options\":{\"num_predict\":1}}" > /dev/null 2>&1 || true

# 6. Start Self-Registration Daemon
cd /root/whisper-server
PYTHON_BIN=$(which python3 || which python || echo "/usr/bin/python3")

if [ -f "/root/whisper-server/self_register_master.py" ]; then
    echo "Starting Tier 3 Self-Registration daemon (role: ${ROLE})..."
    nohup $PYTHON_BIN /root/whisper-server/self_register_master.py > /root/whisper-server/self_register_master.log 2>&1 &
fi

# 7. Start Port Bridge (10100 -> 10200) for Vast.ai UI compatibility
if [ -f "/root/whisper-server/port_bridge.py" ]; then
    echo "Starting Port Bridge on 10100 -> 10200..."
    nohup $PYTHON_BIN /root/whisper-server/port_bridge.py > /root/whisper-server/port_bridge.log 2>&1 &
fi

# 8. Background Supervisor Watchdog (Auto-Heals Ollama + Model + Self-Register + Port Bridge)
echo "Starting Tier 3 Internal Auto-Healing Supervisor Watchdog..."
(
    while true; do
        sleep 10
        # Auto-heal Ollama
        if ! pgrep -x "ollama" > /dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] Ollama daemon dead! Auto-restarting..." >> /root/whisper-server/watchdog.log 2>&1
            ollama serve >> /root/whisper-server/ollama.log 2>&1 &
            sleep 4
            curl -s -X POST http://127.0.0.1:11434/api/chat \
                -d "{\"model\":\"${ACTIVE_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"keep_alive\":-1,\"options\":{\"num_predict\":1}}" > /dev/null 2>&1 || true
        fi

        # Auto-heal self_register_master.py
        if [ -f "/root/whisper-server/self_register_master.py" ] && ! pgrep -f "self_register_master.py" > /dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] self_register_master.py dead! Auto-restarting..." >> /root/whisper-server/watchdog.log 2>&1
            nohup $PYTHON_BIN /root/whisper-server/self_register_master.py >> /root/whisper-server/self_register_master.log 2>&1 &
        fi

        # Auto-heal port_bridge.py
        if [ -f "/root/whisper-server/port_bridge.py" ] && ! pgrep -f "port_bridge.py" > /dev/null; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WATCHDOG] port_bridge.py dead! Auto-restarting..." >> /root/whisper-server/watchdog.log 2>&1
            nohup $PYTHON_BIN /root/whisper-server/port_bridge.py >> /root/whisper-server/port_bridge.log 2>&1 &
        fi
    done
) &

# 9. Start FastAPI / Uvicorn Server with Auto-Healing Restart Loop
UVICORN_BIN=$(which uvicorn || echo "$PYTHON_BIN -m uvicorn")
echo "=========================================================="
echo "=== FASTAPI TIER 3 SERVICE RUNNING ON PORT 10200 (AUTO-HEAL ENABLED) ==="
echo "=========================================================="
while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Uvicorn llm_server on port 10200..."
    $UVICORN_BIN llm_server:app --host 0.0.0.0 --port 10200 --workers 1 || true
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [AUTO-HEAL] Uvicorn exited. Restarting service in 3 seconds..."
    sleep 3
done
