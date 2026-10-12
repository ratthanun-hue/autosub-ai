#!/bin/bash
export OLLAMA_MODEL="qwen2.5:32b"
export GPU_NAME="Tesla V100 32GB"
export VRAM_GB="32"
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_KEEP_ALIVE="-1"
export OLLAMA_NUM_PARALLEL="2"
export MAX_PARALLEL_BATCHES="2"
export BATCH_SIZE="60"
cd /root/whisper-server
exec /usr/bin/python3 -m uvicorn llm_server:app --host 0.0.0.0 --port 10200
