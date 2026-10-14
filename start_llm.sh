#!/bin/bash
export STAGE1_MODEL="scb10x/typhoon2.1-gemma3-12b:latest"
export STAGE2_MODEL="qwen2.5:14b-instruct-q5_K_M"
export OLLAMA_MODEL="qwen2.5:14b-instruct-q5_K_M"
export GPU_NAME="Tesla V100 32GB"
export VRAM_GB="32"
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_KEEP_ALIVE="-1"
export OLLAMA_NUM_PARALLEL="2"
export MAX_PARALLEL_BATCHES="2"
export BATCH_SIZE="25"
cd /root/whisper-server
exec /usr/bin/python3 -m uvicorn llm_server:app --host 0.0.0.0 --port 10200

