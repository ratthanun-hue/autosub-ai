#!/bin/bash
# ==============================================================================
# AutoSub-AI Dual-Image Build & Push Tool
# Builds Decoupled Docker Images:
#   1. Dynamic Node (Audio Worker: Steps 0-5) -> Dockerfile.dynamic
#   2. Master Node (LLM Master: Step 6)       -> Dockerfile.master
# ==============================================================================

set -e

REGISTRY="${1:-anakim}"
TAG="${2:-latest}"

echo "=========================================================="
echo "=== 1/2 BUILDING DYNAMIC WORKER (STEPS 0-5 AUDIO) ==="
echo "=========================================================="
docker build -f Dockerfile.dynamic -t "${REGISTRY}/autosub-worker:${TAG}" .

echo "=========================================================="
echo "=== 2/2 BUILDING MASTER LLM (STEP 6 QWEN2.5) ==="
echo "=========================================================="
docker build -f Dockerfile.master -t "${REGISTRY}/autosub-master:${TAG}" .

echo "=========================================================="
echo "=== BUILD COMPLETE ==="
echo "Images created:"
echo "  - ${REGISTRY}/autosub-worker:${TAG}"
echo "  - ${REGISTRY}/autosub-master:${TAG}"
echo "=========================================================="
echo "To push to Docker Hub, run:"
echo "  docker push ${REGISTRY}/autosub-worker:${TAG}"
echo "  docker push ${REGISTRY}/autosub-master:${TAG}"
echo "=========================================================="
