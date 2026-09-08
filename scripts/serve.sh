#!/usr/bin/env bash
# Serve the vision model with vLLM (OpenAI-compatible API on :8000).
# Usage: scripts/serve.sh [model] [gpu-id]
set -euo pipefail
MODEL="${1:-Qwen/Qwen3-VL-8B-Instruct}"
GPU="${2:-0}"
cd "$(dirname "$0")/.."
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
CUDA_VISIBLE_DEVICES="$GPU" exec .venv/bin/vllm serve "$MODEL" \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 16}' \
    --max-num-seqs 16 \
    --served-model-name "$MODEL"
