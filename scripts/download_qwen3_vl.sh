#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/admin1/ct/Agentic-VLA"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
LOCAL_DIR="${LOCAL_DIR:-${PROJECT_ROOT}/weights/Qwen3-VL-8B-Instruct}"

mkdir -p "${LOCAL_DIR}"

echo "[INFO] Downloading ${MODEL_ID}"
echo "[INFO] Local dir: ${LOCAL_DIR}"

conda run -n openpi python /home/admin1/ct/Agentic-VLA/scripts/download_qwen3_vl.py \
  --model-id "${MODEL_ID}" \
  --local-dir "${LOCAL_DIR}"
