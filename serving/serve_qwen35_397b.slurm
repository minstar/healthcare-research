#!/bin/bash
#SBATCH --job-name=serve-qwen35-397b
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/data/project/private/minstar/workspace/healthcare-research/serving/logs/serve_qwen35_397b_%j.log

# Held-out baseline: Qwen3.5-397B-A17B (intermediate generation, frontier-scale, non-roster).
# Qwen3_5MoeForConditionalGeneration (same arch family as the roster Qwen3.6) ~755G bf16.
set -euo pipefail

MODEL=/data/project/public/checkpoints/Qwen3.5-397B-A17B
SERVED_NAME=qwen3.5-397b
PORT=8000
CONDA=/data/project/private/minstar/miniconda3
ENV=kimi
MAX_LEN=${MAX_LEN:-32768}
TP=8

mkdir -p /data/project/private/minstar/workspace/healthcare-research/serving/logs
source "${CONDA}/etc/profile.d/conda.sh"
conda activate "${ENV}"

echo "=== node: $(hostname) | port: ${PORT} | model: ${MODEL} ==="
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader

exec vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization 0.92 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3
