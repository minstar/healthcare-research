#!/bin/bash
#SBATCH --job-name=serve-qwen3-235b
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/data/project/private/minstar/workspace/healthcare-research/serving/logs/serve_qwen3_235b_%j.log

# Held-out baseline: Qwen3-235B-A22B-Instruct-2507 (older Qwen3 generation, non-roster).
# Qwen3MoeForCausalLM, bf16 ~440G. Non-thinking instruct -> no reasoning parser.
set -euo pipefail

MODEL=/data/project/public/checkpoints/Qwen3-235B-A22B-Instruct-2507
SERVED_NAME=qwen3-235b
PORT=8000
CONDA=/data/project/private/minstar/miniconda3
ENV=kimi
MAX_LEN=${MAX_LEN:-32768}
TP=8

mkdir -p /data/project/private/minstar/workspace/healthcare-research/serving/logs
source "${CONDA}/etc/profile.d/conda.sh"
conda activate "${ENV}"

# Isolate flashinfer JIT cache from the polluted shared ~/.cache/flashinfer (other venvs leave
# stale .cu source paths there, breaking the ninja fmha build; see job 23019/24492).
export FLASHINFER_WORKSPACE_BASE=/data/project/private/minstar/.cache/flashinfer_hc
mkdir -p "$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

echo "=== node: $(hostname) | port: ${PORT} | model: ${MODEL} ==="
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader

exec vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml
