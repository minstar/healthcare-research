#!/bin/bash
#SBATCH --job-name=serve-qwen36
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/data/project/private/minstar/workspace/healthcare-research/serving/logs/serve_qwen36_%j.log

# Serve Qwen3.6-35B-A3B (Qwen3_5MoeForConditionalGeneration, ~67G bf16) with vLLM.
# Standard MoE — no DeepGEMM/DSA needed. Second eval target for the benchmark, to
# test whether the difficulty inversion + parametric-fallback pattern is the MODEL
# or the QUESTIONS.
#
# Submit:  sbatch serving/serve_qwen36.slurm
# Endpoint: http://<node>:8000/v1  (node printed in the log)

set -euo pipefail

MODEL=/data/project/public/checkpoints/Qwen3.6-35B-A3B
SERVED_NAME=qwen-3.6
PORT=8000
CONDA=/data/project/private/minstar/miniconda3
ENV=kimi
MAX_LEN=${MAX_LEN:-32768}
TP=4

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

exec vllm serve "${MODEL}" \
    --served-model-name "${SERVED_NAME}" \
    --host 0.0.0.0 --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3
