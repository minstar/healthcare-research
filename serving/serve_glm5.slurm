#!/bin/bash
#SBATCH --job-name=serve-glm5
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/data/project/private/minstar/workspace/healthcare-research/serving/logs/serve_glm5_%j.log

# Held-out baseline: GLM-5 (predecessor of GLM-5.1). Same arch (GlmMoeDsaForCausalLM, FP8).
set -euo pipefail

MODEL=/data/project/public/checkpoints/GLM-5-FP8
SERVED_NAME=glm-5
PORT=8000
CONDA=/data/project/private/minstar/miniconda3
ENV=kimi
MAX_LEN=${MAX_LEN:-32768}
TP=8

mkdir -p /data/project/private/minstar/workspace/healthcare-research/serving/logs
source "${CONDA}/etc/profile.d/conda.sh"
conda activate "${ENV}"

export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export XDG_CACHE_HOME="/data/project/private/minstar/.cache/glm5_serve"
mkdir -p "$XDG_CACHE_HOME"
TORCH_LIB="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
python -c "import deep_gemm; print('[ok] deep_gemm', deep_gemm.__version__)"

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
    --tool-call-parser glm47 \
    --reasoning-parser glm45
