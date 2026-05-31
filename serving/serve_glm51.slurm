#!/bin/bash
#SBATCH --job-name=serve-glm51
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=/data/project/private/minstar/workspace/healthcare-research/serving/logs/serve_glm51_%j.log

# Serve GLM-5.1-FP8 with vLLM (OpenAI-compatible + tool calling) on one 8xH200 node.
# This endpoint backs the benchmark harness (_OpenAIBackend, real tool execution),
# the LLM judge, and bulk data synthesis (gold gen, Stage-2 re-judgment).
#
# GLM-5.1 arch = GlmMoeDsaForCausalLM (FP8, ~707GB, 78 layers, 200k ctx).
# Supported in vllm 0.19.1 (kimi env). Verified: registry + glm47 tool parser.
#
# Submit:  sbatch serving/serve_glm51.slurm
# Then on the serving node the endpoint is http://<node>:${PORT}/v1
# (the log prints the node hostname; use it as the harness --model base_url).

set -euo pipefail

MODEL=/data/project/public/checkpoints/GLM-5.1-FP8
SERVED_NAME=glm-5.1
PORT=8000
CONDA=/data/project/private/minstar/miniconda3
ENV=kimi
MAX_LEN=${MAX_LEN:-32768}     # full ctx is 202k; cap to preserve KV cache headroom
TP=8

mkdir -p /data/project/private/minstar/workspace/healthcare-research/serving/logs
source "${CONDA}/etc/profile.d/conda.sh"
conda activate "${ENV}"

# GLM-5.1 DSA needs DeepGEMM (copied into the kimi env from eval_sglang, abi3).
# Its JIT compiler needs a CUDA toolkit on PATH.
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
# Isolated, writable JIT cache — the shared /home/minstar/.cache/flashinfer got polluted
# by another venv's stale path, breaking the fmha ninja build (job 23019).
export XDG_CACHE_HOME="/data/project/private/minstar/.cache/glm_serve"
mkdir -p "$XDG_CACHE_HOME"
# deep_gemm v2.1.1 (built from source vs torch 2.10) links libc10.so → needs torch/lib on the loader path
TORCH_LIB="$(python -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
echo "nvcc: $(command -v nvcc) | $(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+')"
python -c "import deep_gemm; print('[ok] deep_gemm', deep_gemm.__version__)"

echo "=== node: $(hostname) | port: ${PORT} | model: ${MODEL} ==="
nvidia-smi --query-gpu=index,memory.total --format=csv,noheader

# Quick arch/parser sanity check before the (slow) load
python - <<'PY'
from vllm.model_executor.models.registry import ModelRegistry
assert "GlmMoeDsaForCausalLM" in ModelRegistry.get_supported_archs(), "arch unsupported"
print("[ok] GlmMoeDsaForCausalLM supported")
PY

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
