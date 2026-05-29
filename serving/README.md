# Serving

OpenAI-compatible model endpoints that back the benchmark harness
(`_OpenAIBackend`, real tool execution), the LLM judge, and bulk data synthesis.

## GLM-5.1-FP8 (`serve_glm51.slurm`)

1-node 8×H200, TP=8, vLLM 0.19.1 (`kimi` conda env). Model:
`/data/project/public/checkpoints/GLM-5.1-FP8` (707G FP8, arch `GlmMoeDsaForCausalLM`).

### DeepGEMM dependency (DSA)

GLM-5.1 uses DeepSeek Sparse Attention (`glm_moe_dsa`); vLLM's sparse-attention
indexer + FP8-MoE weight post-processing require **DeepGEMM**, which is not in the
`kimi` env by default.

> ⚠️ Copying the prebuilt `deep_gemm` from `eval_sglang` does NOT work: although its
> `.so` is Python-`abi3` (loads under py3.11), it was built against torch 2.9.1 while
> `kimi` has torch 2.10.0. The **torch C++ ABI is not stable across versions**, so
> the custom op crashes at load with
> `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage`.

DeepGEMM must be **built from source against the kimi env's torch 2.10**:

```bash
conda activate kimi
export CUDA_HOME=/usr/local/cuda            # toolkit 13.0 on the nodes
export PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST="9.0a"   # H200 = sm90a
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM && ./install.sh                 # builds deep_gemm._C vs torch 2.10 → wheel → pip install
# verify (outside the source dir): python -c "import deep_gemm; print(deep_gemm.__version__)"  # 2.5.0
```

The C++ core is small (GEMM kernels JIT-compile at runtime); the serve script also
exports `CUDA_HOME` so that runtime JIT finds `nvcc`.

### Run

```bash
sbatch serving/serve_glm51.slurm
# endpoint: http://<node>:8000/v1  (node printed in serving/logs/serve_glm51_<jobid>.log)
```

The serve job prints the node hostname; use it as the harness `--model` base_url
and as `OPENAI_API_BASE` for the judge.

## Roles (avoid circularity)

- GLM-5.1 → bulk synthesis (gold gen, Stage-2 re-judgment) + judge of **other** models.
- Eval targets → GLM-5.1 (self, sanity only) and Qwen3.6-35B-A3B.
- Never let a model judge its own answers against its own gold.
