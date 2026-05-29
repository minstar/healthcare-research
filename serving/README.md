# Serving

OpenAI-compatible model endpoints that back the benchmark harness
(`_OpenAIBackend`, real tool execution), the LLM judge, and bulk data synthesis.

## GLM-5.1-FP8 (`serve_glm51.slurm`)

1-node 8×H200, TP=8, vLLM 0.19.1 (`kimi` conda env). Model:
`/data/project/public/checkpoints/GLM-5.1-FP8` (707G FP8, arch `GlmMoeDsaForCausalLM`).

### DeepGEMM dependency (DSA)

GLM-5.1 uses DeepSeek Sparse Attention (`glm_moe_dsa`); vLLM's sparse-attention
indexer requires **DeepGEMM**, which is not in the `kimi` env by default. It is
present in the `eval_sglang` env as an `abi3` wheel (Python-version-portable), so
it was installed into `kimi` by copying the package:

```bash
cp -r /data/project/private/minstar/miniconda3/envs/eval_sglang/lib/python3.12/site-packages/deep_gemm \
      /data/project/private/minstar/miniconda3/envs/kimi/lib/python3.11/site-packages/deep_gemm
# verify: python -c "import deep_gemm"   (abi3 .so loads under py3.11)
```

DeepGEMM JIT-compiles kernels at runtime, so the serve script exports
`CUDA_HOME=/usr/local/cuda` (toolkit 13.0 on the nodes) and puts `nvcc` on PATH.

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
