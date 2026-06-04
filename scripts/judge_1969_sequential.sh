#!/usr/bin/env bash
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy
GJOB=$(cat serving/.glm_jobid)
# wait for GLM up
for i in $(seq 1 60); do
  GN=$(squeue -j $GJOB -h -o '%N' 2>/dev/null)
  [ -n "$GN" ] && curl -s --max-time 4 http://$GN:8000/v1/models 2>/dev/null | grep -q glm-5.1 && { echo "$GN">serving/.serving_node; break; }
  [ -z "$(squeue -j $GJOB -h -o '%T' 2>/dev/null)" ] && { echo "GLM job gone"; exit 1; }
  sleep 30
done
GN=$(cat serving/.serving_node); R=data/eval_samples/rubrics_1969.jsonl
echo "GLM up at $GN; judging SEQUENTIALLY"
for m in glm qwen dsv4; do
  echo "=== judging $m (sequential) ==="
  python scripts/checklist_judge.py --traces results/baseline_${m}_1969/traces.jsonl \
    --rubrics $R --judge openai/glm-5.1 --base http://$GN:8000/v1 \
    --out results/baseline_${m}_1969/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx | tail -2
done
echo "SEQJUDGE_DONE"
