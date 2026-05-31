#!/usr/bin/env bash
# Wait for GLM endpoint, then launch all verification + empirical-eval tracks.
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy

GJOB=23035; QJOB=23020; VJOB=23021
# --- wait for GLM up ---
for i in $(seq 1 60); do
  GN=$(squeue -j $GJOB -h -o '%N' 2>/dev/null)
  if [ -n "$GN" ] && curl -s --max-time 4 http://$GN:8000/v1/models 2>/dev/null | grep -q glm-5.1; then
    echo "$GN" > serving/.serving_node; break
  fi
  [ -z "$(squeue -j $GJOB -h -o '%T' 2>/dev/null)" ] && { echo "GLM job gone"; exit 1; }
  sleep 30
done
GN=$(cat serving/.serving_node); QN=$(squeue -j $QJOB -h -o '%N' 2>/dev/null); VN=$(squeue -j $VJOB -h -o '%N' 2>/dev/null)
echo "ENDPOINTS glm=$GN qwen=$QN v4=$VN"

run(){ nohup bash -c "$1" >/dev/null 2>&1 & echo "launched: $2"; }

# --- verification tracks (GLM judge; small first) ---
run "source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh; conda activate kimi; export OPENAI_API_KEY=dummy; python scripts/audit_self_containment.py --data data/export/mcp_benchmark_v3.jsonl --model glm-5.1 --base http://$GN:8000/v1 --sample 500 --out results/self_containment.jsonl > /tmp/selfcon.log 2>&1; echo DONE>>/tmp/selfcon.log" "self_containment(500)"

run "source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh; conda activate kimi; export OPENAI_API_KEY=dummy; python scripts/audit_citation_relevance_llm.py --gold data/export/mcp_benchmark_with_gold.jsonl --model glm-5.1 --base http://$GN:8000/v1 --sample 400 --out results/citation_relevance_llm.jsonl > /tmp/citllm.log 2>&1; echo DONE>>/tmp/citllm.log" "citation_judge(400)"

# --- rubric generation for the full 1969 (upgraded few-shot prompt) ---
run "source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh; conda activate kimi; export OPENAI_API_KEY=dummy; python scripts/gen_rubrics.py --data data/export/mcp_benchmark_with_gold.jsonl --model glm-5.1 --base http://$GN:8000/v1 --out data/eval_samples/rubrics_1969.jsonl --workers 8 > /tmp/rub1969.log 2>&1; echo DONE>>/tmp/rub1969.log" "rubric_gen(1969)"

# --- Stage 2 grounded open_status re-judgment over the retrieval track ---
run "source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh; conda activate kimi; export OPENAI_API_BASE=http://$GN:8000/v1 OPENAI_API_KEY=dummy; python scripts/stage2_judge.py --data data/export/mcp_benchmark_v2.jsonl --sample 100000 --model glm-5.1 --workers 6 --out data/stage2_full > /tmp/stage2.log 2>&1; echo DONE>>/tmp/stage2.log" "stage2_full(retrieval)"

# --- (1) 3-model agentic eval on 1969 ---
run "source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh; conda activate kimi; export OPENAI_API_BASE=http://$QN:8000/v1 OPENAI_API_KEY=dummy; python harness/run.py --data data/export/mcp_benchmark_with_gold.jsonl --model 'openai::http://$QN:8000/v1::qwen-3.6' --no-judge --limit 2000 --workers 12 --output results/baseline_qwen_1969 > results/baseline_qwen_1969.log 2>&1; echo DONE_\$?>>results/baseline_qwen_1969.log" "eval_qwen(1969)"
run "source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh; conda activate kimi; export OPENAI_API_BASE=http://$VN:8000/v1 OPENAI_API_KEY=dummy; python harness/run.py --data data/export/mcp_benchmark_with_gold.jsonl --model 'openai::http://$VN:8000/v1::deepseek-v4-flash' --no-judge --limit 2000 --workers 12 --output results/baseline_dsv4_1969 > results/baseline_dsv4_1969.log 2>&1; echo DONE_\$?>>results/baseline_dsv4_1969.log" "eval_v4(1969)"
run "source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh; conda activate kimi; export OPENAI_API_BASE=http://$GN:8000/v1 OPENAI_API_KEY=dummy; python harness/run.py --data data/export/mcp_benchmark_with_gold.jsonl --model 'openai::http://$GN:8000/v1::glm-5.1' --no-judge --limit 2000 --workers 10 --output results/baseline_glm_1969 > results/baseline_glm_1969.log 2>&1; echo DONE_\$?>>results/baseline_glm_1969.log" "eval_glm(1969)"

echo "ALL_LAUNCHED"
