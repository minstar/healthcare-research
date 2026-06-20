#!/usr/bin/env bash
# Frontier closed-API baselines (held-out) on the 657 core set, via litellm.
# 3 distinct families: gpt-4.1 (OpenAI), gemini-2.5-pro (Google), claude-sonnet-4.5 (Anthropic/OpenRouter).
# Judged by the local GLM-5.1 checklist judge (free). Token usage is recorded by the harness;
# cost is computed at the end. Resumable.
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
# load API keys, but DROP the Anthropic proxy vars so litellm uses real providers / OpenRouter
set -a; source /data/project/private/minstar/.env 2>/dev/null; set +a
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL
export OPENAI_API_KEY_SERVED=dummy   # served-model judge uses its own base; keep real OPENAI_API_KEY for litellm
LOG=results/api_baselines_run.log
CORE=data/eval_samples/core_eval.jsonl
RUB=data/eval_samples/rubrics_1969_uid.jsonl
NCORE=$(wc -l < $CORE)
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }

# resolve local GLM-5.1 judge endpoint (served by run_baselines on normal)
GN=""
for i in $(seq 1 120); do
  jid=$(squeue -u minstar -h -n serve-glm51 -o '%i' 2>/dev/null | head -1)
  [ -n "$jid" ] && n=$(squeue -j "$jid" -h -o '%N' 2>/dev/null) && [ -n "$n" ] && \
    curl -s --max-time 4 "http://$n:8000/v1/models" 2>/dev/null | grep -q glm-5.1 && { GN=$n; break; }
  sleep 60
done
[ -z "$GN" ] && { log "no GLM-5.1 judge endpoint; abort"; exit 1; }
log "GLM-5.1 judge @ $GN | core n=$NCORE"

cnt(){ python - "$1" <<'PY'
import json,sys
try: print(len({json.loads(l)["task_id"] for l in open(sys.argv[1]) if json.loads(l).get("checklist_score") is not None}))
except FileNotFoundError: print(0)
PY
}

run_api(){ # $1=tag $2=litellm-model  (OPENAI_API_KEY for litellm must be the REAL key)
  local tag=$1 model=$2
  if [ "$(wc -l < results/${tag}/traces.jsonl 2>/dev/null || echo 0)" -lt "$NCORE" ]; then
    log "[eval] $tag ($model) on core; workers=6"
    python harness/run.py --data $CORE --model "litellm::$model" --no-judge \
      --workers 6 --output results/${tag} >> "$LOG" 2>&1
  fi
  if [ "$(cnt results/${tag}/checklist_glm.jsonl)" -lt "$NCORE" ]; then
    log "[judge] $tag (GLM-5.1 @ $GN)"
    OPENAI_API_KEY=dummy python scripts/checklist_judge.py --traces results/${tag}/traces.jsonl \
      --rubrics $RUB --judge openai/glm-5.1 --base http://$GN:8000/v1 \
      --out results/${tag}/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx >> "$LOG"
  fi
  log "[done] $tag scored=$(cnt results/${tag}/checklist_glm.jsonl)/$NCORE"
}

run_api api_gpt55       "gpt-5.5"
run_api api_opus47      "openrouter/anthropic/claude-opus-4.7"
run_api api_gemini31pro "gemini/gemini-3.1-pro-preview"

log "=== token + cost accounting ==="
OPENAI_API_KEY=dummy python scripts/compute_api_cost.py 2>&1 | tee -a "$LOG"
echo "API_BASELINES_DONE" | tee -a "$LOG"
