#!/usr/bin/env bash
# FREE (local GLM-5.1 judge). Orchestrates the frontier-leaderboard judging end-to-end:
#   1. wait for GPT-5.5 (tools + no-tools) trace generation to finish (423 each)
#   2. judge GPT-5.5 traces with the local GLM-5.1 judge (opus+gemini judged separately)
#   3. wait until all 4 frontier tags are fully judged (423 each)
# Endpoint is resolved DYNAMICALLY each judging call (serve-glm51 lives on the
# preemptible partition, so its node can change after a preemption/restart).
# Resumable: checklist_judge skips already-scored task_ids.
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
RUB=data/eval_samples/rubrics_1969_uid.jsonl
LOG=results/orchestrate_frontier_judge.log
: > "$LOG"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }

n(){ wc -l < "$1" 2>/dev/null || echo 0; }
scored(){ python - "$1" <<'PY'
import json,sys
try: print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get("checklist_score") is not None))
except FileNotFoundError: print(0)
PY
}
glm_node(){
  local jid node
  jid=$(squeue -u minstar -h -n serve-glm51 -o '%i' 2>/dev/null | head -1)
  [ -z "$jid" ] && return 1
  node=$(squeue -j "$jid" -h -o '%N' 2>/dev/null)
  [ -z "$node" ] && return 1
  curl -s --max-time 4 "http://$node:8000/v1/models" 2>/dev/null | grep -q glm-5.1 || return 1
  echo "$node"
}

# --- 1. wait for GPT-5.5 trace generation (both variants) ---
log "waiting for GPT-5.5 trace gen (tools + no-tools) to reach 423 each"
until [ "$(n results/api_gpt55_robust/traces.jsonl)" -ge 423 ] && \
      [ "$(n results/api_gpt55_notool_robust/traces.jsonl)" -ge 423 ]; do
  sleep 120
done
log "GPT55_TRACES_DONE tools=$(n results/api_gpt55_robust/traces.jsonl) notool=$(n results/api_gpt55_notool_robust/traces.jsonl)"

# --- 2. judge the two GPT-5.5 tags ---
for tag in api_gpt55_robust api_gpt55_notool_robust; do
  if [ "$(scored results/$tag/checklist_glm.jsonl)" -ge 423 ]; then
    log "[skip] $tag already judged"; continue
  fi
  GN=""
  for i in $(seq 1 120); do GN=$(glm_node) && [ -n "$GN" ] && break; sleep 30; done
  [ -z "$GN" ] && { log "ABORT: no GLM-5.1 judge endpoint for $tag"; exit 1; }
  log "[judge] $tag @ $GN"
  OPENAI_API_KEY=dummy python scripts/checklist_judge.py \
    --traces results/$tag/traces.jsonl --rubrics "$RUB" \
    --judge openai/glm-5.1 --base "http://$GN:8000/v1" \
    --out results/$tag/checklist_glm.jsonl --workers 8 >> "$LOG" 2>&1
  log "[done] $tag scored=$(scored results/$tag/checklist_glm.jsonl)/423"
done

# --- 3. wait until ALL 4 frontier tags are fully judged ---
log "waiting for all 4 frontier tags to reach 423 judged"
until [ "$(scored results/api_opus47_robust/checklist_glm.jsonl)"        -ge 423 ] && \
      [ "$(scored results/api_gemini3pro_robust/checklist_glm.jsonl)"    -ge 423 ] && \
      [ "$(scored results/api_gpt55_robust/checklist_glm.jsonl)"         -ge 423 ] && \
      [ "$(scored results/api_gpt55_notool_robust/checklist_glm.jsonl)"  -ge 423 ]; do
  sleep 120
done
log "ALL_FRONTIER_4_DONE"
for t in api_opus47_robust api_gemini3pro_robust api_gpt55_robust api_gpt55_notool_robust; do
  log "  $t: scored=$(scored results/$t/checklist_glm.jsonl)/423"
done
echo "ALL_FRONTIER_4_DONE" >> "$LOG"
