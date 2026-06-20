#!/usr/bin/env bash
# Held-out local models: run all 3 evals IN PARALLEL against their already-up endpoints,
# and scancel each model's serve job the moment its generation is done (free idle GPU).
# Judging (against the GLM-5.1 judge, which stays up) happens after. no-tool GLM-5.1 already
# has traces; it is just (re)judged here.
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy
LOG=results/baselines_run.log
CORE=data/eval_samples/core_eval.jsonl
RUB=data/eval_samples/rubrics_1969_uid.jsonl
NCORE=$(wc -l < $CORE)
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }
node_of(){ local jid=$(squeue -u minstar -h -n "$1" -o '%i' 2>/dev/null|head -1); [ -n "$jid" ] && squeue -j "$jid" -h -o '%N' 2>/dev/null; }
jid_of(){ squeue -u minstar -h -n "$1" -o '%i' 2>/dev/null|head -1; }
cnt(){ python - "$1" <<'PY'
import json,sys
try: print(len({json.loads(l)["task_id"] for l in open(sys.argv[1]) if json.loads(l).get("checklist_score") is not None}))
except FileNotFoundError: print(0)
PY
}

# eval one held-out (parallel), then scancel its serve job to free the GPU
run_heldout(){ # jobname served tag
  local jn=$1 sv=$2 tag=$3 node jid
  for i in $(seq 1 60); do node=$(node_of "$jn"); [ -n "$node" ] && curl -s --max-time 4 "http://$node:8000/v1/models" 2>/dev/null | grep -q "$sv" && break; sleep 30; done
  [ -z "$node" ] && { log "[$tag] no endpoint, skip"; return; }
  if [ "$(wc -l < results/baseline_${tag}/traces.jsonl 2>/dev/null||echo 0)" -lt "$NCORE" ]; then
    log "[eval//] $tag @ $node (parallel)"
    OPENAI_API_BASE=http://$node:8000/v1 python harness/run.py --data $CORE \
      --model "openai::http://$node:8000/v1::$sv" --no-judge --workers 12 \
      --output results/baseline_${tag} >> "$LOG" 2>&1
  fi
  jid=$(jid_of "$jn")
  [ -n "$jid" ] && scancel "$jid" && log "[gpu freed] scancel $jn ($jid) — $tag generation done ($(wc -l < results/baseline_${tag}/traces.jsonl) traces)"
}

log "=== held-out PARALLEL eval (free each GPU on completion) ==="
run_heldout serve-glm5          glm-5        heldout_glm5         &
run_heldout serve-qwen3-235b    qwen3-235b   heldout_qwen3_235b   &
run_heldout serve-qwen35-397b   qwen3.5-397b heldout_qwen35_397b  &
wait
log "=== all held-out generation done; GPUs freed. Judging (GLM-5.1 judge) ==="

GN=$(node_of serve-glm51); [ -z "$GN" ] && { log "GLM judge gone, abort judging"; exit 1; }
for tag in notool_glm51 heldout_glm5 heldout_qwen3_235b heldout_qwen35_397b; do
  [ "$(cnt results/baseline_${tag}/checklist_glm.jsonl)" -lt "$NCORE" ] || continue
  log "[judge] $tag"
  python scripts/checklist_judge.py --traces results/baseline_${tag}/traces.jsonl \
    --rubrics $RUB --judge openai/glm-5.1 --base http://$GN:8000/v1 \
    --out results/baseline_${tag}/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx >> "$LOG"
done
log "=== local baselines judged. (GLM-5.1 judge kept for API track) ==="
python scripts/build_leaderboard.py 2>&1 | tee -a "$LOG"
echo "HELDOUT_PARALLEL_DONE" | tee -a "$LOG"
