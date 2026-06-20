#!/usr/bin/env bash
# Baseline experiment (①+③): held-out models on the 657 core set (circularity test) +
# no-tool ablation + self-containment before/after. GLM-5.1 is the checklist judge.
# Resumable; stderr-only logging in wait_ep so $node capture stays clean.
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

# resolve a serving node by jobname (normal partition; stays up)
ep(){ # $1=jobname $2=served-name -> echoes node (stdout) once healthy
  local jn=$1 sv=$2 jid n st
  while :; do
    jid=$(squeue -u minstar -h -n "$jn" -o '%i' 2>/dev/null | head -1)
    [ -z "$jid" ] && { log "no job $jn in queue; abort step"; echo ""; return 1; }
    st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null)
    if [ "$st" = "RUNNING" ]; then
      n=$(squeue -j "$jid" -h -o '%N' 2>/dev/null)
      curl -s --max-time 4 "http://$n:8000/v1/models" 2>/dev/null | grep -q "$sv" && { echo "$n"; return 0; }
    fi
    sleep 60
  done
}
cnt(){ python - "$1" <<'PY'
import json,sys
try: print(len({json.loads(l)["task_id"] for l in open(sys.argv[1]) if json.loads(l).get("checklist_score") is not None}))
except FileNotFoundError: print(0)
PY
}
eval_model(){ # $1=node $2=served $3=tag $4=extra(harness flags)
  local node=$1 sv=$2 tag=$3 extra=${4:-}
  if [ "$(wc -l < results/baseline_${tag}/traces.jsonl 2>/dev/null || echo 0)" -lt "$NCORE" ]; then
    log "[eval] $tag on core ($NCORE)"
    OPENAI_API_BASE=http://$node:8000/v1 python harness/run.py --data $CORE \
      --model "openai::http://$node:8000/v1::$sv" --no-judge --workers 10 $extra \
      --output results/baseline_${tag} >> "$LOG" 2>&1
  fi
  if [ "$(cnt results/baseline_${tag}/checklist_glm.jsonl)" -lt "$NCORE" ]; then
    log "[judge] $tag"
    python scripts/checklist_judge.py --traces results/baseline_${tag}/traces.jsonl \
      --rubrics $RUB --judge openai/glm-5.1 --base http://$GN:8000/v1 \
      --out results/baseline_${tag}/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx >> "$LOG"
  fi
}

log "=== baseline experiment start (core n=$NCORE) ==="
GN=$(ep serve-glm51 glm-5.1); [ -z "$GN" ] && { log "no GLM judge; abort"; exit 1; }
log "GLM-5.1 judge @ $GN"

# ③ self-containment BEFORE (extracted/pre-refine), GLM judge
if [ ! -s results/self_containment_before.jsonl ]; then
  log "[self-cont before] auditing extracted (pre-refine) sample"
  python scripts/audit_self_containment.py --data data/extracted/all_extracted_questions.jsonl \
    --model glm-5.1 --base http://$GN:8000/v1 --sample 500 \
    --out results/self_containment_before.jsonl >> "$LOG" 2>&1 || log "self-cont before failed"
fi

# no-tool baseline (GLM-5.1, tools disabled) on core
eval_model "$GN" glm-5.1 notool_glm51 "--no-tools"

# held-out models on core (each waits for its own endpoint)
for spec in "serve-glm5 glm-5 heldout_glm5" "serve-qwen3-235b qwen3-235b heldout_qwen3_235b" "serve-qwen35-397b qwen3.5-397b heldout_qwen35_397b"; do
  set -- $spec; JN=$1; SV=$2; TAG=$3
  N=$(ep "$JN" "$SV"); [ -z "$N" ] && { log "skip $TAG (no endpoint)"; continue; }
  log "$TAG @ $N"
  eval_model "$N" "$SV" "$TAG"
  scancel $(squeue -u minstar -h -n "$JN" -o '%i' 2>/dev/null | head -1) 2>/dev/null && log "scancel $JN (eval done)"
done

# leaderboard
log "=== building leaderboard ==="
python scripts/build_leaderboard.py 2>&1 | tee -a "$LOG"
scancel $(squeue -u minstar -h -n serve-glm51 -o '%i' 2>/dev/null | head -1) 2>/dev/null && log "scancel GLM judge"
echo "BASELINES_DONE" | tee -a "$LOG"
