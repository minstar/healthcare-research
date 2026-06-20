#!/usr/bin/env bash
# FREE (local GLM-5.1 judge). Scores the full-core(657) frontier traces from
# run_frontier_fullcore.sh with the SAME rubric + judge as the roster/frozen leaderboard,
# so the new 657 numbers are comparable to the frozen-423 column. Mirrors
# judge_frontier_robust.sh but over core_eval.jsonl(657) and the *_fullcore dirs.
# Run ONLY after the money script logs FRONTIER_FULLCORE_GEN_DONE.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-kimi}"
RUB=data/eval_samples/rubrics_1969_uid.jsonl
DATA=data/eval_samples/core_eval.jsonl
LOG=results/judge_fullcore.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }

# --- coverage guard: every full-core id MUST have a rubric, else abort BEFORE judging ---
python - "$DATA" "$RUB" <<'PY' || exit 1
import json,sys
ids={json.loads(l)["task_id"] for l in open(sys.argv[1])}
rub={json.loads(l)["task_id"] for l in open(sys.argv[2])}
miss=ids-rub
print(f"coverage: {len(ids)-len(miss)}/{len(ids)} full-core ids have rubrics")
sys.exit(1 if miss else 0)
PY
log "rubric coverage OK (657)"

# --- resolve local GLM-5.1 judge endpoint (serve-glm51) ---
GN=""
for i in $(seq 1 120); do
  jid=$(squeue -u "$USER" -h -n serve-glm51 -o '%i' 2>/dev/null | head -1)
  [ -n "$jid" ] && n=$(squeue -j "$jid" -h -o '%N' 2>/dev/null) && [ -n "$n" ] && \
    curl -s --max-time 4 "http://$n:8000/v1/models" 2>/dev/null | grep -q glm-5.1 && { GN=$n; break; }
  sleep 60
done
[ -z "$GN" ] && { log "no GLM-5.1 judge endpoint; abort"; exit 1; }
log "GLM-5.1 judge @ $GN"

cnt(){ python - "$1" <<'PY'
import json,sys
try: print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get("checklist_score") is not None))
except FileNotFoundError: print(0)
PY
}

N=$(wc -l < "$DATA")
for tag in api_gpt55_fullcore api_opus47_fullcore api_gemini3pro_fullcore api_gpt55_notool_fullcore; do
  tr=results/${tag}/traces.jsonl
  [ -f "$tr" ] || { log "[skip] $tag no traces"; continue; }
  if [ "$(cnt results/${tag}/checklist_glm.jsonl)" -lt "$N" ]; then
    log "[judge] $tag (GLM-5.1 @ $GN)"
    OPENAI_API_KEY=dummy python scripts/checklist_judge.py --traces "$tr" \
      --rubrics "$RUB" --judge openai/glm-5.1 --base "http://$GN:8000/v1" \
      --out results/${tag}/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx >> "$LOG"
  fi
  log "[done] $tag scored=$(cnt results/${tag}/checklist_glm.jsonl)/$N"
done
echo "JUDGE_FULLCORE_DONE" | tee -a "$LOG"
