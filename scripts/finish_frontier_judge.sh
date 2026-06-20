#!/usr/bin/env bash
# Finishing pass: after serve-glm51 (24004) resumes from preemption, judge the
# ~9 remaining frontier traces across all 4 tags (resumable: scored ids are skipped).
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
RUB=data/eval_samples/rubrics_1969_uid.jsonl
LOG=results/finish_frontier_judge.log; : > "$LOG"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }
glm_node(){ local jid node
  jid=$(squeue -u minstar -h -n serve-glm51 -o '%i' 2>/dev/null|head -1); [ -z "$jid" ]&&return 1
  node=$(squeue -j "$jid" -h -t R -o '%N' 2>/dev/null); [ -z "$node" ]&&return 1
  curl -s --max-time 4 "http://$node:8000/v1/models" 2>/dev/null|grep -q glm-5.1||return 1
  echo "$node"; }
scored(){ python - "$1" <<'PY'
import json,sys
try: print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get("checklist_score") is not None))
except FileNotFoundError: print(0)
PY
}
log "waiting for serve-glm51 to resume + become healthy"
GN=""; while :; do GN=$(glm_node) && [ -n "$GN" ] && break; sleep 60; done
log "GLM-5.1 judge back @ $GN"
for tag in api_opus47_robust api_gemini3pro_robust api_gpt55_robust api_gpt55_notool_robust; do
  [ "$(scored results/$tag/checklist_glm.jsonl)" -ge 423 ] && { log "[ok] $tag already 423"; continue; }
  log "[finish] $tag (have $(scored results/$tag/checklist_glm.jsonl)/423)"
  OPENAI_API_KEY=dummy python scripts/checklist_judge.py --traces results/$tag/traces.jsonl \
    --rubrics "$RUB" --judge openai/glm-5.1 --base "http://$GN:8000/v1" \
    --out results/$tag/checklist_glm.jsonl --workers 8 >> "$LOG" 2>&1
  log "[done] $tag scored=$(scored results/$tag/checklist_glm.jsonl)/423"
done
log "FINISH_FRONTIER_JUDGE_DONE"
