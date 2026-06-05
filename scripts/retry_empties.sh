#!/usr/bin/env bash
# Re-fetch responses for EMPTY/[ERROR] answers in the frontier runs (run AFTER all jobs finish).
# For each tag: find empty task_ids -> build a subset -> re-run harness on just those ->
# merge the new non-empty answers back into traces.jsonl (replacing the empty rows). Cheap.
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
set -a; source /data/project/private/minstar/.env 2>/dev/null; set +a
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL
DATA=data/eval_samples/robust_core_eval.jsonl
LOG=results/retry_empties.log
: > "$LOG"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }

declare -A MODELS=(
  [api_gpt55_robust]="gpt-5.5"
  [api_opus47_robust]="openrouter/anthropic/claude-opus-4.7"
  [api_gemini3pro_robust]="gemini/gemini-3.1-pro-preview"
  [api_gpt55_notool_robust]="gpt-5.5"
)
for tag in "${!MODELS[@]}"; do
  model="${MODELS[$tag]}"
  tr="results/${tag}/traces.jsonl"
  [ -f "$tr" ] || { log "[skip] $tag no traces"; continue; }
  # empty/error task_ids
  python - "$tr" "$DATA" "results/${tag}/_retry_in.jsonl" <<'PY'
import json,sys
tr,data,out=sys.argv[1:4]
bad={json.loads(l)["task_id"] for l in open(tr)
     if not (json.loads(l).get("model_answer") or "").strip()
     or str(json.loads(l).get("model_answer","")).startswith("[ERROR]")}
rows=[json.loads(l) for l in open(data) if json.loads(l)["task_id"] in bad]
open(out,"w").writelines(json.dumps(r)+"\n" for r in rows)
print(len(rows))
PY
  nbad=$(wc -l < "results/${tag}/_retry_in.jsonl")
  [ "$nbad" -eq 0 ] && { log "[ok] $tag no empties"; continue; }
  log "[retry] $tag: $nbad empty -> re-running"
  extra=""; [ "$tag" = "api_gpt55_notool_robust" ] && extra="--no-tools"
  python harness/run.py --data "results/${tag}/_retry_in.jsonl" --model "litellm::$model" \
    --no-judge --workers 4 $extra --output "results/${tag}/_retry_out" >> "$LOG" 2>&1
  # merge: replace empty rows in traces.jsonl with the new non-empty answers
  python - "$tr" "results/${tag}/_retry_out/traces.jsonl" <<'PY'
import json,sys
tr,new=sys.argv[1:3]
fix={}
for l in open(new):
    r=json.loads(l); a=(r.get("model_answer") or "").strip()
    if a and not a.startswith("[ERROR]"): fix[r["task_id"]]=r
rows=[json.loads(l) for l in open(tr)]
n=0
for i,r in enumerate(rows):
    a=(r.get("model_answer") or "").strip()
    if (not a or a.startswith("[ERROR]")) and r["task_id"] in fix:
        rows[i]=fix[r["task_id"]]; n+=1
open(tr,"w").writelines(json.dumps(r)+"\n" for r in rows)
print(f"merged {n} fixed answers")
PY
  still=$(python - "$tr" <<'PY'
import json,sys
print(sum(1 for l in open(sys.argv[1]) if not (json.loads(l).get("model_answer") or "").strip()))
PY
)
  log "[done] $tag merged; remaining empty=$still"
done
echo "RETRY_EMPTIES_DONE" | tee -a "$LOG"
