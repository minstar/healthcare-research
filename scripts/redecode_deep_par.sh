#!/usr/bin/env bash
# Deep-failure stability, PARALLEL finish: GLM-5.1 already done (50/50). Generate Qwen3.6 and
# DeepSeek-V4 CONCURRENTLY on two NORMAL nodes (user-requested), then serve GLM-5.1 to judge all
# three deep trace sets and count flips. Resumable (qwen36 may already have partial traces).
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
export OPENAI_API_KEY=dummy EVAL_TEMPERATURE=0.0
DATA=data/eval_samples/redecode_deep_sample.jsonl
RUB=data/eval_samples/rubrics_1969_uid.jsonl
N=$(wc -l < "$DATA")
LOG=results/redecode_deep/run_par.log
mkdir -p results/redecode_deep
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }
jid_of(){ squeue -u minstar -h -n "$1" -o '%i' 2>/dev/null | head -1; }
traces(){ wc -l < "results/$1/traces.jsonl" 2>/dev/null || echo 0; }
DSV4=/data/project/private/minstar/settings/serving/deepseek_v4_flash_vllm.sh

# parallel generate one model: ensure serve up -> generate (resumable) -> scancel
gen() {  # tag jobname served serve-script
  local tag=$1 jn=$2 served=$3 script=$4 jid st node
  while [ "$(traces $tag)" -lt "$N" ]; do
    jid=$(jid_of "$jn")
    if [ -z "$jid" ]; then jid=$(sbatch --parsable "$script"); log "[$tag] submitted $jn -> $jid"; sleep 20; continue; fi
    st=$(squeue -j "$jid" -h -o '%T' 2>/dev/null)
    if [ "$st" = "RUNNING" ]; then
      node=$(squeue -j "$jid" -h -o '%N' 2>/dev/null)
      if [ -n "$node" ] && curl -s --max-time 4 "http://$node:8000/v1/models" 2>/dev/null | grep -q "$served"; then
        log "[$tag] endpoint up @ $node; generating (T=0, resume from $(traces $tag))"
        OPENAI_API_BASE=http://$node:8000/v1 python harness/run.py --data "$DATA" \
          --model "openai::http://$node:8000/v1::$served" --no-judge --workers 12 \
          --output results/$tag >> "$LOG" 2>&1
        continue
      fi
    fi
    sleep 45
  done
  jid=$(jid_of "$jn"); [ -n "$jid" ] && scancel "$jid" && log "[$tag] scancel $jn -> node freed ($(traces $tag) traces)"
}

log "=== PARALLEL generate Qwen3.6 + DeepSeek-V4 (GLM-5.1 already $(traces redecode_deep_glm51)/$N) ==="
gen redecode_deep_qwen36 serve-qwen36 qwen-3.6 serving/serve_qwen36.slurm &
gen redecode_deep_dsv4   serve-dsv4flash deepseek-v4-flash "$DSV4" &
wait
log "=== both gen done: qwen36=$(traces redecode_deep_qwen36) dsv4=$(traces redecode_deep_dsv4) ==="

# ---- judge phase: serve GLM-5.1, judge all three ----
log "=== judging deep-sample traces (GLM-5.1 judge) ==="
GN=""
while [ -z "$GN" ]; do
  jid=$(jid_of serve-glm51)
  [ -z "$jid" ] && { jid=$(sbatch --parsable serving/serve_glm51.slurm); log "submitted GLM judge $jid"; sleep 20; continue; }
  node=$(squeue -j "$jid" -h -o '%N' 2>/dev/null)
  [ "$(squeue -j "$jid" -h -o '%T')" = RUNNING ] && [ -n "$node" ] && \
    curl -s --max-time 4 "http://$node:8000/v1/models" 2>/dev/null | grep -q glm-5.1 && GN=$node || sleep 45
done
log "GLM-5.1 judge @ $GN"
cnt(){ python - "$1" <<'PY'
import json,sys
try: print(sum(1 for l in open(sys.argv[1]) if json.loads(l).get("checklist_score") is not None))
except FileNotFoundError: print(0)
PY
}
for tag in redecode_deep_glm51 redecode_deep_qwen36 redecode_deep_dsv4; do
  [ "$(cnt results/$tag/checklist_glm.jsonl)" -ge "$N" ] && { log "[judge] $tag already done"; continue; }
  log "[judge] $tag"
  python scripts/checklist_judge.py --traces results/$tag/traces.jsonl \
    --rubrics "$RUB" --judge openai/glm-5.1 --base "http://$GN:8000/v1" \
    --out results/$tag/checklist_glm.jsonl --workers 8 2>&1 | grep -ivE httpx >> "$LOG"
done
jid=$(jid_of serve-glm51); [ -n "$jid" ] && scancel "$jid" && log "scancel GLM-5.1 judge (deep judging done)"

# ---- flip analysis ----
conda activate base
python - <<'PY' 2>&1 | tee -a "$LOG"
import json
ids=set(json.load(open("results/redecode_deep_ids.json")))
def load(p):
    return {json.loads(l)["task_id"]:float(json.loads(l)["checklist_score"])
            for l in open(p) if json.loads(l).get("checklist_score") is not None}
S2={"GLM-5.1":load("results/redecode_deep_glm51/checklist_glm.jsonl"),
    "Qwen3.6":load("results/redecode_deep_qwen36/checklist_glm.jsonl"),
    "DeepSeek-V4":load("results/redecode_deep_dsv4/checklist_glm.jsonl")}
checked=[t for t in ids if all(t in S2[m] for m in S2)]
flips=[t for t in checked if any(S2[m][t]>=0.5 for m in S2)]
out={"deep_sample_n":len(checked),"flipped_to_pass":len(flips),
     "flip_rate_pct":round(100*len(flips)/len(checked),1) if checked else None,"flipped_ids":flips}
json.dump(out,open("results/redecode_deep_stability.json","w"),indent=2)
print(json.dumps(out,indent=2))
print(f"\n=> {len(flips)}/{len(checked)} riskiest deep-failures flipped to pass; "
      f"assumption {'HOLDS' if len(flips)<=2 else 'NEEDS REVISION'}")
PY
echo "REDECODE_DEEP_DONE" | tee -a "$LOG"
