#!/usr/bin/env bash
# 10-question dry-run per frontier model -> exact token usage -> $/question projection.
# No judge (cost calibration only). Runs the 3 models in parallel.
set -u
cd /data/project/private/minstar/workspace/healthcare-research
source /data/project/private/minstar/miniconda3/etc/profile.d/conda.sh
conda activate kimi
set -a; source /data/project/private/minstar/.env 2>/dev/null; set +a
unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN ANTHROPIC_MODEL
DATA=data/eval_samples/dryrun10.jsonl
LOG=results/dryrun_api.log
: > "$LOG"

run(){ local tag=$1 model=$2
  echo "$(date '+%T') [dry] $tag ($model)" | tee -a "$LOG"
  python harness/run.py --data "$DATA" --model "litellm::$model" --no-judge \
    --workers 5 --output "results/dryrun_${tag}" >> "$LOG" 2>&1
  echo "$(date '+%T') [done] $tag" | tee -a "$LOG"
}

run dry_gpt55       "gpt-5.5" &
run dry_opus47      "openrouter/anthropic/claude-opus-4.7" &
run dry_gemini31pro "gemini/gemini-3.1-pro-preview" &
wait
echo "DRYRUN_DONE" | tee -a "$LOG"
