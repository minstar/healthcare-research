#!/usr/bin/env bash
# Cost watchdog for the Table-3 full-core fill. Polls incremental (NEW) spend every 90s
# via fullcore_cost.py --cap. If new spend exceeds CAP it KILLS the gen processes
# (defense-in-depth against the documented agentic-eval overrun). Exits 0 cleanly when
# the money script logs FRONTIER_FULLCORE_GEN_DONE. READ + KILL only; spends nothing.
set -u
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "${CONDA_ENV:-kimi}"
CAP=${1:-450}
LOG=results/frontier_fullcore.log
PIDFILE=results/frontier_fullcore.pids
DATA=data/eval_samples/core_eval.jsonl

kill_run(){
  echo "WATCHDOG KILL: NEW spend exceeded \$$CAP — terminating gen processes"
  [ -f "$PIDFILE" ] && awk '{print $1}' "$PIDFILE" | xargs -r kill 2>/dev/null
  pkill -f "harness/run.py --data $DATA" 2>/dev/null
}

echo "watchdog armed: cap=\$$CAP poll=90s"
while true; do
  out=$(python scripts/fullcore_cost.py --cap "$CAP" 2>/dev/null); rc=$?
  echo "$out"
  if [ "$rc" -eq 2 ]; then kill_run; echo "WATCHDOG_KILLED"; exit 1; fi
  if grep -q FRONTIER_FULLCORE_GEN_DONE "$LOG" 2>/dev/null; then echo "WATCHDOG: gen complete, exiting clean"; exit 0; fi
  sleep 90
done
