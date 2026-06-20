"""Pipeline orchestrator: schedule → run → verify → auto-update plan.

A lightweight DAG runner for the benchmark pipeline. It:
  - resolves stage dependencies (topological) and runs only READY stages,
  - runs each stage's declarative verify checks after it completes,
  - auto-edits state.json (machine state) and PLAN.md (human-readable plan),
  - is idempotent: a stage whose outputs exist and verify-pass is skipped,
  - is safe to schedule: `cost: high` stages are skipped by the watch loop and
    by `run` unless explicitly requested (--only ID or --include-costly).

Commands:
  status                 show the DAG + current state
  run [--only ID] [--include-costly] [--force] [--dry-run]
                         run ready stages to completion
  verify [ID]            re-run verification for a stage (or all)
  plan                   regenerate PLAN.md from current state
  watch --interval SEC   loop: pick up newly-ready (non-costly) stages

Schedule it (auto experiment scheduling) e.g. via cron / the /schedule skill:
  */30 * * * *  cd <repo> && python3 orchestration/orchestrator.py run >> orchestration/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ORCH = ROOT / "orchestration"
PLAN_YAML = ORCH / "pipeline.yaml"
STATE_JSON = ORCH / "state.json"
PLAN_MD = ORCH / "PLAN.md"
RUNS = ORCH / "runs"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------- verification --------------------------------
def _iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def compute_stat(stat: str, path: Path) -> dict:
    """Return variables a verify `assert` expression can reference."""
    if stat == "line_count":
        return {"total": sum(1 for _ in path.open())}
    if stat == "taxonomy_filled":
        total = filled = 0
        for r in _iter_jsonl(path):
            total += 1
            if r.get("taxonomy_l1", "Other") != "Other" or r.get("taxonomy_l2"):
                filled += 1
        return {"total": total, "filled": filled}
    if stat == "status_vocab":
        vocab = {r.get("open_status") for r in _iter_jsonl(path)}
        return {"n_distinct": len(vocab), "vocab": sorted(map(str, vocab))}
    if stat == "stage2_guardrail":
        hallucinated = sum(1 for r in _iter_jsonl(path)
                           if str(r.get("guardrail", "")).startswith("hallucinated"))
        return {"hallucinated": hallucinated}
    raise ValueError(f"unknown stat: {stat}")


def run_check(check: dict) -> tuple[bool, str]:
    kind = check["check"]
    path = ROOT / check["path"] if "path" in check else None
    if kind == "file_exists":
        return (path.exists(), f"exists={path.exists()}")
    if kind == "min_lines":
        if not path.exists():
            return (False, "missing file")
        n = sum(1 for _ in path.open())
        return (n >= check["n"], f"{n} lines (need >= {check['n']})")
    if kind == "jsonl_keys":
        if not path.exists():
            return (False, "missing file")
        first = next(_iter_jsonl(path), {})
        missing = [k for k in check["keys"] if k not in first]
        return (not missing, "ok" if not missing else f"missing keys {missing}")
    if kind == "metric":
        if not path.exists():
            return (False, "missing file")
        vars_ = compute_stat(check["stat"], path)
        try:
            ok = bool(eval(check["assert"], {"__builtins__": {}}, vars_))  # noqa: S307
        except Exception as e:
            return (False, f"assert error: {e}")
        return (ok, f"{check['assert']} | {vars_}")
    return (False, f"unknown check {kind}")


def verify_stage(stage: dict) -> tuple[bool, list[dict]]:
    results = []
    for chk in stage.get("verify", []):
        ok, detail = run_check(chk)
        results.append({"check": chk["check"], "ok": ok, "detail": detail})
    return (all(r["ok"] for r in results) if results else True, results)


# ------------------------------- state -------------------------------------
def load_plan() -> dict:
    return yaml.safe_load(PLAN_YAML.read_text())


def load_state() -> dict:
    if STATE_JSON.exists():
        return json.loads(STATE_JSON.read_text())
    return {"stages": {}, "updated": None}


def save_state(state: dict):
    state["updated"] = now()
    STATE_JSON.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    write_plan_md(load_plan(), state)


STATUS_ICON = {"verified": "✅", "done": "☑️", "failed": "❌", "running": "🔄",
               "pending": "⬜", "blocked": "🔒", "skipped": "⏭️"}


def write_plan_md(plan: dict, state: dict):
    m = plan["meta"]
    lines = [f"# Pipeline plan — {m['name']}", "", f"_{m['description']}_", "",
             f"Auto-generated by orchestrator · updated {state.get('updated') or now()}", "",
             "| | Stage | Status | Cost | Deps | Last result |",
             "|--|-------|--------|------|------|-------------|"]
    for st in plan["stages"]:
        s = state["stages"].get(st["id"], {})
        status = s.get("status", "pending")
        icon = STATUS_ICON.get(status, "⬜")
        vr = s.get("verify", [])
        vsum = "—"
        if vr:
            passed = sum(1 for v in vr if v["ok"])
            vsum = f"{passed}/{len(vr)} checks" + ("" if passed == len(vr) else " ⚠️")
        deps = ", ".join(st.get("deps", [])) or "—"
        lines.append(f"| {icon} | `{st['id']}` | {status} | {st.get('cost','?')} | {deps} | {vsum} |")
    lines += ["", "## Stage detail", ""]
    for st in plan["stages"]:
        s = state["stages"].get(st["id"], {})
        lines.append(f"### `{st['id']}` — {st.get('desc','')}")
        lines.append(f"- status: **{s.get('status','pending')}**  ·  cost: {st.get('cost','?')}"
                     f"  ·  ended: {s.get('ended','—')}")
        for v in s.get("verify", []):
            lines.append(f"  - {'✓' if v['ok'] else '✗'} {v['check']}: {v['detail']}")
        lines.append("")
    PLAN_MD.write_text("\n".join(lines))


# ------------------------------ scheduling ---------------------------------
def deps_ok(stage: dict, state: dict) -> bool:
    return all(state["stages"].get(d, {}).get("status") in ("verified", "done")
               for d in stage.get("deps", []))


def stage_status(stage: dict, state: dict) -> str:
    """Reconcile recorded status with reality (outputs may already exist)."""
    sid = stage["id"]
    recorded = state["stages"].get(sid, {}).get("status")
    if recorded in ("running", "failed"):
        return recorded
    ok, _ = verify_stage(stage)
    outs = [ROOT / o for o in stage.get("outputs", [])]
    if outs_exist(outs) and ok:
        return "verified"
    if not deps_ok(stage, state):
        return "blocked"
    return "pending"


def outs_exist(outs: list[Path]) -> bool:
    return bool(outs) and all(o.exists() for o in outs)


def reconcile(plan: dict, state: dict):
    for st in plan["stages"]:
        sid = st["id"]
        cur = state["stages"].setdefault(sid, {})
        if cur.get("status") in ("running", "failed"):
            continue
        new = stage_status(st, state)
        if new == "verified" and cur.get("status") != "verified":
            ok, results = verify_stage(st)
            cur.update(status="verified", verify=results, ended=cur.get("ended") or now())
        else:
            cur["status"] = new


# ------------------------------- execution ---------------------------------
def execute(stage: dict, state: dict) -> bool:
    sid = stage["id"]
    RUNS.mkdir(parents=True, exist_ok=True)
    log_path = RUNS / f"{sid}.log"
    state["stages"][sid] = {"status": "running", "started": now(), "cmd": stage["cmd"]}
    save_state(state)
    print(f"  ▶ running {sid}: {stage['cmd']}")
    with log_path.open("w") as log:
        proc = subprocess.run(stage["cmd"], shell=True, cwd=ROOT, stdout=log,
                              stderr=subprocess.STDOUT)
    rc = proc.returncode
    ok, results = verify_stage(stage)
    entry = {"cmd": stage["cmd"], "started": state["stages"][sid]["started"],
             "ended": now(), "returncode": rc, "verify": results, "log": str(log_path)}
    if rc == 0 and ok:
        entry["status"] = "verified"
        print(f"  ✅ {sid} verified")
    else:
        entry["status"] = "failed"
        why = f"rc={rc}" if rc else "verify failed: " + "; ".join(
            f"{r['check']}({r['detail']})" for r in results if not r["ok"])
        entry["error"] = why
        print(f"  ❌ {sid} failed — {why}  (log: {log_path})")
    state["stages"][sid] = entry
    save_state(state)
    return entry["status"] == "verified"


def ready_stages(plan: dict, state: dict, include_costly: bool, force: bool) -> list[dict]:
    out = []
    for st in plan["stages"]:
        s = state["stages"].get(st["id"], {})
        if s.get("status") == "verified" and not force:
            continue
        if not deps_ok(st, state):
            continue
        if st.get("cost") == "high" and not include_costly:
            continue
        out.append(st)
    return out


# -------------------------------- CLI --------------------------------------
def cmd_status(plan, state):
    reconcile(plan, state)
    save_state(state)
    for st in plan["stages"]:
        s = state["stages"].get(st["id"], {})
        ic = STATUS_ICON.get(s.get("status", "pending"), "⬜")
        print(f"  {ic} {st['id']:20s} {s.get('status','pending'):9s} "
              f"cost={st.get('cost','?'):6s} deps={','.join(st.get('deps',[])) or '-'}")
    print(f"\nPLAN.md + state.json updated. costly stages run only with --only/--include-costly.")


def cmd_run(plan, state, args):
    reconcile(plan, state)
    save_state(state)
    if args.only:
        targets = [s for s in plan["stages"] if s["id"] == args.only]
        if not targets:
            sys.exit(f"unknown stage {args.only}")
        if not deps_ok(targets[0], state) and not args.force:
            sys.exit(f"{args.only} blocked: deps not satisfied")
    else:
        targets = None
    ran = 0
    while True:
        batch = (targets if targets is not None
                 else ready_stages(plan, state, args.include_costly, args.force))
        batch = [s for s in batch if not (state["stages"].get(s["id"], {}).get("status")
                                           == "verified" and not args.force)]
        if not batch:
            break
        for st in batch:
            if args.dry_run:
                print(f"  [dry-run] would run {st['id']} (cost={st.get('cost')})")
                state["stages"].setdefault(st["id"], {})["status"] = "verified"  # pretend
            else:
                execute(st, state)
            ran += 1
        if targets is not None or args.dry_run:
            break
    print(f"\n{'[dry-run] ' if args.dry_run else ''}{ran} stage(s) processed.")
    if not args.dry_run:
        reconcile(plan, state); save_state(state)


def cmd_watch(plan, state, args):
    print(f"watch: every {args.interval}s; runs ready non-costly stages. Ctrl-C to stop.")
    while True:
        reconcile(plan, state); save_state(state)
        batch = ready_stages(plan, state, include_costly=False, force=False)
        if batch:
            print(f"[{now()}] {len(batch)} ready: {[s['id'] for s in batch]}")
            for st in batch:
                execute(st, state)
        else:
            print(f"[{now()}] nothing ready.")
        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    pr = sub.add_parser("run")
    pr.add_argument("--only"); pr.add_argument("--include-costly", action="store_true")
    pr.add_argument("--force", action="store_true"); pr.add_argument("--dry-run", action="store_true")
    pv = sub.add_parser("verify"); pv.add_argument("stage", nargs="?")
    sub.add_parser("plan")
    pw = sub.add_parser("watch"); pw.add_argument("--interval", type=int, default=1800)
    args = ap.parse_args()

    plan, state = load_plan(), load_state()
    if args.cmd == "status":
        cmd_status(plan, state)
    elif args.cmd == "run":
        cmd_run(plan, state, args)
    elif args.cmd == "watch":
        cmd_watch(plan, state, args)
    elif args.cmd == "plan":
        reconcile(plan, state); save_state(state); print(f"wrote {PLAN_MD}")
    elif args.cmd == "verify":
        for st in plan["stages"]:
            if args.stage and st["id"] != args.stage:
                continue
            ok, results = verify_stage(st)
            print(f"  {'✅' if ok else '❌'} {st['id']}")
            for r in results:
                print(f"      {'✓' if r['ok'] else '✗'} {r['check']}: {r['detail']}")


if __name__ == "__main__":
    main()
