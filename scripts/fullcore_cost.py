#!/usr/bin/env python3
"""Incremental cost + progress for the Table-3 full-core(657) fill run.

Seeded dirs already contain the paid frozen-423 traces. This computes the cost of
ONLY the NEW rows (current total minus a frozen baseline snapshot), so the number
reported is the genuinely-new spend for filling the 4 empty Full-core(657) cells.

Usage:
  python scripts/fullcore_cost.py --snapshot   # record baseline (run ONCE before launch)
  python scripts/fullcore_cost.py              # print one-line incremental status
  python scripts/fullcore_cost.py --cap 450    # exit 2 if incremental TOTAL exceeds cap

Exit codes: 0 ok | 2 over cap (watchdog kills the run on this).
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_FULL = 657
BASELINE = os.path.join(ROOT, "results/frontier_fullcore.baseline.json")

# OpenRouter pass-through pricing, live-verified 2026-06-19 (in $/tok, out $/tok)
PRICE = {
    "gpt55":    (5e-6, 30e-6),
    "opus47":   (5e-6, 25e-6),
    "gem3pro":  (2e-6, 12e-6),
    "gpt55_nt": (5e-6, 30e-6),
}
DIRS = {
    "gpt55":    "api_gpt55_fullcore",
    "opus47":   "api_opus47_fullcore",
    "gem3pro":  "api_gemini3pro_fullcore",
    "gpt55_nt": "api_gpt55_notool_fullcore",
}


def dir_cost(label):
    """Return (n_rows, cost_usd, n_empty) for a fullcore traces.jsonl."""
    f = os.path.join(ROOT, "results", DIRS[label], "traces.jsonl")
    if not os.path.exists(f):
        return 0, 0.0, 0
    cin, cout = PRICE[label]
    pin = pout = n = empty = 0
    for l in open(f):
        try:
            r = json.loads(l)
        except Exception:
            continue
        t = r.get("tokens", {})
        pin += t.get("prompt", 0); pout += t.get("completion", 0); n += 1
        ans = r.get("model_answer", "")
        if (not str(ans).strip()) or str(ans).startswith("[ERROR]"):
            empty += 1
    return n, pin * cin + pout * cout, empty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true", help="record current cost as the frozen baseline")
    ap.add_argument("--cap", type=float, default=None, help="exit 2 if incremental total exceeds this USD cap")
    args = ap.parse_args()

    if args.snapshot:
        base = {lab: dir_cost(lab)[1] for lab in DIRS}
        base["_n_rows"] = {lab: dir_cost(lab)[0] for lab in DIRS}
        json.dump(base, open(BASELINE, "w"), indent=2)
        print("baseline snapshot ->", BASELINE)
        for lab in DIRS:
            print(f"  {lab:<9} rows={base['_n_rows'][lab]:>4}  baseline=${base[lab]:.2f}")
        return 0

    base = json.load(open(BASELINE)) if os.path.exists(BASELINE) else {}
    parts, inc_total, cur_total, empty_total = [], 0.0, 0.0, 0
    for lab in DIRS:
        n, cost, empty = dir_cost(lab)
        b = float(base.get(lab, cost))   # if no baseline, incremental 0
        inc = max(0.0, cost - b)
        inc_total += inc; cur_total += cost; empty_total += empty
        em = f" ERR={empty}" if empty else ""
        parts.append(f"{lab} {n}/{N_FULL} +${inc:.0f}{em}")
    cap = args.cap
    capstr = f"/{cap:.0f}" if cap else ""
    print(f"[{time.strftime('%H:%M')}] " + " | ".join(parts) +
          f"  ==> NEW ${inc_total:.0f}{capstr} (cur total ${cur_total:.0f}, ERR={empty_total})")
    if cap is not None and inc_total > cap:
        print(f"!!! OVER CAP: new spend ${inc_total:.2f} > ${cap:.2f} !!!")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
