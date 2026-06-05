#!/usr/bin/env python3
"""One-line live cost + progress for the Track-C run (used by the 15-min monitor)."""
import json, os, time
PRICE = {"gpt-5.5": (5e-6, 30e-6), "openrouter/anthropic/claude-opus-4.7": (5e-6, 25e-6),
         "gemini/gemini-3.1-pro-preview": (2e-6, 12e-6)}
TRACES = {"gpt55": ("api_gpt55_robust", "gpt-5.5"),
          "opus47": ("api_opus47_robust", "openrouter/anthropic/claude-opus-4.7"),
          "gem3pro": ("api_gemini3pro_robust", "gemini/gemini-3.1-pro-preview"),
          "gpt55_nt": ("api_gpt55_notool_robust", "gpt-5.5")}
N = 423
parts, total = [], 0.0
for label, (tag, model) in TRACES.items():
    f = f"results/{tag}/traces.jsonl"
    if not os.path.exists(f):
        parts.append(f"{label} 0/{N} $0"); continue
    pin = pout = n = empty = 0
    for l in open(f):
        r = json.loads(l); t = r.get("tokens", {})
        pin += t.get("prompt", 0); pout += t.get("completion", 0); n += 1
        if not (r.get("model_answer") or "").strip():
            empty += 1
    cin, cout = PRICE[model]
    c = pin * cin + pout * cout
    total += c
    em = f" EMPTY={empty}!" if empty else ""
    parts.append(f"{label} {n}/{N} ${c:.0f}{em}")
for label, jf in [("L2pop", "results/l2_fullpop.jsonl"), ("L2ft", "results/l2_fulltext.jsonl")]:
    if not os.path.exists(jf):
        parts.append(f"{label} 0 $0"); continue
    pin = pout = n = 0
    for l in open(jf):
        d = json.loads(l); pin += d.get("in_tok", 0); pout += d.get("out_tok", 0); n += 1
    cin, cout = PRICE["openrouter/anthropic/claude-opus-4.7"]
    c = pin * cin + pout * cout
    total += c
    parts.append(f"{label} {n} ${c:.0f}")
print(f"[{time.strftime('%H:%M')}] " + " | ".join(parts) + f"  ==> TOTAL ${total:.0f}/1000")
