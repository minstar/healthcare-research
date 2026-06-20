"""Token accounting + EXACT per-item cost for all Track-C paid experiments.

Reads:
  - results/<tag>/traces.jsonl  (harness records tokens={'prompt','completion'}) -> #1 leaderboard, #4 no-tools
  - results/l2_*.jsonl          (each row carries in_tok/out_tok/judge_model)     -> #2 full-pop, #3 full-text judge
Applies verified per-token pricing. Output: results/api_cost.json + a table + a $1000 budget check.
"""
import json, os
import litellm

BUDGET_CAP = 1000.0  # user-set hard ceiling across ALL experiments

# tag -> litellm model id (must match the dirs run_frontier_robust.sh writes)
MODELS = {
    "api_gpt55_robust": "gpt-5.5",
    "api_opus47_robust": "openrouter/anthropic/claude-opus-4.7",
    "api_gemini3pro_robust": "gemini/gemini-3.1-pro-preview",
    "api_gpt55_notool_robust": "gpt-5.5",
}
# L2 judge output files (each row carries in_tok/out_tok/judge_model) -> Track-C #2/#3
JUDGE_FILES = ["results/l2_fullpop.jsonl", "results/l2_fulltext.jsonl"]

# verified pricing (USD per token) from provider/OpenRouter listings 2026-06.
PRICE_OVERRIDE = {
    "gpt-5.5": (5e-6, 30e-6),
    "openrouter/anthropic/claude-opus-4.7": (5e-6, 25e-6),
    "openrouter/anthropic/claude-haiku-4.5": (1e-6, 5e-6),
    "gemini/gemini-3.1-pro-preview": (2e-6, 12e-6),
}


def price(model):
    if model in PRICE_OVERRIDE:
        return PRICE_OVERRIDE[model]
    key = model.split("/")[-1]
    c = litellm.model_cost.get(model) or litellm.model_cost.get(key) or {}
    return c.get("input_cost_per_token", 0.0), c.get("output_cost_per_token", 0.0)


rows, total = [], 0.0

# --- #1 / #4 agentic trace costs ---
for tag, model in MODELS.items():
    f = f"results/{tag}/traces.jsonl"
    if not os.path.exists(f):
        rows.append({"item": tag, "model": model, "status": "not run"}); continue
    pin = pout = n = 0
    for l in open(f):
        t = json.loads(l).get("tokens", {})
        pin += t.get("prompt", 0); pout += t.get("completion", 0); n += 1
    cin, cout = price(model)
    cost = pin * cin + pout * cout
    total += cost
    rows.append({"item": tag, "model": model, "n": n, "input_tokens": pin, "output_tokens": pout,
                 "total_cost_usd": round(cost, 2)})

# --- #2 / #3 judge costs (tokens stored per row) ---
for jf in JUDGE_FILES:
    if not os.path.exists(jf):
        rows.append({"item": os.path.basename(jf), "model": "judge", "status": "not run"}); continue
    pin = pout = n = 0; jm = "?"
    for l in open(jf):
        d = json.loads(l)
        pin += d.get("in_tok", 0); pout += d.get("out_tok", 0); n += 1
        jm = d.get("judge_model", jm)
    cin, cout = price(jm)
    cost = pin * cin + pout * cout
    total += cost
    rows.append({"item": os.path.basename(jf), "model": jm, "n": n,
                 "input_tokens": pin, "output_tokens": pout, "total_cost_usd": round(cost, 2)})

out = {"per_item": rows, "grand_total_usd": round(total, 2),
       "budget_cap_usd": BUDGET_CAP, "under_budget": total <= BUDGET_CAP}
json.dump(out, open("results/api_cost.json", "w"), indent=2)
print('\n=== TRACK-C TOKEN & COST (per item) ===')
print(f'{"item":<28}{"model":<40}{"n":>6}{"in_tok":>13}{"out_tok":>12}{"cost$":>9}')
for r in rows:
    if r.get("status"):
        print(f'{r["item"]:<28}{r["model"]:<40}{"":>6}{"":>13}{"":>12}  {r["status"]}'); continue
    print(f'{r["item"]:<28}{r["model"]:<40}{r["n"]:>6}{r["input_tokens"]:>13,}{r["output_tokens"]:>12,}{r["total_cost_usd"]:>9.2f}')
print(f'{"GRAND TOTAL":<74}{"":>12}{total:>9.2f}   (cap ${BUDGET_CAP:.0f}, {"OK" if total<=BUDGET_CAP else "OVER!!"})')
print("-> results/api_cost.json")
