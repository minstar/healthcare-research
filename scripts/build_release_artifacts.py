#!/usr/bin/env python3
"""Build frozen release artifacts (Reviewer R3 M1): a versioned robust-core ID list and a
tool-response replay cache, so the benchmark is a REPRODUCIBLE FROZEN ARTIFACT rather than a
re-runnable live-API selection (which churns ~30-46% run-to-run, see deep-failure stability).

Outputs:
  results/release/robust_core_ids_v3.json   -- the 423 frozen task_ids + provenance metadata
  results/release/tool_response_cache.jsonl  -- per (model, task_id) ordered tool-call->result_preview
  results/release/tool_cache_manifest.json   -- counts + caveats (previews are truncated)
"""
import json, glob, os

ROOT = "/data/project/private/minstar/workspace/healthcare-research"
R = f"{ROOT}/results"
os.makedirs(f"{R}/release", exist_ok=True)

# ---- M1a: frozen robust-core ID list ----
rc = json.load(open(f"{R}/robust_core_ids.json"))
idlist = {
    "name": "OpenBioRQ robust core (frozen snapshot)",
    "corpus_version": "v3 base / 1969 gold slice",
    "n": len(rc),
    "as_of": "2026-06",
    "selection": "all three roster models (GLM-5.1, Qwen3.6, DeepSeek-V4) fail (checklist<0.5) at T=0",
    "grading_judge": "GLM-5.1 checklist judge + rubrics_1969_uid.jsonl",
    "reproducibility_note": ("This is the authors' single-T=0 snapshot. The live selection procedure "
        "is NOT reproducible run-to-run (~30-46% membership churn from live-API drift and erratic "
        "agentic tool use; see deep-failure stability test). Reproduce by REPLAYING the frozen "
        "tool_response_cache against this fixed ID list, not by re-running live selection."),
    "task_ids": sorted(rc),
}
json.dump(idlist, open(f"{R}/release/robust_core_ids_v3.json", "w"), indent=2)

# ---- M1b: tool-response replay cache ----
SRC = {
    "GLM-5.1": "baseline_glm51_tools", "Qwen3.6": "baseline_roster_qwen36_t0",
    "DeepSeek-V4": "baseline_roster_dsv4_t0", "Gemini-3-Pro": "api_gemini3pro_robust",
    "Opus-4.7": "api_opus47_robust", "GPT-5.5": "api_gpt55_robust",
}
n_records = 0
n_calls = 0
by_model = {}
with open(f"{R}/release/tool_response_cache.jsonl", "w") as out:
    for model, d in SRC.items():
        p = f"{R}/{d}/traces.jsonl"
        if not glob.glob(p):
            by_model[model] = "MISSING"; continue
        mc = 0; mcalls = 0
        for l in open(p):
            t = json.loads(l)
            calls = [{"round": c.get("round"), "tool": c.get("tool"),
                      "arguments": c.get("arguments"), "result_preview": c.get("result_preview")}
                     for c in (t.get("tool_calls") or [])]
            out.write(json.dumps({"model": model, "task_id": t["task_id"], "tool_calls": calls},
                                 ensure_ascii=False) + "\n")
            n_records += 1; mc += 1; mcalls += len(calls); n_calls += len(calls)
        by_model[model] = {"trace_records": mc, "tool_calls": mcalls}

manifest = {
    "description": "Frozen tool-response replay cache for OpenBioRQ agentic runs.",
    "records": n_records, "total_tool_calls": n_calls, "by_model": by_model,
    "schema": "{model, task_id, tool_calls:[{round, tool, arguments, result_preview}]}",
    "caveats": [
        "result_preview is the TRUNCATED preview the harness logged (fixed char budget per §4.2), "
        "not the full API payload; sufficient for inspecting/replaying what the agent actually saw, "
        "but not a full mirror of the upstream APIs.",
        "Roster traces are over the 657 core (filter task_id to the 423 in robust_core_ids_v3.json "
        "for the robust-core subset); frontier traces are over the 423 robust core.",
    ],
}
json.dump(manifest, open(f"{R}/release/tool_cache_manifest.json", "w"), indent=2)
print(f"M1a: robust_core_ids_v3.json ({len(rc)} ids)")
print(f"M1b: tool_response_cache.jsonl ({n_records} records, {n_calls} tool calls)")
print("by_model:", json.dumps(by_model))
