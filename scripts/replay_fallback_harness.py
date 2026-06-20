#!/usr/bin/env python3
"""Replay-with-live-fallback tool-call harness for the OpenBioRQ benchmark.

WHAT THIS IS (honest framing)
-----------------------------
This module deterministically REPLAYS the recorded tool responses in
``results/release/tool_response_cache.jsonl`` and FALLS BACK to a live tool
call for any query that is not in the cache.

It is NOT a "frozen environment that lets anyone score a brand-new model fully
offline." A prior measurement found the cache has <1% cross-model query overlap,
so a new model replaying against it gets ~99% cache-MISS and must fall through
to live tools. What the cache DOES give you for free:

  * Full coverage of the 6 audited (model, task_id, tool, args) runs, so those
    runs can be deterministically RE-GRADED offline (no live API drift).
  * A pluggable ``live_call`` so a new model's uncached queries are served by a
    real MCP/tool backend you supply (this module never calls one itself).

CACHE SCHEMA (from scripts/build_release_artifacts.py)
------------------------------------------------------
One JSON record per (model, task_id):

    {"model": str,
     "task_id": str,
     "tool_calls": [{"round": int, "tool": str,
                     "arguments": dict, "result_preview": str}, ...]}

There is no explicit stored key. The KEY for a tool response is constructed here
from (model, task_id, tool, canonical(arguments)). ``round`` is intentionally
EXCLUDED: a tool response is content-addressed by its query, so the same query
issued in different rounds maps to the same cached response.

REPLAY-FIDELITY CAVEAT (do not over-claim)
------------------------------------------
``result_preview`` is the TRUNCATED preview the original harness logged (fixed
char budget per the paper's §4.2), NOT the full upstream API payload. Replaying
reproduces exactly what the agent *saw* in the truncated trace, but it is not a
byte-for-byte mirror of the live API. Re-grading that depends only on the
preview is faithful; anything needing the untruncated payload is not.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Callable, Optional, Tuple, Any, Dict


def canonical_args(arguments: Any) -> str:
    """Stable string form of a tool's arguments for content-addressing."""
    try:
        return json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return repr(arguments)


class ReplayFallbackToolClient:
    """Replay cached tool responses; fall back to a live call on a cache miss.

    Parameters
    ----------
    cache_path : str
        Path to tool_response_cache.jsonl.
    live_call : callable, optional
        ``live_call(model, task_id, tool_name, arguments) -> response``.
        Invoked on a cache miss. If None, a miss raises CacheMiss.
    scope : {'within-run', 'cross-model'}
        'within-run'  : key includes `model`; only a model's OWN cached calls
                        hit. Used to deterministically re-grade the 6 runs.
        'cross-model' : key excludes `model`; a new model may reuse ANY model's
                        cached response for the same (task_id, tool, args).
                        Used to MAXIMIZE reuse for a new model.
    """

    VALID_SCOPES = ("within-run", "cross-model")

    class CacheMiss(KeyError):
        pass

    def __init__(self, cache_path: str,
                 live_call: Optional[Callable[..., Any]] = None,
                 scope: str = "within-run"):
        if scope not in self.VALID_SCOPES:
            raise ValueError(f"scope must be one of {self.VALID_SCOPES}, got {scope!r}")
        self.cache_path = cache_path
        self.live_call = live_call
        self.scope = scope
        # _store[key] = result_preview ; key depends on scope
        self._store: Dict[Tuple, Any] = {}
        # bookkeeping for coverage reporting (scope-independent)
        self._records = 0
        self._tool_calls = 0
        self._models = Counter()
        self._load(cache_path)
        self.hits = 0
        self.misses = 0
        self.live_served = 0

    # ---- key construction ----
    def _key(self, model: str, task_id: str, tool_name: str, arguments: Any) -> Tuple:
        ak = canonical_args(arguments)
        if self.scope == "within-run":
            return (model, task_id, tool_name, ak)
        return (task_id, tool_name, ak)  # cross-model: model excluded

    def _load(self, path: str) -> None:
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            model = rec["model"]
            task_id = rec["task_id"]
            self._records += 1
            self._models[model] += 1
            for c in rec.get("tool_calls") or []:
                self._tool_calls += 1
                k = self._key(model, task_id, c.get("tool"), c.get("arguments"))
                # last-write-wins; same query -> same response, so harmless.
                self._store[k] = c.get("result_preview")

    # ---- main entry point ----
    def call(self, model: str, task_id: str, tool_name: str,
             arguments: Any) -> Tuple[Any, str]:
        """Return (response, source) with source in {'cache', 'live'}."""
        k = self._key(model, task_id, tool_name, arguments)
        if k in self._store:
            self.hits += 1
            return self._store[k], "cache"
        self.misses += 1
        if self.live_call is None:
            raise self.CacheMiss(
                f"cache miss, live tool required: model={model!r} task_id={task_id!r} "
                f"tool={tool_name!r} args={canonical_args(arguments)[:120]!r} "
                f"(scope={self.scope!r}). Provide live_call= to fall back to a real tool.")
        resp = self.live_call(model, task_id, tool_name, arguments)
        self.live_served += 1
        return resp, "live"

    # ---- introspection ----
    def stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        return {
            "scope": self.scope,
            "calls": total,
            "hits": self.hits,
            "misses": self.misses,
            "live_served": self.live_served,
            "hit_rate": (self.hits / total) if total else None,
            "miss_rate": (self.misses / total) if total else None,
            "cache_coverage": self.coverage(),
        }

    def coverage(self) -> Dict[str, Any]:
        return {
            "n_records": self._records,
            "n_tool_calls": self._tool_calls,
            "n_unique_keys": len(self._store),
            "scope_of_key": self.scope,
            "models": dict(self._models),
        }

    def reset_counters(self) -> None:
        self.hits = self.misses = self.live_served = 0


# ---------------------------------------------------------------------------
# SELF-TEST (no model, no API, no MCP) -- run: python scripts/replay_fallback_harness.py
# ---------------------------------------------------------------------------
def _iter_records(cache_path: str):
    for line in open(cache_path):
        line = line.strip()
        if line:
            yield json.loads(line)


def run_selftest(cache_path: str, out_path: Optional[str] = None) -> Dict[str, Any]:
    # roster of audited models present in the cache
    models = []
    for rec in _iter_records(cache_path):
        if rec["model"] not in models:
            models.append(rec["model"])

    # ---- 1) within-run: replay each model's OWN recorded calls -> expect ~100% ----
    wr_client = ReplayFallbackToolClient(cache_path, live_call=None, scope="within-run")
    within_run = {}
    for m in models:
        wr_client.reset_counters()
        for rec in _iter_records(cache_path):
            if rec["model"] != m:
                continue
            for c in rec.get("tool_calls") or []:
                resp, src = wr_client.call(m, rec["task_id"], c.get("tool"), c.get("arguments"))
                assert src == "cache", f"within-run unexpected miss for {m}"
        st = wr_client.stats()
        within_run[m] = {"calls": st["calls"], "hits": st["hits"],
                         "misses": st["misses"], "hit_rate": st["hit_rate"]}

    # ---- 2) cross-model: hit-rate a NEW model gets reusing OTHER models' caches ----
    # For each model A (standing in as the "new" model), build a cross-model cache
    # from the OTHER 5 models only, then replay A's distinct queries against it.
    # A live_call that just records the miss lets us count without raising.
    cross_model = {}
    for held_out in models:
        sub_path_keys = set()  # (task_id, tool, args) of the OTHER models
        a_queries = set()      # distinct (task_id, tool, args) issued by A
        for rec in _iter_records(cache_path):
            m = rec["model"]
            for c in rec.get("tool_calls") or []:
                key = (rec["task_id"], c.get("tool"), canonical_args(c.get("arguments")))
                if m == held_out:
                    a_queries.add(key)
                else:
                    sub_path_keys.add(key)
        if not a_queries:
            continue
        hit = sum(1 for q in a_queries if q in sub_path_keys)
        cross_model[held_out] = {
            "distinct_queries": len(a_queries),
            "found_in_others": hit,
            "hit_rate": hit / len(a_queries),
        }

    # aggregate cross-model (pooled over distinct queries)
    tot_q = sum(v["distinct_queries"] for v in cross_model.values())
    tot_h = sum(v["found_in_others"] for v in cross_model.values())
    cross_pooled = (tot_h / tot_q) if tot_q else None

    # global coverage
    cov = wr_client.coverage()
    # unique cross-model queries across the whole cache
    all_cm = set()
    for rec in _iter_records(cache_path):
        for c in rec.get("tool_calls") or []:
            all_cm.add((rec["task_id"], c.get("tool"), canonical_args(c.get("arguments"))))

    wr_rates = [v["hit_rate"] for v in within_run.values()]
    summary = {
        "cache_path": cache_path,
        "coverage": {
            "n_records": cov["n_records"],
            "n_tool_calls": cov["n_tool_calls"],
            "n_unique_within_run_keys": cov["n_unique_keys"],
            "n_unique_cross_model_queries": len(all_cm),
            "models": cov["models"],
        },
        "within_run": {
            "per_model": within_run,
            "min_hit_rate": min(wr_rates) if wr_rates else None,
            "all_100pct": all(abs(r - 1.0) < 1e-9 for r in wr_rates) if wr_rates else None,
            "note": "each model replays its OWN recorded calls -> deterministic re-grade",
        },
        "cross_model": {
            "per_model_held_out": cross_model,
            "pooled_hit_rate": cross_pooled,
            "note": ("hit-rate a NEW model would get reusing the OTHER 5 models' caches; "
                     "reproduces the measured <1% cross-model query overlap -> ~99% live fallback"),
        },
        "replay_fidelity_caveat": (
            "result_preview is TRUNCATED (fixed char budget); replay reproduces what the agent "
            "saw, not the full upstream API payload."),
        "honest_summary": (
            "ENABLES: deterministic offline RE-GRADE of the 6 audited runs (within-run hit ~100%). "
            "DOES NOT ENABLE: fully offline scoring of a new model (cross-model hit <1% -> ~99% live "
            "fallback via the pluggable live_call)."),
    }

    if out_path:
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import os
    ROOT = "/data/project/private/minstar/workspace/healthcare-research"
    cache = os.path.join(ROOT, "results/release/tool_response_cache.jsonl")
    out = os.path.join(ROOT, "results/replay_fallback_selftest.json")
    s = run_selftest(cache, out_path=out)

    print("== cache coverage ==")
    print(json.dumps(s["coverage"], indent=2))
    print("\n== within-run (re-grade the 6 audited runs) ==")
    for m, v in s["within_run"]["per_model"].items():
        print(f"  {m:14s} calls={v['calls']:6d} hit_rate={v['hit_rate']:.4f}")
    print(f"  all_100pct={s['within_run']['all_100pct']} min={s['within_run']['min_hit_rate']:.4f}")
    print("\n== cross-model (new model reusing others' caches) ==")
    for m, v in s["cross_model"]["per_model_held_out"].items():
        print(f"  {m:14s} distinct_q={v['distinct_queries']:5d} "
              f"found_in_others={v['found_in_others']:4d} hit_rate={v['hit_rate']:.4%}")
    print(f"  POOLED cross-model hit_rate = {s['cross_model']['pooled_hit_rate']:.4%}")
    print("\n== honest summary ==")
    print(" ", s["honest_summary"])
    print(f"\nwrote {out}")
