"""B-1 deterministic cleanup → clean benchmark v2 (no LLM).

Applies the auditable, non-judgment fixes surfaced by audit_contamination.py:
  1. open_status vocabulary normalization (11 variants → 4 canonical)
  2. REMOVE synthetic templates + dead trials (terminated/withdrawn/suspended)
  3. RELABEL trials that are completed-with-results but labeled "open"
  4. Attach provenance: status_method, audit_flag, n_followups, src_year

Nothing is deleted in place: removed items are quarantined to a separate file with
reasons, and a changelog records every action. Stage-2 LLM re-judgment (the 78% with
follow-up literature) is handled separately.

Usage:
    python scripts/cleanup_v2.py \
        --data data/expanded/all_questions_combined.jsonl \
        --audit data/audit_full/audit_per_item.jsonl \
        --out-dir data/export
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

# 11 observed status strings → 4 canonical labels
CANONICAL = {
    "open": "open",
    "partially_open": "open",
    "partially_answered": "partially_answered",
    "partially_resolved": "partially_answered",
    "partially resolved": "partially_answered",
    "partially_addressed": "partially_answered",
    "mostly_resolved": "partially_answered",
    "largely_resolved": "partially_answered",
    "answered": "answered",
    "resolved": "answered",
    "closed": "answered",
}


def normalize_status(s: str) -> str:
    return CANONICAL.get((s or "").strip().lower(), "unknown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/expanded/all_questions_combined.jsonl")
    ap.add_argument("--audit", default="data/audit_full/audit_per_item.jsonl")
    ap.add_argument("--out-dir", default="data/export")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data)]
    audit = {}
    for l in open(args.audit):
        a = json.loads(l)
        audit[a["source_id"]] = a  # source_id may repeat; per-source signal is identical

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kept, removed = [], []
    changelog = collections.Counter()
    label_changes = collections.Counter()

    for r in rows:
        sid = str(r.get("source_id", ""))
        a = audit.get(sid, {})
        flag = a.get("flag", "")
        before = r.get("open_status", "")

        # --- attach provenance ---
        r["status_method"] = a.get("method", "none")
        r["audit_flag"] = flag or "ok"
        r["n_followups"] = a.get("n_followups", 0)
        r["src_year"] = a.get("src_year", "")

        # --- 1. normalize status vocabulary ---
        norm = normalize_status(before)
        if norm != before:
            label_changes[f"{before} -> {norm}"] += 1
        r["open_status"] = norm

        # --- 2. REMOVE (synthetic templates, dead trials) ---
        if flag == "synthetic_template":
            r["_removed_reason"] = "synthetic_template (KEGG/UniProt-derived, not literature-extracted)"
            removed.append(r); changelog["removed_synthetic"] += 1; continue
        if flag == "trial_dead":
            r["_removed_reason"] = f"dead_trial ({a.get('detail','')})"
            removed.append(r); changelog["removed_dead_trial"] += 1; continue

        # --- 3. RELABEL trial completed+results but "open" ---
        if flag == "trial_resolved_but_open":
            r["open_status"] = "partially_answered"
            r["status_reasoning"] = ("[auto] Trial COMPLETED with results posted on "
                                     "ClinicalTrials.gov; downgraded from 'open'. " + r.get("status_reasoning", ""))
            changelog["relabeled_trial_resolved"] += 1

        # mark (do not remove) completed-trial open-framing for Stage-2 review
        if flag == "trial_completed":
            changelog["flagged_trial_completed_review"] += 1

        kept.append(r)

    # --- write outputs ---
    v2_path = out_dir / "mcp_benchmark_v2.jsonl"
    q_path = out_dir / "quarantine_v2.jsonl"
    with v2_path.open("w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with q_path.open("w") as f:
        for r in removed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # final status distribution
    final_status = collections.Counter(r["open_status"] for r in kept)
    method_cov = collections.Counter(r["status_method"] for r in kept)

    report = {
        "input": len(rows),
        "kept": len(kept),
        "quarantined": len(removed),
        "actions": dict(changelog),
        "label_normalizations": dict(label_changes),
        "final_status_distribution": dict(final_status),
        "verification_method_coverage": dict(method_cov),
        "stage2_needed": sum(1 for r in kept if r["audit_flag"] in
                             ("heavy_followup_recheck", "trial_completed", "arxiv_citation_check_needed")),
    }
    with (out_dir / "cleanup_v2_report.json").open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {v2_path} ({len(kept)}) + {q_path} ({len(removed)})")


if __name__ == "__main__":
    main()
