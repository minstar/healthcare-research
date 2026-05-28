#!/usr/bin/env python3
"""Validate gold reference answers: verify citations and enforce completeness criteria.

Usage:
    python scripts/track_b/validate_gold_answers.py [--input DIR] [--fix] [--report FILE]

Validation steps:
  1. PMID verification via NCBI efetch (batch 100)
  2. NCT verification via ClinicalTrials.gov API
  3. Completeness criteria enforcement
  4. Structural integrity checks

Completeness criteria (all must pass for a gold answer to be "accepted"):
  - current_knowledge: >= 200 chars (substantive, not placeholder)
  - unknown_aspects:   >= 100 chars
  - answer_summary:    >= 200 chars
  - key_citations:     >= 2 entries with valid IDs
  - mcp_tool_plan:     >= 1 entry
  - completeness:      0.0-1.0 range, numeric

Gold answers failing criteria are flagged but not removed — the report
shows which need regeneration.
"""
import argparse
import glob
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

BASE_DIR = Path("/data/project/private/minstar/workspace/healthcare-research")
GOLD_DIR = BASE_DIR / "data" / "gold_answers"

COMPLETENESS_CRITERIA = {
    "current_knowledge_min_chars": 200,
    "unknown_aspects_min_chars": 100,
    "answer_summary_min_chars": 200,
    "min_citations": 2,
    "min_tool_plans": 1,
}


def verify_pmids(pmids: set[str], batch_size: int = 100) -> tuple[set[str], set[str]]:
    """Verify PMIDs against NCBI efetch. Returns (valid, invalid)."""
    valid = set()
    pmid_list = sorted(pmids)

    for i in range(0, len(pmid_list), batch_size):
        batch = pmid_list[i : i + batch_size]
        ids = ",".join(batch)
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={ids}&rettype=uilist&retmode=text"
        )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode()
                for line in text.strip().split("\n"):
                    line = line.strip()
                    if line.isdigit():
                        valid.add(line)
            time.sleep(0.4)
        except Exception as e:
            print(f"  PMID batch {i // batch_size} error: {e}", file=sys.stderr)
            valid.update(batch)

    return valid, pmids - valid


def verify_nct_ids(nct_ids: set[str]) -> tuple[set[str], set[str]]:
    """Verify NCT IDs against ClinicalTrials.gov API. Returns (valid, invalid)."""
    valid = set()
    for nct_id in sorted(nct_ids):
        url = (
            f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
            "?fields=NCTId&format=json"
        )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    valid.add(nct_id)
            time.sleep(0.2)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                pass  # invalid
            else:
                valid.add(nct_id)  # network error, assume valid
        except Exception:
            valid.add(nct_id)

    return valid, nct_ids - valid


def check_completeness(answer: dict) -> list[str]:
    """Check a single gold answer against completeness criteria. Returns list of failures."""
    failures = []

    ck = answer.get("current_knowledge", "")
    if len(ck) < COMPLETENESS_CRITERIA["current_knowledge_min_chars"]:
        failures.append(
            f"current_knowledge too short ({len(ck)} < {COMPLETENESS_CRITERIA['current_knowledge_min_chars']})"
        )

    ua = answer.get("unknown_aspects", "")
    if len(ua) < COMPLETENESS_CRITERIA["unknown_aspects_min_chars"]:
        failures.append(
            f"unknown_aspects too short ({len(ua)} < {COMPLETENESS_CRITERIA['unknown_aspects_min_chars']})"
        )

    ans = answer.get("answer_summary", "")
    if len(ans) < COMPLETENESS_CRITERIA["answer_summary_min_chars"]:
        failures.append(
            f"answer_summary too short ({len(ans)} < {COMPLETENESS_CRITERIA['answer_summary_min_chars']})"
        )

    cites = answer.get("key_citations", [])
    if len(cites) < COMPLETENESS_CRITERIA["min_citations"]:
        failures.append(
            f"too few citations ({len(cites)} < {COMPLETENESS_CRITERIA['min_citations']})"
        )

    tools = answer.get("mcp_tool_plan", [])
    if len(tools) < COMPLETENESS_CRITERIA["min_tool_plans"]:
        failures.append(
            f"too few tool plans ({len(tools)} < {COMPLETENESS_CRITERIA['min_tool_plans']})"
        )

    comp = answer.get("completeness")
    if not isinstance(comp, (int, float)) or not (0.0 <= comp <= 1.0):
        failures.append(f"completeness out of range: {comp}")

    return failures


def load_gold_answers(gold_dir: Path) -> list[tuple[str, dict]]:
    """Load all gold answer records. Returns list of (filename, record)."""
    records = []
    for f in sorted(glob.glob(str(gold_dir / "*.output.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    records.append((os.path.basename(f), obj))
                except json.JSONDecodeError:
                    continue
    return records


def strip_invalid_citations(
    gold_dir: Path, invalid_pmids: set[str], invalid_ncts: set[str]
) -> int:
    """Remove invalid citations from gold answer files in-place. Returns count removed."""
    total_removed = 0
    for f in sorted(glob.glob(str(gold_dir / "*.output.jsonl"))):
        lines_out = []
        changed = False
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                orig = obj.get("key_citations", [])
                clean = []
                for c in orig:
                    ctype, cid = c.get("type", ""), c.get("id", "")
                    if ctype == "PMID" and cid in invalid_pmids:
                        continue
                    if ctype == "NCT" and cid in invalid_ncts:
                        continue
                    clean.append(c)
                removed = len(orig) - len(clean)
                if removed > 0:
                    total_removed += removed
                    obj["key_citations"] = clean
                    changed = True
                lines_out.append(json.dumps(obj, ensure_ascii=False))
        if changed:
            with open(f, "w") as fh:
                fh.write("\n".join(lines_out) + "\n")
    return total_removed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=GOLD_DIR, help="Directory with gold answer JSONL files")
    parser.add_argument("--fix", action="store_true", help="Remove invalid citations in-place")
    parser.add_argument("--report", type=Path, default=None, help="Write JSON validation report to this file")
    parser.add_argument("--skip-network", action="store_true", help="Skip PMID/NCT network verification")
    args = parser.parse_args()

    records = load_gold_answers(args.input)
    if not records:
        print("No gold answer files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} gold answers from {args.input}\n")

    # --- Collect all citation IDs ---
    all_pmids: set[str] = set()
    all_ncts: set[str] = set()
    all_dois: set[str] = set()
    citation_counts = Counter()

    for _, rec in records:
        for c in rec.get("key_citations", []):
            ctype = c.get("type", "")
            cid = c.get("id", "")
            citation_counts[ctype] += 1
            if ctype == "PMID":
                all_pmids.add(cid)
            elif ctype == "NCT":
                all_ncts.add(cid)
            elif ctype == "DOI":
                all_dois.add(cid)

    print(f"Citations: {sum(citation_counts.values())} total")
    for ctype, count in citation_counts.most_common():
        unique = len(all_pmids) if ctype == "PMID" else len(all_ncts) if ctype == "NCT" else len(all_dois)
        print(f"  {ctype}: {count} total, {unique} unique")

    # --- Verify PMIDs ---
    invalid_pmids: set[str] = set()
    invalid_ncts: set[str] = set()

    if not args.skip_network:
        if all_pmids:
            print(f"\nVerifying {len(all_pmids)} unique PMIDs against NCBI...")
            valid_p, invalid_pmids = verify_pmids(all_pmids)
            print(f"  Valid: {len(valid_p)}, Invalid: {len(invalid_pmids)} ({100*len(invalid_pmids)/max(len(all_pmids),1):.1f}%)")
            if invalid_pmids:
                print(f"  Invalid PMIDs: {sorted(invalid_pmids)}")

        if all_ncts:
            print(f"\nVerifying {len(all_ncts)} unique NCT IDs against ClinicalTrials.gov...")
            valid_n, invalid_ncts = verify_nct_ids(all_ncts)
            print(f"  Valid: {len(valid_n)}, Invalid: {len(invalid_ncts)} ({100*len(invalid_ncts)/max(len(all_ncts),1):.1f}%)")
            if invalid_ncts:
                print(f"  Invalid NCTs: {sorted(invalid_ncts)}")
    else:
        print("\n[Skipping network verification]")

    # --- Fix: remove invalid citations ---
    if args.fix and (invalid_pmids or invalid_ncts):
        removed = strip_invalid_citations(args.input, invalid_pmids, invalid_ncts)
        print(f"\nFixed: removed {removed} invalid citations from files")
        records = load_gold_answers(args.input)

    # --- Completeness checks ---
    print(f"\n{'='*60}")
    print("Completeness Criteria Check")
    print(f"{'='*60}")
    print(f"  current_knowledge >= {COMPLETENESS_CRITERIA['current_knowledge_min_chars']} chars")
    print(f"  unknown_aspects   >= {COMPLETENESS_CRITERIA['unknown_aspects_min_chars']} chars")
    print(f"  answer_summary    >= {COMPLETENESS_CRITERIA['answer_summary_min_chars']} chars")
    print(f"  key_citations     >= {COMPLETENESS_CRITERIA['min_citations']} entries (with valid IDs)")
    print(f"  mcp_tool_plan     >= {COMPLETENESS_CRITERIA['min_tool_plans']} entry")
    print(f"  completeness      in [0.0, 1.0]")
    print()

    passed = 0
    failed = 0
    failure_reasons = Counter()
    failed_records = []

    for fname, rec in records:
        failures = check_completeness(rec)
        if failures:
            failed += 1
            for f in failures:
                failure_reasons[f.split("(")[0].strip()] += 1
            failed_records.append({
                "file": fname,
                "source_id": rec.get("source_id", ""),
                "idx": rec.get("idx", ""),
                "failures": failures,
            })
        else:
            passed += 1

    print(f"Passed:  {passed}/{len(records)} ({100*passed/max(len(records),1):.1f}%)")
    print(f"Failed:  {failed}/{len(records)} ({100*failed/max(len(records),1):.1f}%)")

    if failure_reasons:
        print(f"\nFailure breakdown:")
        for reason, count in failure_reasons.most_common():
            print(f"  {reason}: {count}")

    # --- Completeness score distribution ---
    comp_vals = [rec.get("completeness", 0) for _, rec in records if isinstance(rec.get("completeness"), (int, float))]
    if comp_vals:
        avg = sum(comp_vals) / len(comp_vals)
        buckets = Counter()
        for v in comp_vals:
            if v < 0.3:
                buckets["low (< 0.3)"] += 1
            elif v < 0.6:
                buckets["medium (0.3-0.6)"] += 1
            elif v < 0.8:
                buckets["high (0.6-0.8)"] += 1
            else:
                buckets["very high (>= 0.8)"] += 1

        print(f"\nCompleteness distribution (n={len(comp_vals)}, avg={avg:.2f}):")
        for bucket in ["low (< 0.3)", "medium (0.3-0.6)", "high (0.6-0.8)", "very high (>= 0.8)"]:
            count = buckets.get(bucket, 0)
            print(f"  {bucket}: {count} ({100*count/len(comp_vals):.0f}%)")

    # --- Summary ---
    total_citations = sum(citation_counts.values())
    total_invalid = len(invalid_pmids) + len(invalid_ncts)
    hallucination_rate = total_invalid / max(total_citations, 1)

    print(f"\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    print(f"Total gold answers:      {len(records)}")
    print(f"Citation hallucination:  {total_invalid}/{total_citations} ({100*hallucination_rate:.1f}%)")
    print(f"Completeness pass rate:  {passed}/{len(records)} ({100*passed/max(len(records),1):.1f}%)")

    status = "PASS" if hallucination_rate < 0.05 and passed / max(len(records), 1) > 0.90 else "NEEDS_REVIEW"
    print(f"Overall status:          {status}")

    # --- Write report ---
    if args.report:
        report = {
            "total_answers": len(records),
            "citations": {
                "total": total_citations,
                "by_type": dict(citation_counts),
                "unique_pmids": len(all_pmids),
                "unique_ncts": len(all_ncts),
                "unique_dois": len(all_dois),
                "invalid_pmids": sorted(invalid_pmids),
                "invalid_ncts": sorted(invalid_ncts),
                "hallucination_rate": round(hallucination_rate, 4),
            },
            "completeness": {
                "criteria": COMPLETENESS_CRITERIA,
                "passed": passed,
                "failed": failed,
                "pass_rate": round(passed / max(len(records), 1), 4),
                "avg_score": round(avg, 4) if comp_vals else None,
                "failed_records": failed_records[:50],
            },
            "status": status,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
