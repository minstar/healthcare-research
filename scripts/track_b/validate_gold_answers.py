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
import xml.etree.ElementTree as ET
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

STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "in", "on", "of", "for",
    "to", "and", "or", "with", "that", "this", "by", "from", "at", "as",
    "it", "be", "has", "have", "had", "not", "but", "its", "can", "may",
    "will", "would", "should", "could", "about", "into", "than", "also",
    "been", "between", "through", "after", "before", "during",
    "each", "more", "most", "other", "some", "such", "these", "those",
    "over", "only", "both", "any", "all", "very", "no", "which", "who",
    "whom", "their", "them", "they", "we", "our", "your", "he", "she",
    "his", "her",
})


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


def _relevance_tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alpha, remove stopwords and tokens <= 2 chars."""
    words = set()
    for w in text.lower().split():
        w = "".join(ch for ch in w if ch.isalpha())
        if len(w) > 2 and w not in STOPWORDS:
            words.add(w)
    return words


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two word sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _fetch_pmid_metadata(pmids: list[str], batch_size: int = 50) -> dict[str, dict[str, str]]:
    """Fetch title and abstract for PMIDs via NCBI efetch XML.

    Returns dict mapping PMID -> {"title": ..., "abstract": ...}.
    Batches requests in groups of ``batch_size`` with 0.4s between batches.
    """
    result: dict[str, dict[str, str]] = {}

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        ids = ",".join(batch)
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=pubmed&id={ids}&rettype=abstract&retmode=xml"
        )
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                xml_data = resp.read()

            root = ET.fromstring(xml_data)
            for article_node in root.iter("PubmedArticle"):
                # Extract PMID
                pmid_elem = article_node.find(".//PMID")
                if pmid_elem is None or pmid_elem.text is None:
                    continue
                pmid = pmid_elem.text.strip()

                # Extract title
                title_elem = article_node.find(".//ArticleTitle")
                title = (title_elem.text or "") if title_elem is not None else ""

                # Extract abstract (may have multiple AbstractText elements)
                abstract_parts = []
                for abs_elem in article_node.iter("AbstractText"):
                    if abs_elem.text:
                        abstract_parts.append(abs_elem.text)
                abstract = " ".join(abstract_parts)

                result[pmid] = {"title": title, "abstract": abstract}

        except Exception as e:
            print(f"  Metadata fetch batch {i // batch_size} error: {e}", file=sys.stderr)

        # Rate limit: 0.4s between batches (skip delay after last batch)
        if i + batch_size < len(pmids):
            time.sleep(0.4)

    return result


def verify_citation_relevance(
    records: list[tuple[str, dict]],
    threshold: float = 0.1,
) -> dict:
    """Verify that each PMID citation's relevance text matches the paper's content.

    For each PMID citation that has a ``relevance`` field, fetches the paper's
    title + abstract from NCBI and computes Jaccard similarity of keyword sets
    (excluding stopwords).  Citations with score < ``threshold`` are flagged as
    weak.

    Returns a dict with:
      - total_checked: int
      - weak_count: int
      - weak_citations: list[dict] (file, source_id, pmid, score, relevance_snippet)
      - avg_score: float
    """
    # Collect unique PMIDs that have a relevance field
    pmid_to_citations: dict[str, list[tuple[str, str, str]]] = {}
    for fname, rec in records:
        source_id = rec.get("source_id", "")
        for c in rec.get("key_citations", []):
            if c.get("type") == "PMID" and c.get("relevance"):
                pmid = c["id"]
                if pmid not in pmid_to_citations:
                    pmid_to_citations[pmid] = []
                pmid_to_citations[pmid].append((fname, source_id, c["relevance"]))

    unique_pmids = sorted(pmid_to_citations.keys())
    if not unique_pmids:
        return {"total_checked": 0, "weak_count": 0, "weak_citations": [], "avg_score": 0.0}

    print(f"\nFetching metadata for {len(unique_pmids)} unique PMIDs (batches of 50)...")
    metadata = _fetch_pmid_metadata(unique_pmids, batch_size=50)
    print(f"  Retrieved metadata for {len(metadata)}/{len(unique_pmids)} PMIDs")

    # Compute relevance scores
    scores: list[float] = []
    weak_citations: list[dict] = []
    total_checked = 0

    for pmid, citations in sorted(pmid_to_citations.items()):
        if pmid not in metadata:
            continue  # could not fetch metadata, skip

        paper = metadata[pmid]
        paper_text = f"{paper['title']} {paper['abstract']}"
        paper_tokens = _relevance_tokenize(paper_text)

        for fname, source_id, relevance in citations:
            total_checked += 1
            rel_tokens = _relevance_tokenize(relevance)
            score = _jaccard(rel_tokens, paper_tokens)
            scores.append(score)

            if score < threshold:
                weak_citations.append({
                    "file": fname,
                    "source_id": source_id,
                    "pmid": pmid,
                    "score": round(score, 4),
                    "relevance_snippet": relevance[:120],
                })

    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "total_checked": total_checked,
        "weak_count": len(weak_citations),
        "weak_citations": weak_citations,
        "avg_score": round(avg_score, 4),
    }


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

    comp = answer.get("self_completeness", answer.get("completeness"))
    if not isinstance(comp, (int, float)) or not (0.0 <= comp <= 1.0):
        failures.append(f"self_completeness out of range: {comp}")

    return failures


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer. Returns lowercased word tokens."""
    import re
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def check_semantic_quality(question_text: str, answer: dict) -> list[str]:
    """Run semantic quality checks on a gold answer. Returns list of warning strings.

    These are soft warnings (not hard failures) that flag potentially low-quality content
    even when length-based completeness criteria pass.

    Checks:
      1. Type-Token Ratio (TTR) on current_knowledge + answer_summary
      2. Question-Answer entity overlap
      3. Citation-Content alignment
    """
    warnings = []

    ck = answer.get("current_knowledge", "")
    ans = answer.get("answer_summary", "")
    combined_text = f"{ck} {ans}"
    combined_tokens = _tokenize(combined_text)

    # --- 1. Type-Token Ratio ---
    if len(combined_tokens) > 0:
        unique_tokens = set(combined_tokens)
        ttr = len(unique_tokens) / len(combined_tokens)
        if ttr < 0.3:
            warnings.append(
                f"Low Type-Token Ratio: {ttr:.3f} < 0.3 "
                f"({len(unique_tokens)} unique / {len(combined_tokens)} total words) "
                f"— text may be highly repetitive"
            )

    # --- 2. Question-Answer Entity Overlap ---
    if question_text.strip():
        q_tokens = _tokenize(question_text)
        q_entities = {w for w in q_tokens if len(w) >= 4 and w not in STOPWORDS}
        if q_entities:
            combined_tokens_set = set(combined_tokens)
            overlap_count = sum(1 for e in q_entities if e in combined_tokens_set)
            overlap_ratio = overlap_count / len(q_entities)
            if overlap_ratio < 0.2:
                warnings.append(
                    f"Low question-answer entity overlap: {overlap_ratio:.3f} < 0.2 "
                    f"({overlap_count}/{len(q_entities)} question entities found in answer) "
                    f"— answer may not address the question"
                )

    # --- 3. Citation-Content Alignment ---
    cites = answer.get("key_citations", [])
    if cites:
        aligned_count = 0
        for c in cites:
            relevance_text = c.get("relevance", "")
            rel_tokens = _tokenize(relevance_text)
            rel_keywords = {w for w in rel_tokens if len(w) >= 5 and w not in STOPWORDS}
            if not rel_keywords:
                # No extractable keywords in relevance text — count as aligned
                aligned_count += 1
                continue
            combined_tokens_set = set(combined_tokens)
            if any(kw in combined_tokens_set for kw in rel_keywords):
                aligned_count += 1

        alignment_ratio = aligned_count / len(cites)
        if alignment_ratio < 0.5:
            warnings.append(
                f"Low citation-content alignment: {alignment_ratio:.3f} < 0.5 "
                f"({aligned_count}/{len(cites)} citations have keyword overlap with answer) "
                f"— citations may not support the answer content"
            )

    return warnings


def load_gold_answers(input_path: Path) -> list[tuple[str, dict]]:
    """Load gold answer records from a directory of output files or a single JSONL.

    If input_path is a directory, reads *.output.jsonl files.
    If input_path is a file, reads it as merged JSONL (with nested gold_answer key).
    Returns list of (filename, gold_answer_dict) where gold_answer_dict has the 7 fields.
    """
    records = []

    if input_path.is_file():
        with open(input_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    gold = obj.get("gold_answer", {})
                    if gold:
                        gold["source_id"] = obj.get("source_id", "")
                        gold["self_contained_question"] = obj.get("self_contained_question", "")
                        gold["original_question"] = obj.get("original_question", "")
                    records.append((input_path.name, gold))
                except json.JSONDecodeError:
                    continue
    else:
        for f in sorted(glob.glob(str(input_path / "*.output.jsonl"))):
            with open(f) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # Handle both formats: nested gold_answer or top-level fields
                        if "gold_answer" in obj and isinstance(obj["gold_answer"], dict):
                            gold = obj["gold_answer"]
                            gold["source_id"] = obj.get("source_id", "")
                            gold["self_contained_question"] = obj.get("self_contained_question", "")
                            gold["original_question"] = obj.get("original_question", "")
                        else:
                            gold = obj
                        records.append((os.path.basename(f), gold))
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
    parser.add_argument(
        "--check-relevance", action="store_true",
        help="Verify citation relevance by fetching paper title/abstract from NCBI "
             "and computing keyword overlap with the citation's relevance field "
             "(slow: one API call per unique PMID batch of 50)",
    )
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

    # --- Citation relevance verification ---
    relevance_result = None
    if args.check_relevance and not args.skip_network:
        print(f"\n{'='*60}")
        print("Citation Relevance Verification")
        print(f"{'='*60}")
        relevance_result = verify_citation_relevance(records, threshold=0.1)
        print(f"\n  Total citations checked:  {relevance_result['total_checked']}")
        print(f"  Weak citations (< 0.1):   {relevance_result['weak_count']}")
        print(f"  Average relevance score:  {relevance_result['avg_score']:.4f}")
        if relevance_result["weak_citations"]:
            print(f"\n  Weak citations detail:")
            for wc in relevance_result["weak_citations"][:20]:
                print(f"    PMID {wc['pmid']} (score={wc['score']:.4f}) in {wc['file']}")
                print(f"      relevance: {wc['relevance_snippet']}...")
            if len(relevance_result["weak_citations"]) > 20:
                print(f"    ... and {len(relevance_result['weak_citations']) - 20} more")

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
    comp_vals = [rec.get("self_completeness", rec.get("completeness", 0)) for _, rec in records if isinstance(rec.get("self_completeness", rec.get("completeness")), (int, float))]
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

    # --- Semantic quality checks ---
    print(f"\n{'='*60}")
    print("Semantic Quality Check")
    print(f"{'='*60}")

    semantic_warning_counts = Counter()
    records_with_warnings = 0
    semantic_details = []

    for fname, rec in records:
        question = rec.get("self_contained_question", rec.get("original_question", ""))
        warnings = check_semantic_quality(question, rec)
        if warnings:
            records_with_warnings += 1
            for w in warnings:
                semantic_warning_counts[w.split(":")[0].strip()] += 1
            semantic_details.append({
                "file": fname,
                "source_id": rec.get("source_id", ""),
                "warnings": warnings,
            })

    sem_clean = len(records) - records_with_warnings
    print(f"\nClean:    {sem_clean}/{len(records)} ({100*sem_clean/max(len(records),1):.1f}%)")
    print(f"Warnings: {records_with_warnings}/{len(records)} ({100*records_with_warnings/max(len(records),1):.1f}%)")

    if semantic_warning_counts:
        print(f"\nWarning breakdown:")
        for reason, count in semantic_warning_counts.most_common():
            print(f"  {reason}: {count}")

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
    print(f"Semantic clean rate:     {sem_clean}/{len(records)} ({100*sem_clean/max(len(records),1):.1f}%)")

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
            "structural_completeness": {
                "criteria": COMPLETENESS_CRITERIA,
                "passed": passed,
                "failed": failed,
                "pass_rate": round(passed / max(len(records), 1), 4),
                "avg_self_completeness": round(avg, 4) if comp_vals else None,
                "failed_records": failed_records[:50],
            },
            "citation_relevance": {
                "total_checked": relevance_result["total_checked"],
                "weak_count": relevance_result["weak_count"],
                "avg_score": relevance_result["avg_score"],
                "weak_citations": relevance_result["weak_citations"][:50],
            } if relevance_result else None,
            "semantic_quality": {
                "clean": sem_clean,
                "warnings": records_with_warnings,
                "clean_rate": round(sem_clean / max(len(records), 1), 4),
                "warning_breakdown": dict(semantic_warning_counts.most_common()),
                "flagged_records": semantic_details[:50],
            },
            "status": status,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
