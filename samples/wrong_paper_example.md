# Wrong-Paper Citation — A Worked Example

A real, unedited trace from the **OpenBioRQ** audit that shows the benchmark's central finding:
*a citation can resolve to a real paper and still not support the claim it is attached to* —
**existence ≠ correctness**.

---

## 1. The question (open research question, no answer key)

| field | value |
|---|---|
| `task_id` | `PMID:33498787#1` |
| track | `expand` (Cochrane research-gap literature) |
| source paper | PMID 33498787 |
| answering model | GLM-5.1 |
| tool calls in this trace | 15 (PubMed · ClinicalTrials.gov · openFDA) |

> **What are the long-term safety profiles and potential adverse effects of mRNA vaccines
> administered to diverse populations over months and years of follow-up?**

Long-term, multi-year safety across diverse populations is genuinely unsettled — which is why this
qualifies as an *open* question with no answer key. The honest response is a grounded synthesis of
what is and is not known.

---

## 2. What the agent answered (verbatim excerpt)

> ### Pfizer-BioNTech (BNT162b2)
> - The original Phase 3 trial (Polack et al., *NEJM*, 2020; PMID: 33301246) enrolled ~44,000 participants.
> - **6-month follow-up data (Thomas et al., *NEJM*, 2021; PMID: 34407296)** showed:
>   - **Vaccine efficacy remained ~91% against symptomatic COVID-19.**
>   - Serious adverse events were balanced between vaccine (0.6%) and placebo (0.5%).
>   - No new safety signals emerged beyond the initial reporting period.

---

## 3. The problem: `PMID 34407296` is the wrong paper

| | |
|---|---|
| **Claim** | "Vaccine efficacy remained ~91% against symptomatic COVID-19" |
| **Citation attached** | `PMID 34407296` |
| **What PMID 34407296 actually is** | *"Comparison of the efficacy of topical insulin with autologous serum eye drops in persistent epithelial defects of the cornea"* — an **ophthalmology** study |
| **Does it support the claim?** | **❌ No** — it never mentions a vaccine |

- ✅ **L1 — existence:** the identifier resolves to a real, indexed article → an existence check **passes**.
- ❌ **L2 — content support:** the cited paper is about corneal eye-drops, not vaccines → **wrong-paper**.

**Three independent judges agree it is wrong-paper:**

| judge | verdict |
|---|---|
| GLM-5.1 (primary) | `no` |
| Opus-4.7 (independent family) | `no` |
| human (non-expert, blind) | `no` |

---

## 4. Why this is the *dangerous* case

The underlying fact may even be correct — the real Thomas et al. 6-month follow-up did report ~91%
efficacy, and the agent named the right authors and journal. It simply attached the **wrong
identifier**. The reasoning isn't the problem; the **provenance** is. A reader who clicks the PMID
lands on an unrelated eye-drops study, and an existence-only check would have waved the citation
through.

This is exactly why OpenBioRQ audits **L2 content support**, not just **L1 existence**: across the
full audit, biomedical agents almost never *fabricate* identifiers (≈0.7%), yet roughly **one in
six** of their *real* citations is wrong-paper.

---

*Provenance: `results/expand_glm/traces.jsonl` (trace), `results/cite_audit.jsonl` (L2 audit),
`results/cite_kappa/human_annotation_50.csv` row 34 (three-judge labels). Part of the OpenBioRQ
benchmark.*
