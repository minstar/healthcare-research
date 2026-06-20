# Paper-claim provenance (verified numbers, exact files & definitions)

Every claim we'd put in the paper, with its source file and precise definition.
Corrected/caveated where the earlier informal phrasing was loose.

| claim | number | source file | definition | status |
|---|---|---|---|---|
| judge reliability (checklist) | Spearman 0.79–0.86 | results/baseline_*_300/checklist_{glm,qwen}.jsonl | rank corr of the two judges' checklist scores on the SAME answers | ✅ verified |
| 3rd-model corroboration | V4 fails 88% of GLM+Qwen both-fail | results/nanje_3model_verification.jsonl (n=288) | of items both GLM&Qwen score <0.5, fraction V4 also <0.5 | ✅ verified |
| core-nanje fraction | all-3-fail 50% | nanje_3model_verification.jsonl | 3-model bucket @threshold 0.5 | ✅ verified |
| self-rated difficulty invalid | — | results/empirical_difficulty_checklist*.jsonl | self-rating uncorrelated with empirical pass | ✅ verified |
| **parametric fallback (CORRECTED)** | **GLM 74% explicit bail; Qwen/V4 ~0%** | baseline_*_1969/traces.jsonl | answer contains a bail-phrase ("tools returned nothing"/"training knowledge"). **Model-specific, NOT a universal 60–70%.** the "non-attempt" 66–96% is a strict SFT filter (bail OR no-PMID-retrieved) and must NOT be reported as a fallback rate. | ⚠️ corrected |
| **ungrounded citation (CAVEATED)** | ~90% of cited PMIDs, but base is tiny (≈0.05 cites/trace) | results/fabrication_rate_1969.json | cited-in-answer PMID not present in the trace's own retrieved tool results. Report WITH base: models barely cite (qwen 140 / dsv4 85 citations across 1969 traces). | ⚠️ caveated |
| gold PMID existence | 0.0% invalid (0/9195) | data/export/validation_report.json | PMID resolves to a real NCBI record | ✅ verified |
| **gold citation correctness (RECONCILED)** | **74% of citations point to the WRONG paper** | results/citation_relevance_llm.jsonl (n=360 w/ abstract) | LLM judges whether the cited PMID's abstract supports the stated relevance claim; "no"=268/360. **VERIFIED: 12/12 sampled "no" cases are genuine mismatches** (0 token overlap claim↔real title; e.g. "magnesium BP meta-analysis" → a school-meals paper). | ✅ verified |

## The sharp, defensible framing for the paper
Existence-checks (the standard "hallucination" metric) report **0% hallucination** because every
cited PMID is a real ID — yet **74% of those IDs point to an unrelated paper**. The gold-generating
LLM emitted plausible relevance descriptions attached to fabricated/wrong PMID numbers. This is a
failure mode that existence-checks structurally miss; content-grounded verification is required.
(Note: this concerns GOLD-answer citations; we therefore do NOT use gold PMIDs as ground truth —
the benchmark grades via question-derived checklists, not gold citations.)
