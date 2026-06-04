# OpenBioRQ — Experiment Status (for the paper)

Mirrors ResearchMath-14k's Section 2-5 structure; the differentiator is the **2-level
citation audit** (existence → content support) on model trajectories. Numbers below are
live results from `results/` (re-run the scripts to refresh).

## Experiment matrix

| Exp | Mirrors | Script(s) | Status |
|-----|---------|-----------|--------|
| A — construction quality | RM §2 | `audit_self_containment.py`, `pipeline/dedup.py`, `stage2_judge.py` | self-cont + dedup ✅; **source-vs-retrieval status ablation** TODO |
| B1 — behavior | RM §3.2/4 | `exp_behavioral.py` | ✅ done |
| B2 — 2-level citation factuality (**heart**) | RM §4 (deeper) | `exp_cite_audit.py` | L1 ✅ running→done; **L2 (content support)** running |
| C — judge agreement | (RM weak) | `checklist_judge.py` | inter-judge Spearman 0.82 ✅; **human κ** TODO |
| D — learning (SFT) | RM §5 | (trajectories ready) | deferred (SFT later); "Supervision Corpus" claim pending Exp D |

Trajectories: 3 models (GLM-5.1, Qwen3.6, DeepSeek-V4) × {1969 gold, priority 525,
expand 483} = **8,931 agentic traces** (`results/{baseline_*_1969,priority_*,expand_*}/traces.jsonl`).

## Exp B1 — behavioral metrics (1969 set)

`results/exp_behavioral.json` — `python scripts/exp_behavioral.py`

| Model | non-attempt | zero-tool (collapse) | avg tools | cite-rate |
|-------|------------:|---------------------:|----------:|----------:|
| GLM-5.1 | 26.2% | 20.8% | 12.6 | 3.9% |
| Qwen3.6 | 19.5% | 19.6% | 6.1 | 23.9% |
| DeepSeek-V4 | 0.8% | 31.3% | 8.3 | 38.5% |

- **Agentic collapse is real and model-divergent.** On the harder `priority` set GLM's
  non-attempt rate jumps to **69%** (zero-tool 65%) — a tool-use collapse ResearchMath does
  not measure.
- Citation *behavior* diverges 10× across models (GLM 4% vs DeepSeek-V4 38%), which sets up
  the factuality question: of those that cite, how many cite correctly?

## Exp B2 — 2-level citation audit (the paper's heart)

`results/cite_audit.jsonl` — `python scripts/exp_cite_audit.py` (L1: `--no-judge`; L2: `--base <glm>`)

Two distinct audits, **kept separate** to avoid the conflation reviewers will probe:

- **Gold-citation audit** (`results/citation_relevance_llm.jsonl`): LLM-generated *gold*
  answers — ~100% real PMIDs but ~74% point to the wrong paper. This is *why gold citations
  are not used as ground truth*, not a model-trajectory claim.
- **Trace-citation audit** (`results/cite_audit.jsonl`): citations models *emit* during
  agentic answering — the ResearchMath-mirror factuality story.

### L1 — existence (does the cited ID exist?)

Preliminary (model-divergent fabrication):

| Model | citations | exist % | fabricated | NCT fabrication |
|-------|----------:|--------:|-----------:|----------------:|
| GLM-5.1 | ~121 | ~84% | ~16% | **~27%** (16/59) |
| Qwen3.6 | ~1048 | ~99.5% | <1% | 5/106 |
| DeepSeek-V4 | ~2059 | ~99.7% | <1% | 3/63 |

- GLM fabricates trial IDs at scale ("based on my training knowledge…NCT0…") — fake NCTs.
- DeepSeek-V4 cites the most *and* nearly all exist → existence ≠ correctness; L2 is the test.

### L2 — content support (does the real paper support the claim?)

`supports ∈ {yes, partial, no}`; **wrong-paper = exists & supports==no**. (running — fill on completion.)

## Bio-specific additions (planned)

- `open_status` carries an **as-of date** (evidence-evolving: open→partially-answered over time).
- Framing: **Research Assistance Benchmark, NOT Clinical Decision Support** (abstract + limitations).
- Harm framing: a real PMID supporting a *wrong* claim is clinical **misinformation**, not mere
  academic sloppiness — sharper hook than math's fake references.
- Provenance appendix: each item × (PMID, status, difficulty, domain, generating script).

## Open decisions

1. "Supervision Corpus" in title/abstract ⇒ commit to Exp D (SFT on filtered traces) or drop.
2. Generation-paradox replication needs an older-gen model (current roster is all current-gen).
3. Human-annotation budget for Exp C (judge κ).
