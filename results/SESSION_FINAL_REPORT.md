# Session final report — open-medical-questions benchmark (2026-05-29 → 06-01)

## 1. Data
- Corpus v3: **12,553** (retrieval_verified 6,648 + expert_consensus 5,905 = JLA 3,759 + NICE 2,146).
- **+382 new** from authoritative priority-setting crawler (581 docs: consensus/scientific
  statements, society research agendas, top-10 priorities, CHNRI, Delphi) → PubMed track 6,733→7,115.
- Source diversity: top-50 docs = **10.7%** (vs ResearchMath 25.78% — we are MORE diverse).

## 2. Empirical difficulty (VALIDATED on 300-set, 3 models)
- 3-model checklist eval (GLM-5.1, Qwen3.6, DeepSeek-V4): both-fail **143 (50% core nanje)** /
  split 129 (45%) / both-pass 16 (6%).
- DeepSeek-V4 (different family) also fails **88%** of the GLM+Qwen both-fail items → the hard
  core is genuine, not a 2-model blindspot.
- Self-rated difficulty is **invalid** (no correlation with model performance) → deprecated.
- 1969-scale: all 3 models' **traces collected (1,969 each)**; checklist judging INCOMPLETE
  (GLM serving hit 12h walltime mid-judging) → resumable on a GLM re-serve.

## 3. Judge reliability (key fix)
- Free-form 5-dim judging was unreliable (same answers 0.50 vs 0.30 by judge; coverage/
  completeness Spearman 0.19).
- **Per-question checklist rubrics** (must_mention/acknowledge/ground/avoid) → judge agreement
  **Spearman 0.79–0.86**. This is what makes the benchmark trustworthy.

## 4. Verification vs paper (Appendix H)
- self-containment: **85.4% fully** (500; retrieval .975 / consensus .852) — near paper's 94.2%.
- citation correctness: 🔴 **~74% of gold citation PMIDs MISMATCHED** (point to unrelated papers).
  Old "0% hallucination" only checked PMID existence; LLM-judge revealed the real problem.
- Stage-2 open_status grounded re-judgment: FAILED twice on GLM contention/timeout → clean re-run pending.

## 5. Paper elements applied
- **Trajectory→SFT flywheel**: eval traces filtered (non-attempt/fake-cite/quality) into SFT
  messages. 300×3: 900→58 kept. **DeepSeek-V4 = best teacher**. @1969: ~1,140 candidates pre-quality.
- **Citation-fabrication metric**: ~90% of model citations are ungrounded (cited, not retrieved).
- **Contamination concentration** metric (10.7%).

## 6. Negative results caught by verify-before-trust
- **P0 agent prompt FAILED A/B** — broke DeepSeek-V4 tool use (0-tool traces 2%→91%); reverted.
- **Jaccard citation metric** was an artifact (replaced by LLM-judge).
- **Stage-2 contention** failures (×2) → must run GLM jobs sequentially.

## 7. Infra learnings
- Cluster is **B200 (Blackwell)**, not H200. GLM-5.1 needs DeepGEMM v2.1.1 built vs torch 2.10.
  DeepSeek-V4 served via `deepseekv4-cu130` docker; emits recursively-nested tool args (handled).
- **GLM node cannot take >~8 concurrent thinking-model requests** — running 3–4 GLM jobs at once
  caused the Stage-2 and checklist failures. Run GLM jobs sequentially.

## GPU
All my serving jobs ended (GLM 23035 TIMEOUT@12h, Qwen/V4 scancelled). Nothing of mine running.
Training jobs (pt2-minstar-tau3 21567_*/21568_*) untouched.

## Pending (need a fresh GLM re-serve, run SEQUENTIALLY)
1969 checklist judging → 3-model difficulty at scale → v3.2 recalibration; Stage-2 clean re-run;
gold-citation regeneration (74% mismatch); P0 v2 tool-preserving redesign; Elo cross-benchmark.
