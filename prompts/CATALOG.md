# Prompt Catalog & Registry

Single source of truth for every LLM prompt in the benchmark, mapped to the
ResearchMath-14K appendix structure (arxiv:2605.28003, Appendix H) so our paper can
document them at the same rigor. Each entry: current location, purpose, paper
category, detail assessment, and upgrade priority.

> ⚠️ Prompts are currently inline in code (scattered across ~12 files). This catalog
> is the migration target. Do NOT edit `harness/completion_runner.py` while an eval
> run is in flight — apply agent-prompt upgrades between runs only.

## Paper appendix categories (H.1–H.4) ↔ our prompts

| Paper (Appendix H) | Our prompt | File | Detail | Priority |
|--------------------|-----------|------|--------|----------|
| H.1 Extractor Agent | `SYSTEM_PROMPT` | pipeline/extractor.py | structured; **no few-shot, no accept/reject criteria** | P1 |
| H.1 Refiner Agent | `SYSTEM_PROMPT` + `USER_TEMPLATE` | pipeline/refiner.py | structured; status now retrieval-grounded via status_verifier | P2 |
| H.1 Refiner (status) | grounded-judgment prompt | pipeline/status_verifier.py (Stage-2 design) / scripts/stage2_judge.py | evidence-constrained; OK | P3 |
| H.2 Difficulty Comparison | (we use empirical pass-rate, not LLM difficulty) | — | replaced by 3-model empirical labels | n/a |
| H.3 Response Generation | gold-answer prompt | scripts/track_b/generate_gold_answers.py | detailed schema | P2 |
| H.3 Response (model-under-test) | **agent system prompt** | harness/completion_runner.py (`_OpenAIBackend`) | 🔴 **3 sentences only** | **P0** |
| H.4 Factuality / Judge | checklist judge `SYS` | scripts/checklist_judge.py | works (Spearman 0.79-0.86); no few-shot | P1 |
| (new) Rubric generation | `SYS` | scripts/gen_rubrics.py | works; no few-shot, no examples | P1 |
| (legacy) free-form judge | `_JUDGE_SYSTEM_PROMPT` | harness/judge.py | DEPRECATED (vague dims caused judge variance) | retire |

## Key gaps vs paper rigor
1. **Agent prompt is 3 sentences** (P0). For an OPEN-question agentic benchmark it must
   specify: the epistemic nature (you often CANNOT fully solve — map known vs unknown);
   tool strategy (reformulate on empty results, use multiple tools, don't bail to memory);
   grounding (cite PMIDs/NCTs for claims, never fabricate "tools returned nothing");
   output structure. The current thin prompt under-elicits capability and is a likely
   cause of the 60%+ parametric-fallback we measured.
2. **Zero few-shot examples** in any prompt. The paper uses worked examples.
3. **No explicit accept/reject criteria** in the extractor (what is a genuine open
   question vs review-section boilerplate).
4. **Scattered inline prompts** — not versioned or documented.

## Upgrade plan (apply between eval runs)
- P0: rewrite the agent system prompt (draft in `prompts/agent_system.md`).
- P1: add few-shot + accept/reject to extractor, checklist-judge, rubric-gen.
- Migrate prompts to `prompts/*.md` (text) + a thin loader, so code imports from here.
