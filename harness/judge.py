"""LLM-as-judge evaluation comparing model answer against gold answer."""
from __future__ import annotations

import asyncio
import json
import logging
import statistics
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

logger = logging.getLogger(__name__)

# Scoring dimensions and their weights
DIMENSIONS = {
    "coverage": 0.25,
    "evidence_quality": 0.20,
    "tool_usage": 0.25,
    "reasoning_quality": 0.15,
    "completeness": 0.15,
}

_JUDGE_SYSTEM_PROMPT = """\
You are a medical research evaluation expert. You will be given:
1. A medical research question
2. A gold-standard reference answer
3. A model's answer (including any tool calls it made)

Score the model's answer on 5 dimensions, each from 0.0 to 1.0:

1. **coverage** (weight 0.25): Did the answer address the key aspects identified in the gold answer? Does it cover the main points, findings, or considerations?

2. **evidence_quality** (weight 0.20): Are citations accurate? Are evidence levels (RCT, meta-analysis, case study, etc.) correctly identified? Are sources real and relevant?

3. **tool_usage** (weight 0.25): Did the model use the right tools with appropriate queries? Were the tools used effectively to gather relevant evidence? Did it use tools that match the question's needs?

4. **reasoning_quality** (weight 0.15): Is the reasoning coherent and logical? Does the model appropriately acknowledge uncertainty? Does it correctly synthesize information from multiple sources?

5. **completeness** (weight 0.15): Does the answer match the expected depth and completeness? Is it appropriately comprehensive without being off-topic?

Return ONLY a JSON object with this exact structure (no markdown, no extra text):
{
    "scores": {
        "coverage": <float 0.0-1.0>,
        "evidence_quality": <float 0.0-1.0>,
        "tool_usage": <float 0.0-1.0>,
        "reasoning_quality": <float 0.0-1.0>,
        "completeness": <float 0.0-1.0>
    },
    "reasoning": "<brief explanation of scores>"
}
"""

_JUDGE_USER_TEMPLATE = """\
## Question
{question}

## Gold-Standard Reference Answer
{gold_answer}

## Model's Answer
{model_answer}

## Model's Tool Calls
{tool_calls}

Please evaluate the model's answer against the gold standard and return your JSON scoring.
"""


@dataclass
class JudgeResult:
    task_id: str
    scores: dict[str, float]
    weighted_total: float
    reasoning: str


@dataclass
class EnsembleResult:
    """Result from multi-judge ensemble scoring."""

    median_scores: dict[str, float]  # per-dimension median
    main_score: float  # weighted median
    individual_results: list[JudgeResult]  # each judge's full result
    agreement: dict[str, float]  # per-dimension std dev (lower = more agreement)
    cohens_kappa: float  # inter-judge agreement on pass/fail
    passed: bool


def _cohens_kappa_pairwise(
    decisions_a: list[bool], decisions_b: list[bool]
) -> float:
    """Compute Cohen's kappa between two raters' binary decisions.

    kappa = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement by chance.
    """
    n = len(decisions_a)
    if n == 0:
        return 0.0

    # Observed agreement
    agree = sum(1 for a, b in zip(decisions_a, decisions_b) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    pos_a = sum(decisions_a) / n
    pos_b = sum(decisions_b) / n
    neg_a = 1 - pos_a
    neg_b = 1 - pos_b
    p_e = (pos_a * pos_b) + (neg_a * neg_b)

    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    return (p_o - p_e) / (1 - p_e)


def _compute_cohens_kappa(
    judge_results_list: list[list[JudgeResult]],
    pass_threshold: float = 0.60,
) -> float:
    """Compute average pairwise Cohen's kappa across all judge pairs.

    Args:
        judge_results_list: List of result-lists, one per judge. Each inner list
            contains JudgeResult objects in the same task order.
        pass_threshold: Score threshold for pass/fail classification.

    Returns:
        Average pairwise kappa. Returns 1.0 if only one judge.
    """
    n_judges = len(judge_results_list)
    if n_judges < 2:
        return 1.0

    # Build per-judge decision vectors (pass=True, fail=False)
    decision_vectors: list[list[bool]] = []
    for results in judge_results_list:
        decisions = [jr.weighted_total >= pass_threshold for jr in results]
        decision_vectors.append(decisions)

    # Average pairwise kappa
    kappas = []
    for i, j in combinations(range(n_judges), 2):
        k = _cohens_kappa_pairwise(decision_vectors[i], decision_vectors[j])
        kappas.append(k)

    return sum(kappas) / len(kappas) if kappas else 1.0


class EnsembleJudge:
    """Multi-judge ensemble that runs several LLM judges and aggregates results.

    Args:
        judge_models: List of litellm model specs,
            e.g. ["gemini/gemini-2.5-pro", "gpt-4o", "claude-opus-4-20250514"].
        weights: Optional per-dimension weights. If None, uses the global DIMENSIONS.
        pass_threshold: Score threshold for pass/fail classification (default: 0.60).
    """

    def __init__(
        self,
        judge_models: list[str],
        weights: dict[str, float] | None = None,
        pass_threshold: float = 0.60,
    ):
        if not judge_models:
            raise ValueError("judge_models must contain at least one model")
        self.judge_models = judge_models
        self.weights = weights or DIMENSIONS
        self.pass_threshold = pass_threshold

    async def score(
        self,
        task_id: str,
        question: str,
        gold_answer: dict | str | None,
        model_answer: str,
        tool_calls: list[dict],
    ) -> EnsembleResult:
        """Run all judges in parallel and aggregate results.

        Returns:
            EnsembleResult with median scores, agreement stats, and kappa.
        """
        # Run all judges concurrently
        coros = [
            judge_single(
                task_id=task_id,
                question=question,
                gold_answer=gold_answer,
                model_answer=model_answer,
                tool_calls=tool_calls,
                judge_model=model,
            )
            for model in self.judge_models
        ]
        individual_results: list[JudgeResult] = await asyncio.gather(*coros)

        # Compute per-dimension median and std dev
        median_scores: dict[str, float] = {}
        agreement: dict[str, float] = {}

        for dim in DIMENSIONS:
            dim_scores = [jr.scores.get(dim, 0.0) for jr in individual_results]
            median_scores[dim] = round(statistics.median(dim_scores), 4)
            if len(dim_scores) >= 2:
                agreement[dim] = round(statistics.stdev(dim_scores), 4)
            else:
                agreement[dim] = 0.0

        # Weighted median score using the per-dimension medians
        main_score = 0.0
        for dim, weight in self.weights.items():
            main_score += median_scores.get(dim, 0.0) * weight
        main_score = round(main_score, 4)

        # Cohen's kappa: for a single task, each judge gives one pass/fail decision.
        # We compute pairwise agreement on this single item. With one item
        # kappa degenerates, so we use it mainly at batch level.
        # Here we still compute it for consistency: perfect agreement = 1.0.
        kappa = _compute_cohens_kappa(
            [[jr] for jr in individual_results],
            pass_threshold=self.pass_threshold,
        )

        passed = main_score >= self.pass_threshold

        logger.info(
            "Ensemble judged %s: main_score=%.3f, kappa=%.3f, judges=%d "
            "(cov=%.2f evd=%.2f tool=%.2f reas=%.2f comp=%.2f)",
            task_id,
            main_score,
            kappa,
            len(individual_results),
            median_scores.get("coverage", 0),
            median_scores.get("evidence_quality", 0),
            median_scores.get("tool_usage", 0),
            median_scores.get("reasoning_quality", 0),
            median_scores.get("completeness", 0),
        )

        return EnsembleResult(
            median_scores=median_scores,
            main_score=main_score,
            individual_results=individual_results,
            agreement=agreement,
            cohens_kappa=kappa,
            passed=passed,
        )

    async def score_batch(
        self,
        items: list[dict],
        workers: int = 4,
    ) -> tuple[list[EnsembleResult], float]:
        """Score a batch of items with bounded concurrency.

        Args:
            items: List of dicts with keys: task_id, question, gold_answer,
                   model_answer, tool_calls.
            workers: Maximum concurrent ensemble scoring calls.

        Returns:
            Tuple of (list of EnsembleResult, batch-level Cohen's kappa).
        """
        semaphore = asyncio.Semaphore(workers)
        ensemble_results: list[EnsembleResult | None] = [None] * len(items)

        async def _worker(idx: int, item: dict) -> None:
            async with semaphore:
                er = await self.score(
                    task_id=item["task_id"],
                    question=item["question"],
                    gold_answer=item.get("gold_answer"),
                    model_answer=item["model_answer"],
                    tool_calls=item.get("tool_calls", []),
                )
                ensemble_results[idx] = er

        await asyncio.gather(*[_worker(i, item) for i, item in enumerate(items)])

        results = [er for er in ensemble_results if er is not None]

        # Compute batch-level Cohen's kappa across all tasks
        if results and len(self.judge_models) >= 2:
            # Reorganize: per-judge list of JudgeResults across all tasks
            n_judges = len(self.judge_models)
            per_judge_results: list[list[JudgeResult]] = [[] for _ in range(n_judges)]
            for er in results:
                for j_idx, jr in enumerate(er.individual_results):
                    per_judge_results[j_idx].append(jr)
            batch_kappa = _compute_cohens_kappa(
                per_judge_results, pass_threshold=self.pass_threshold
            )
        else:
            batch_kappa = 1.0

        return results, batch_kappa


def _compute_weighted_total(scores: dict[str, float]) -> float:
    """Compute the weighted total score from dimension scores."""
    total = 0.0
    for dim, weight in DIMENSIONS.items():
        total += scores.get(dim, 0.0) * weight
    return round(total, 4)


def _format_tool_calls(tool_calls: list[dict]) -> str:
    """Format tool calls for the judge prompt."""
    if not tool_calls:
        return "(No tool calls made)"
    lines = []
    for i, tc in enumerate(tool_calls, 1):
        lines.append(
            f"{i}. {tc.get('tool', 'unknown')}({json.dumps(tc.get('arguments', {}), ensure_ascii=False)})"
        )
        preview = tc.get("result_preview", "")
        if preview:
            lines.append(f"   Result preview: {preview[:200]}")
    return "\n".join(lines)


def _parse_judge_response(raw: str) -> tuple[dict[str, float], str]:
    """Extract scores and reasoning from the judge's response.

    Handles both clean JSON and JSON wrapped in markdown code blocks.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        import re

        match = re.search(r"\{[^{}]*\"scores\"[^{}]*\{[^{}]*\}[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("Could not parse judge response as JSON")
                return {dim: 0.0 for dim in DIMENSIONS}, f"Parse error. Raw: {text[:500]}"
        else:
            logger.warning("No JSON found in judge response")
            return {dim: 0.0 for dim in DIMENSIONS}, f"Parse error. Raw: {text[:500]}"

    scores_raw = data.get("scores", {})
    scores = {}
    for dim in DIMENSIONS:
        val = scores_raw.get(dim, 0.0)
        try:
            scores[dim] = max(0.0, min(1.0, float(val)))
        except (TypeError, ValueError):
            scores[dim] = 0.0

    reasoning = data.get("reasoning", "")
    return scores, reasoning


async def judge_single(
    task_id: str,
    question: str,
    gold_answer: dict | str | None,
    model_answer: str,
    tool_calls: list[dict],
    judge_model: str = "gemini/gemini-2.5-pro",
) -> JudgeResult:
    """Judge a single model answer against the gold standard.

    Args:
        task_id: Unique task identifier.
        question: The original question.
        gold_answer: Gold-standard reference answer (dict or string).
        model_answer: The model's answer text.
        tool_calls: List of tool call records from the completion.
        judge_model: LiteLLM model string for the judge.

    Returns:
        JudgeResult with dimension scores and weighted total.
    """
    import litellm

    # Format gold answer
    if isinstance(gold_answer, dict):
        gold_str = json.dumps(gold_answer, indent=2, ensure_ascii=False)
    elif gold_answer:
        gold_str = str(gold_answer)
    else:
        gold_str = "(No gold answer available)"

    user_content = _JUDGE_USER_TEMPLATE.format(
        question=question,
        gold_answer=gold_str,
        model_answer=model_answer or "(Empty answer)",
        tool_calls=_format_tool_calls(tool_calls),
    )

    try:
        resp = await litellm.acompletion(
            model=judge_model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=4096,  # reasoning judges (GLM/Qwen) spend ~1.5-2k tokens thinking before the JSON
            response_format={"type": "json_object"},
        )
        raw_output = resp.choices[0].message.content or ""
    except Exception as exc:
        logger.error("Judge call failed for %s: %s", task_id, exc)
        scores = {dim: 0.0 for dim in DIMENSIONS}
        return JudgeResult(
            task_id=task_id,
            scores=scores,
            weighted_total=0.0,
            reasoning=f"Judge error: {exc}",
        )

    scores, reasoning = _parse_judge_response(raw_output)
    weighted_total = _compute_weighted_total(scores)

    logger.info(
        "Judged %s: weighted_total=%.3f (cov=%.2f evd=%.2f tool=%.2f reas=%.2f comp=%.2f)",
        task_id,
        weighted_total,
        scores.get("coverage", 0),
        scores.get("evidence_quality", 0),
        scores.get("tool_usage", 0),
        scores.get("reasoning_quality", 0),
        scores.get("completeness", 0),
    )

    return JudgeResult(
        task_id=task_id,
        scores=scores,
        weighted_total=weighted_total,
        reasoning=reasoning,
    )


async def judge_batch(
    results: list[dict],
    judge_model: str = "gemini/gemini-2.5-pro",
    workers: int = 4,
) -> list[JudgeResult]:
    """Judge a batch of results with bounded concurrency.

    Args:
        results: List of dicts with keys: task_id, question, gold_answer,
                 model_answer, tool_calls.
        judge_model: LiteLLM model string for the judge.
        workers: Maximum concurrent judge calls.

    Returns:
        List of JudgeResult objects.
    """
    import asyncio

    semaphore = asyncio.Semaphore(workers)
    judge_results: list[JudgeResult | None] = [None] * len(results)

    async def _worker(idx: int, item: dict) -> None:
        async with semaphore:
            jr = await judge_single(
                task_id=item["task_id"],
                question=item["question"],
                gold_answer=item.get("gold_answer"),
                model_answer=item["model_answer"],
                tool_calls=item.get("tool_calls", []),
                judge_model=judge_model,
            )
            judge_results[idx] = jr

    await asyncio.gather(*[_worker(i, r) for i, r in enumerate(results)])
    return [jr for jr in judge_results if jr is not None]
