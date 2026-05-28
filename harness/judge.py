"""LLM-as-judge evaluation comparing model answer against gold answer."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
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
            max_tokens=2048,
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
