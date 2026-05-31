"""Run a model against benchmark tasks with MCP tool access."""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from .mcp_tools import MCPToolRegistry

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 10

# P0 agent prompt (prompts/agent_system.md): open-question epistemics + tool persistence
# + graded grounding + known/unknown structure. Replaces the old 3-sentence prompt that
# drove ~70% parametric-fallback (non-attempt) trajectories.
AGENT_SYSTEM_PROMPT = (
    "You are a biomedical research agent answering an OPEN research question — one the "
    "field has NOT fully resolved. Do not fabricate a definitive answer; produce an "
    "evidence-grounded synthesis of what is known, what remains unknown, and why.\n\n"
    "TOOLS: use the provided search/database tools as your PRIMARY evidence source. Issue "
    "multiple varied queries (broad then narrow). If a query returns nothing, REFORMULATE "
    "(synonyms, broader terms, related entities) — never conclude 'no evidence exists' from "
    "one empty result, and never claim the tools failed if other queries returned results.\n\n"
    "GROUNDING: back every substantive claim with a retrieved source, cited by identifier "
    "(e.g. PMID:12345678, NCT01234567). Do not present unsupported claims or cite IDs you "
    "did not retrieve.\n\n"
    "ANSWER: (1) current knowledge with citations; (2) the precise open gap and why; "
    "(3) evidence quality/level; (4) a calibrated bottom line that states uncertainty plainly "
    "when the question is genuinely unresolved. Be specific and grounded, not a memory essay."
)


@dataclass
class CompletionResult:
    task_id: str
    model_answer: str
    tool_calls: list[dict]
    trace: list[dict]
    tokens: dict  # {"prompt": int, "completion": int}
    wall_time: float  # seconds


# ===================================================================
# Backend interface
# ===================================================================


class _Backend:
    """Abstract base for model backends."""

    def __init__(self, model: str, registry: MCPToolRegistry, tool_schemas: list[dict]) -> None:
        self.model = model
        self.registry = registry
        self.tool_schemas = tool_schemas

    async def run(self, question: str) -> CompletionResult:
        raise NotImplementedError


# ===================================================================
# OpenAI-compatible backend (works with vLLM, OpenAI, etc.)
# ===================================================================


class _OpenAIBackend(_Backend):
    """Uses the openai Python SDK with tool calling."""

    def __init__(
        self,
        model: str,
        registry: MCPToolRegistry,
        tool_schemas: list[dict],
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
    ) -> None:
        super().__init__(model, registry, tool_schemas)
        self.base_url = base_url
        self.api_key = api_key

    async def run(self, question: str) -> CompletionResult:
        # Lazy import so the module loads even without openai installed
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": question},
        ]

        trace: list[dict] = []
        tool_calls_log: list[dict] = []
        total_prompt = 0
        total_completion = 0
        t0 = time.monotonic()

        for round_idx in range(_MAX_TOOL_ROUNDS):
            try:
                resp = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas if self.tool_schemas else None,
                    tool_choice="auto" if self.tool_schemas else None,
                    temperature=0.3,
                    max_tokens=4096,
                )
            except Exception as exc:
                logger.error("OpenAI API error (round %d): %s", round_idx, exc)
                trace.append({"round": round_idx, "error": str(exc)})
                break

            choice = resp.choices[0]
            usage = resp.usage
            if usage:
                total_prompt += usage.prompt_tokens
                total_completion += usage.completion_tokens

            trace.append(
                {
                    "round": round_idx,
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": (
                        [
                            {
                                "id": tc.id,
                                "function": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                            for tc in choice.message.tool_calls
                        ]
                        if choice.message.tool_calls
                        else None
                    ),
                    "finish_reason": choice.finish_reason,
                }
            )

            # If the model is done (no tool calls), break
            if not choice.message.tool_calls:
                break

            # Append the assistant message (with tool calls) to context
            messages.append(choice.message.model_dump())

            # Execute each tool call
            for tc in choice.message.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                logger.info("Tool call [round %d]: %s(%s)", round_idx, func_name, func_args)
                result = self.registry.execute(func_name, func_args)
                tool_calls_log.append(
                    {
                        "round": round_idx,
                        "tool": func_name,
                        "arguments": func_args,
                        "result_preview": json.dumps(result.get("results", ""), default=str)[:500],
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        wall_time = time.monotonic() - t0

        # Extract final answer from last assistant content
        model_answer = ""
        for entry in reversed(trace):
            if entry.get("role") == "assistant" and entry.get("content"):
                model_answer = entry["content"]
                break

        return CompletionResult(
            task_id="",  # filled by caller
            model_answer=model_answer,
            tool_calls=tool_calls_log,
            trace=trace,
            tokens={"prompt": total_prompt, "completion": total_completion},
            wall_time=wall_time,
        )


# ===================================================================
# LiteLLM backend (any provider)
# ===================================================================


class _LiteLLMBackend(_Backend):
    """Uses litellm for model calls — supports any provider."""

    async def run(self, question: str) -> CompletionResult:
        import litellm

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": question},
        ]

        trace: list[dict] = []
        tool_calls_log: list[dict] = []
        total_prompt = 0
        total_completion = 0
        t0 = time.monotonic()

        for round_idx in range(_MAX_TOOL_ROUNDS):
            try:
                resp = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas if self.tool_schemas else None,
                    tool_choice="auto" if self.tool_schemas else None,
                    temperature=0.3,
                    max_tokens=4096,
                )
            except Exception as exc:
                logger.error("LiteLLM API error (round %d): %s", round_idx, exc)
                trace.append({"round": round_idx, "error": str(exc)})
                break

            choice = resp.choices[0]
            usage = resp.usage
            if usage:
                total_prompt += usage.prompt_tokens or 0
                total_completion += usage.completion_tokens or 0

            trace.append(
                {
                    "round": round_idx,
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": (
                        [
                            {
                                "id": tc.id,
                                "function": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                            for tc in choice.message.tool_calls
                        ]
                        if choice.message.tool_calls
                        else None
                    ),
                    "finish_reason": choice.finish_reason,
                }
            )

            if not choice.message.tool_calls:
                break

            messages.append(choice.message.model_dump())

            for tc in choice.message.tool_calls:
                func_name = tc.function.name
                try:
                    func_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    func_args = {}

                logger.info("Tool call [round %d]: %s(%s)", round_idx, func_name, func_args)
                result = self.registry.execute(func_name, func_args)
                tool_calls_log.append(
                    {
                        "round": round_idx,
                        "tool": func_name,
                        "arguments": func_args,
                        "result_preview": json.dumps(result.get("results", ""), default=str)[:500],
                    }
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    }
                )

        wall_time = time.monotonic() - t0

        model_answer = ""
        for entry in reversed(trace):
            if entry.get("role") == "assistant" and entry.get("content"):
                model_answer = entry["content"]
                break

        return CompletionResult(
            task_id="",
            model_answer=model_answer,
            tool_calls=tool_calls_log,
            trace=trace,
            tokens={"prompt": total_prompt, "completion": total_completion},
            wall_time=wall_time,
        )


# ===================================================================
# Claude CLI backend
# ===================================================================


class _ClaudeBackend(_Backend):
    """Uses `claude --print` CLI with tool schemas passed via stdin prompt."""

    async def run(self, question: str) -> CompletionResult:
        # Build prompt that includes tool schema descriptions
        tool_desc_lines = []
        for schema in self.tool_schemas:
            fn = schema["function"]
            tool_desc_lines.append(
                f"- {fn['name']}: {fn['description']}  "
                f"Parameters: {json.dumps(fn['parameters'], ensure_ascii=False)}"
            )
        tools_block = "\n".join(tool_desc_lines)

        prompt = (
            "You are a medical research assistant with access to the following tools:\n\n"
            f"{tools_block}\n\n"
            "To call a tool, output a JSON block: "
            '{"tool_call": "<tool_name>", "arguments": {...}}\n\n'
            "After gathering evidence, provide your final answer.\n\n"
            f"Question: {question}"
        )

        trace: list[dict] = []
        tool_calls_log: list[dict] = []
        t0 = time.monotonic()

        # Single-shot call via claude --print
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "--print", "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode("utf-8", errors="replace")
        except FileNotFoundError:
            output = "[ERROR] claude CLI not found"
        except asyncio.TimeoutError:
            output = "[ERROR] claude CLI timed out after 120s"

        trace.append({"round": 0, "role": "assistant", "content": output})

        # Attempt to extract inline tool calls from the output
        # (Claude CLI doesn't natively support multi-turn tool use,
        #  so this is best-effort parsing of any JSON tool_call blocks)
        import re

        for match in re.finditer(
            r'\{\s*"tool_call"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]+\})\s*\}',
            output,
        ):
            tool_name = match.group(1)
            try:
                tool_args = json.loads(match.group(2))
            except json.JSONDecodeError:
                tool_args = {}
            tool_calls_log.append(
                {
                    "round": 0,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result_preview": "(not executed — claude CLI single-shot mode)",
                }
            )

        wall_time = time.monotonic() - t0

        return CompletionResult(
            task_id="",
            model_answer=output,
            tool_calls=tool_calls_log,
            trace=trace,
            tokens={"prompt": 0, "completion": 0},
            wall_time=wall_time,
        )


# ===================================================================
# Factory
# ===================================================================


def _parse_model_spec(model_spec: str) -> tuple[str, dict]:
    """Parse a model specification string.

    Formats:
        "openai::http://localhost:8000/v1::solar-open-2-10b"
        "litellm::gemini/gemini-2.5-pro"
        "claude::claude-sonnet-4-20250514"
        "openai::solar-open-2-10b"  (default base_url)

    Returns:
        (backend_name, backend_kwargs)
    """
    parts = model_spec.split("::")
    backend_name = parts[0].lower()

    if backend_name == "openai":
        if len(parts) == 3:
            return "openai", {"base_url": parts[1], "model": parts[2]}
        elif len(parts) == 2:
            return "openai", {"base_url": "https://api.openai.com/v1", "model": parts[1]}
        else:
            return "openai", {"base_url": "https://api.openai.com/v1", "model": "gpt-4o"}
    elif backend_name == "litellm":
        model = parts[1] if len(parts) > 1 else "gpt-4o"
        return "litellm", {"model": model}
    elif backend_name == "claude":
        model = parts[1] if len(parts) > 1 else "claude-sonnet-4-20250514"
        return "claude", {"model": model}
    else:
        # Default: treat as litellm model string
        return "litellm", {"model": model_spec}


def create_backend(
    model_spec: str,
    registry: MCPToolRegistry,
    tool_schemas: list[dict],
) -> _Backend:
    """Create a backend from a model specification string."""
    backend_name, kwargs = _parse_model_spec(model_spec)
    model = kwargs.pop("model", "")

    if backend_name == "openai":
        base_url = kwargs.get("base_url", "http://localhost:8000/v1")
        api_key = kwargs.get("api_key", "EMPTY")
        return _OpenAIBackend(
            model=model,
            registry=registry,
            tool_schemas=tool_schemas,
            base_url=base_url,
            api_key=api_key,
        )
    elif backend_name == "litellm":
        return _LiteLLMBackend(
            model=model,
            registry=registry,
            tool_schemas=tool_schemas,
        )
    elif backend_name == "claude":
        return _ClaudeBackend(
            model=model,
            registry=registry,
            tool_schemas=tool_schemas,
        )
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# ===================================================================
# Public API
# ===================================================================


async def run_task(
    task: Any,  # harness.task_loader.Task
    model_spec: str,
    registry: MCPToolRegistry | None = None,
) -> CompletionResult:
    """Run a single task through the model with tool access.

    Args:
        task: A Task object with .task_id, .question, .suggested_tools.
        model_spec: Model specification string (e.g. "openai::http://..::model").
        registry: MCPToolRegistry instance (created if not provided).

    Returns:
        CompletionResult with all execution details.
    """
    if registry is None:
        registry = MCPToolRegistry()

    tool_schemas = registry.get_tool_schemas(task.suggested_tools)
    backend = create_backend(model_spec, registry, tool_schemas)

    logger.info("Running task %s with %s (%d tools)", task.task_id, model_spec, len(tool_schemas))
    result = await backend.run(task.question)
    result.task_id = task.task_id
    return result


async def run_tasks(
    tasks: list[Any],
    model_spec: str,
    workers: int = 4,
    registry: MCPToolRegistry | None = None,
) -> list[CompletionResult]:
    """Run multiple tasks with bounded concurrency.

    Args:
        tasks: List of Task objects.
        model_spec: Model specification string.
        workers: Maximum concurrent tasks.
        registry: Shared MCPToolRegistry instance.

    Returns:
        List of CompletionResult in the same order as tasks.
    """
    if registry is None:
        registry = MCPToolRegistry()

    semaphore = asyncio.Semaphore(workers)
    results: list[CompletionResult | None] = [None] * len(tasks)

    async def _worker(idx: int, task: Any) -> None:
        async with semaphore:
            try:
                result = await run_task(task, model_spec, registry)
                results[idx] = result
            except Exception as exc:
                logger.error("Task %s failed: %s", task.task_id, exc)
                results[idx] = CompletionResult(
                    task_id=task.task_id,
                    model_answer=f"[ERROR] {exc}",
                    tool_calls=[],
                    trace=[{"error": str(exc)}],
                    tokens={"prompt": 0, "completion": 0},
                    wall_time=0.0,
                )

    await asyncio.gather(*[_worker(i, t) for i, t in enumerate(tasks)])

    completed = sum(1 for r in results if r and not r.model_answer.startswith("[ERROR]"))
    logger.info("Completed %d/%d tasks successfully", completed, len(tasks))

    return [r for r in results if r is not None]
