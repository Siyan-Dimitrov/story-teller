"""Shared Claude text-generation helper.

Every text-LLM call in the pipeline (script passes, image classification,
chapter tagging, cast bible, export metadata, music suggestions, chapter
splitting) goes through :func:`complete`, which drives the Claude Agent SDK.
The SDK authenticates against the user's Claude Code OAuth credentials
(``~/.claude/.credentials.json``), so no Anthropic API key is required when
the user is signed into Claude Code.

Open-source model support (Ollama/Kimi) lives on the ``open-source-models``
branch; master is Claude-only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

import re

from . import config

log = logging.getLogger(__name__)


class ClaudeAuthError(RuntimeError):
    """Raised when the Claude Agent SDK cannot authenticate."""


class ClaudeBackendError(RuntimeError):
    """Raised when a Claude call fails for a non-auth reason."""


async def complete(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    pass_name: str = "llm",
    timeout: float | None = None,
) -> str:
    """Run one Claude text completion. Returns the response text."""
    text, _cost = await complete_with_cost(
        system_prompt,
        user_prompt,
        model=model,
        pass_name=pass_name,
        timeout=timeout,
    )
    return text


async def complete_with_cost(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str | None = None,
    pass_name: str = "llm",
    timeout: float | None = None,
) -> tuple[str, float]:
    """Run one Claude text completion. Returns (text, notional_cost_usd).

    On Windows the FastAPI process runs under ``WindowsSelectorEventLoopPolicy``
    (see ``backend/main.py``), and that loop cannot spawn subprocesses — but
    the SDK shells out to ``claude.exe``. We bounce the SDK call into a
    worker thread with its own ``ProactorEventLoop`` so the subprocess
    transport works without changing the parent loop policy.
    """
    try:
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            query,
            AssistantMessage,
            ResultMessage,
            TextBlock,
        )
    except ImportError as e:
        raise ClaudeBackendError(
            "claude-agent-sdk is not installed. Run `pip install -r requirements.txt`."
        ) from e

    resolved_model = (model or config.CLAUDE_MODEL).strip()
    resolved_timeout = timeout or config.CLAUDE_TIMEOUT_SECONDS

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        model=resolved_model,
        max_turns=1,
        allowed_tools=[],
        disallowed_tools=[
            "Read", "Write", "Edit", "Bash", "Glob", "Grep",
            "WebFetch", "WebSearch", "TaskCreate", "TaskUpdate", "TaskList",
            "NotebookEdit",
        ],
        permission_mode="bypassPermissions",
        setting_sources=[],
    )

    async def _do_call() -> tuple[str, float]:
        chunks: list[str] = []
        cost_usd = 0.0
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        chunks.append(block.text)
            elif isinstance(message, ResultMessage):
                cost_usd = float(getattr(message, "total_cost_usd", 0.0) or 0.0)
        return "".join(chunks).strip(), cost_usd

    def _thread_run() -> tuple[str, float]:
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                asyncio.wait_for(_do_call(), timeout=resolved_timeout)
            )
        finally:
            loop.close()

    try:
        text, cost_usd = await asyncio.to_thread(_thread_run)
    except asyncio.TimeoutError as e:
        raise ClaudeBackendError(
            f"Claude {pass_name} pass timed out after {resolved_timeout:.0f}s"
        ) from e
    except Exception as e:
        msg = str(e).lower()
        if "credential" in msg or "unauthorized" in msg or "auth" in msg or "login" in msg:
            raise ClaudeAuthError(
                "Claude Agent SDK could not authenticate. Run `claude login` once "
                "to sign in with your Claude Code subscription, then retry."
            ) from e
        raise ClaudeBackendError(f"Claude {pass_name} pass failed: {e}") from e

    if not text:
        raise ClaudeBackendError(f"Claude {pass_name} pass returned no text content")
    return text, cost_usd


# ── JSON extraction helpers ──────────────────────────────────

def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


def extract_first_json_object(text: str) -> str:
    """Slice out the first top-level ``{...}`` block, in case the model
    surrounds it with stray prose. We still fail fast on real garbage."""
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


def repair_truncated_json(raw: str) -> str:
    """Attempt to repair JSON truncated by token limits.

    Closes any open strings, arrays, and objects so json.loads can parse
    whatever complete content the LLM managed to emit.
    """
    # Remove any trailing partial escape sequence
    raw = re.sub(r'\\$', '', raw.rstrip())

    in_string = False
    escape = False
    stack: list[str] = []  # tracks open { and [

    for ch in raw:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    # Close the open string if we're inside one
    if in_string:
        raw += '"'

    # Close open containers in reverse order
    for opener in reversed(stack):
        raw += ']' if opener == '[' else '}'

    return raw


def parse_json(raw: str) -> dict[str, Any]:
    """Parse model output as JSON. Tolerates code fences and a trailing
    truncation. Raises ``ClaudeBackendError`` on unrecoverable garbage."""
    text = strip_code_fences(raw)
    text = extract_first_json_object(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("Claude JSON parse failed (%s) — attempting truncation repair", e)
        try:
            return json.loads(repair_truncated_json(text))
        except json.JSONDecodeError as e2:
            preview = text[:400].replace("\n", " ")
            raise ClaudeBackendError(
                f"Claude returned malformed JSON: {e2}. Preview: {preview!r}"
            ) from e2
