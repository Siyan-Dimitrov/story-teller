"""Shared agent primitives — verdict shape and a small Ollama JSON helper."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional

import httpx

from .. import config

log = logging.getLogger(__name__)


Severity = Literal["ok", "minor", "major", "fatal"]
"""ok = ship · minor = note · major = regenerate · fatal = the run is broken"""


@dataclass
class Issue:
    kind: str                         # e.g. "truncation", "empty_image_prompts"
    severity: Severity
    description: str
    scene_index: Optional[int] = None
    suggested_fix: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentVerdict:
    agent: str
    accept: bool                      # ship as-is?
    severity: Severity                # worst severity across issues
    issues: list[Issue] = field(default_factory=list)
    feedback: str = ""                # human/LLM-readable instruction for the next attempt
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "accept": self.accept,
            "severity": self.severity,
            "issues": [i.to_dict() for i in self.issues],
            "feedback": self.feedback,
            "metadata": self.metadata,
        }


_SEVERITY_RANK = {"ok": 0, "minor": 1, "major": 2, "fatal": 3}


def worst_severity(issues: list[Issue]) -> Severity:
    if not issues:
        return "ok"
    return max(issues, key=lambda i: _SEVERITY_RANK[i.severity]).severity


class LLMCallError(RuntimeError):
    """Raised when the LLM response can't be parsed as the requested JSON shape."""


_JSON_BLOCK_RE = re.compile(r"\{.*\}|\[.*\]", re.DOTALL)


def _extract_json(text: str) -> str:
    """Strip common LLM wrappers and return the first JSON object/array substring."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = _JSON_BLOCK_RE.search(text)
    return match.group(0) if match else text


async def call_llm_json(
    *,
    system: str,
    user: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> dict:
    """Call Ollama and return the parsed JSON payload.

    Lower default temperature than the writer prompts — the critic should be
    consistent, not creative. Raises LLMCallError if the model's reply isn't
    parseable JSON after one trim pass.
    """
    payload = {
        "model": model or config.OLLAMA_MODEL,
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(f"{config.OLLAMA_URL}/api/chat", json=payload)
        if resp.status_code != 200:
            raise LLMCallError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
    raw = (body.get("message") or {}).get("content", "")
    if not raw:
        raise LLMCallError("Ollama returned empty content")
    try:
        return json.loads(_extract_json(raw))
    except json.JSONDecodeError as e:
        raise LLMCallError(f"Could not parse JSON from LLM: {e}; raw[:300]={raw[:300]!r}")
