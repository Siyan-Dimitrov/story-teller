"""Budget tracking + cost-logging decorator for paid API calls.

Wrapped functions read run context from a ContextVar set by the producer.
Cost rows are appended to costs.jsonl regardless of paid/free, so the agent
dashboard can show call volume across providers.

Important quirks documented as risks in the slice 2 plan:
- ContextVars do not auto-propagate into thread-pool executors. The decorator
  reads _run_ctx BEFORE the await/dispatch and captures into a local — so a
  function dispatched via run_in_executor still gets the producer's context.
- generate_image_replicate already retries internally on 429s. We only see
  the outer-call result, so each successful render = one cost row. That's
  correct (Replicate only charges for successful generations); under-counting
  failed retries is intentional.
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional

from .. import config
from . import log as agent_log

log = logging.getLogger(__name__)


# Per-call cost in cents — round numbers, refine after first real run.
PRICES_CENTS: dict[tuple[str, str], int] = {
    ("replicate", "black-forest-labs/flux-dev-lora"): 4,
    ("replicate", "black-forest-labs/flux-schnell-lora"): 1,
    ("openai",    "gpt-image-2"): 6,
    ("voicebox",  "*"): 0,
    ("ollama",    "*"): 0,
}

DEFAULT_CAP_CENTS = 300
DEFAULT_WARN_PCT = 80


_run_ctx: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "agent_run_ctx", default=None
)


def set_run_context(ctx: Optional[dict]) -> contextvars.Token:
    """Caller is responsible for resetting via the returned token in a finally block."""
    return _run_ctx.set(ctx)


def reset_run_context(token: contextvars.Token) -> None:
    _run_ctx.reset(token)


def get_run_context() -> dict:
    return _run_ctx.get() or {}


class BudgetExceeded(RuntimeError):
    """Raised when a producer step would exceed the group's cap_cents."""


@dataclass
class BudgetStatus:
    group_id: str
    used_cents: int
    cap_cents: int
    warn_pct: int
    ok: bool
    percent: float

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_cents(provider: str, model: str, image_count: int = 1) -> int:
    rate = PRICES_CENTS.get((provider, model))
    if rate is None:
        rate = PRICES_CENTS.get((provider, "*"), 0)
    return rate * max(1, image_count)


def cost_logged(provider: str, model_attr: Optional[str] = None, *, agent_default: str = "producer"):
    """Decorator that appends a costs.jsonl row after the wrapped call returns.

    `model_attr` is the name of a config attribute holding the model id (e.g.
    "REPLICATE_MODEL"). Pass None for providers without a single model handle
    (voicebox); the row will use "n/a".

    Each row carries the run_id from the calling producer context so the budget
    gate can scope spending to the current run rather than the group's full
    history (a second run on the same group would otherwise pre-burn its cap).
    """

    def deco(fn):
        @functools.wraps(fn)
        async def wrap(*args, **kwargs):
            # Capture context BEFORE the await — survives executor dispatch.
            ctx = dict(get_run_context())
            ok = True
            try:
                return await fn(*args, **kwargs)
            except Exception:
                ok = False
                raise
            finally:
                model = (
                    getattr(config, model_attr, "unknown")
                    if model_attr
                    else "n/a"
                )
                try:
                    agent_log.append(
                        agent_log.COSTS_PATH,
                        {
                            "ts": agent_log.now_iso(),
                            "project_id": ctx.get("project_id"),
                            "group_id": ctx.get("group_id"),
                            "run_id": ctx.get("run_id"),
                            "agent": ctx.get("agent", agent_default),
                            "provider": provider,
                            "model": model,
                            "image_count": 1,
                            "cents": estimate_cents(provider, model, 1) if ok else 0,
                            "ok": ok,
                        },
                    )
                except Exception as logerr:  # never let logging break the caller
                    log.warning(f"cost_logged: failed to append row: {logerr}")

        return wrap

    return deco


# ── Budget policy storage ────────────────────────────────────────

def _read_policies_raw() -> dict:
    if not agent_log.BUDGET_POLICIES_PATH.exists():
        return {}
    try:
        return json.loads(agent_log.BUDGET_POLICIES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.warning(f"budget_policies.json malformed; treating as empty: {e}")
        return {}


def get_policy(group_id: str) -> dict:
    raw = _read_policies_raw()
    if group_id in raw:
        return {**_default_policy(), **raw[group_id]}
    if "_default" in raw:
        return {**_default_policy(), **raw["_default"]}
    return _default_policy()


def _default_policy() -> dict:
    return {"cap_cents": DEFAULT_CAP_CENTS, "warn_pct": DEFAULT_WARN_PCT}


def set_policy(group_id: str, *, cap_cents: int, warn_pct: int = DEFAULT_WARN_PCT) -> dict:
    raw = _read_policies_raw()
    raw[group_id] = {"cap_cents": int(cap_cents), "warn_pct": int(warn_pct)}
    agent_log.BUDGET_POLICIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    agent_log.BUDGET_POLICIES_PATH.write_text(
        json.dumps(raw, indent=2), encoding="utf-8"
    )
    return raw[group_id]


def used_cents_for_group(group_id: str, *, run_id: Optional[str] = None) -> int:
    """Sum ok=True cents from costs.jsonl for this group.

    When run_id is provided, only rows tagged with that run_id are counted —
    this is what the budget gate uses during a producer run so a previously-
    completed run doesn't pre-burn the cap on the same group. When run_id is
    None, the full group history is summed (used by the dashboard view).
    """
    total = 0
    for row in agent_log.read_all(agent_log.COSTS_PATH):
        if row.get("group_id") != group_id:
            continue
        if not row.get("ok", True):
            continue
        if run_id is not None and row.get("run_id") != run_id:
            continue
        total += int(row.get("cents", 0))
    return total


def check_budget(group_id: str, *, run_id: Optional[str] = None) -> BudgetStatus:
    """Compute the current budget status.

    Pass run_id from the active producer context to scope spending to the
    current run. Omitting run_id reports cumulative group history (UI use).
    """
    policy = get_policy(group_id)
    # If caller didn't specify, try to read run_id off the active context so
    # the producer's own ensure_budget_or_raise call is automatically scoped.
    if run_id is None:
        run_id = get_run_context().get("run_id")
    used = used_cents_for_group(group_id, run_id=run_id)
    cap = int(policy["cap_cents"])
    pct = (used / cap * 100.0) if cap > 0 else 0.0
    return BudgetStatus(
        group_id=group_id,
        used_cents=used,
        cap_cents=cap,
        warn_pct=int(policy["warn_pct"]),
        ok=used < cap,
        percent=round(pct, 1),
    )


def ensure_budget_or_raise(group_id: Optional[str]) -> None:
    """Raise BudgetExceeded if the current run is over cap. No-op if group_id is None.

    Scoped to the active producer run via the contextvar's run_id.
    """
    if not group_id:
        return
    status = check_budget(group_id)  # picks up run_id from context automatically
    if not status.ok:
        raise BudgetExceeded(
            f"Group {group_id} budget exceeded: {status.used_cents}¢ used of {status.cap_cents}¢ cap"
        )
