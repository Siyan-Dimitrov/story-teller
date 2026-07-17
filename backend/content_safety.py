"""Deterministic offensive-language scrub for generated scripts.

Public-domain sources (1800s–1930s pulp and folklore) routinely contain
racial slurs and dehumanizing period language. The screenwriter, critic,
and reviser prompts all instruct the model never to reproduce it; this
module is the guaranteed floor — a word-boundary scrub over every field
the pipeline narrates, draws, or prints, run once after script generation.
Replacements are neutral stand-ins for the same grammatical role; every
hit is logged and recorded on the script (``content_safety_notes``) so
the wording can be reviewed.

The scrub only catches unambiguous terms. Racist *framing* without slur
words (a race described as degraded, bestial, subhuman) can't be caught
by patterns — that is handled by the prompt-level instructions.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# (pattern, replacement) — unambiguous slurs and dehumanizing period terms,
# word-boundary and case-insensitive. "coon" excludes the raccoon senses.
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), r)
    for p, r in [
        (r"\bn[ie]gg?(?:er|ah|uh|a)s?\b", "man"),
        (r"\bnegroes\b", "Black people"),
        (r"\bnegro\b", "Black man"),
        (r"\bnegress(?:es)?\b", "Black woman"),
        (r"\bnigras?\b", "Black man"),
        (r"\bdark(?:y|ey|ie)s\b", "Black men"),
        (r"\bdark(?:y|ey|ie)\b", "Black man"),
        (r"\bpickaninn(?:y|ies)\b", "child"),
        (r"\bmulatt(?:oes|os|o|ress|resses)\b", "mixed-heritage"),
        (r"\bhalf-?breeds?\b", "mixed-heritage"),
        (r"\bhalf-?castes?\b", "mixed-heritage"),
        (r"\bcoons?\b(?!\s*(?:hound|dog|hunt|skin))", "man"),
        (r"\bchinaman\b", "Chinese man"),
        (r"\bchinamen\b", "Chinese men"),
        (r"\bjaps?\b", "Japanese"),
        (r"\binjuns?\b", "Native man"),
        (r"\bredskins?\b", "Native man"),
        (r"\bsquaws?\b", "Native woman"),
        (r"\bgyps(?:y|ies)\b", "Romani traveler"),
        (r"\bkikes?\b", "Jewish man"),
        (r"\bwetbacks?\b", "migrant"),
    ]
]


def scrub_text(text: str) -> tuple[str, list[str]]:
    """Return ``(clean_text, terms_removed)``; the text is unchanged if clean."""
    if not text:
        return text, []
    hits: list[str] = []

    for pat, repl in _RULES:
        def _note(m: re.Match, _repl: str = repl) -> str:
            hits.append(m.group(0))
            return _repl

        text = pat.sub(_note, text)
    return text, hits


def scrub_script(script: dict) -> list[str]:
    """Scrub every narrated/drawn/printed field of a script in place.

    Returns human-readable notes ("scene[3].narration: removed ...") —
    empty when the script was already clean.
    """
    notes: list[str] = []

    def _field(obj: dict, field: str, where: str) -> None:
        val = obj.get(field)
        if isinstance(val, str) and val:
            clean, hits = scrub_text(val)
            if hits:
                obj[field] = clean
                terms = ", ".join(sorted({h.lower() for h in hits}))
                notes.append(f"{where}.{field}: removed {terms}")

    for field in ("title", "synopsis", "visual_style"):
        _field(script, field, "script")

    for member in script.get("cast") or []:
        if isinstance(member, dict):
            where = f"cast[{member.get('id', '?')}]"
            for field in ("name", "description", "reference_prompt"):
                _field(member, field, where)

    for i, scene in enumerate(script.get("scenes") or []):
        if not isinstance(scene, dict):
            continue
        where = f"scene[{i}]"
        _field(scene, "narration", where)
        _field(scene, "image_prompt", where)
        prompts = scene.get("image_prompts")
        if isinstance(prompts, list):
            for j, p in enumerate(prompts):
                if isinstance(p, str) and p:
                    clean, hits = scrub_text(p)
                    if hits:
                        prompts[j] = clean
                        terms = ", ".join(sorted({h.lower() for h in hits}))
                        notes.append(f"{where}.image_prompts[{j}]: removed {terms}")
        for line in scene.get("lines") or []:
            if isinstance(line, dict):
                _field(line, "text", f"{where}.lines")

    return notes
