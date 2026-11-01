"""Indirect prompt injection quarantine.

Retrieved text is data, never instructions. Two layers:

1. Chunks that contain instruction-like text aimed at a model ("ignore previous
   instructions", "you are now ...") are quarantined: kept out of the answer context and
   reported in the trace, so an analyst can see what was withheld and why.
2. Everything that does reach a model is fenced inside numbered source blocks with an
   explicit instruction to treat it as quoted material (see `generation.py`).

Pattern matching will not catch every injection. It is a cheap first filter in front of
the structural defence, and the eval keeps both honest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agentic_rag.models import Hit

_PATTERNS = [
    r"ignore (?:all |any )?(?:the )?(?:previous|prior|above) instructions",
    r"disregard (?:all |any )?(?:the )?(?:previous|prior|above|your) (?:instructions|rules)",
    r"you are now (?:in )?\w+",
    r"\bsystem prompt\b",
    r"\bnote to (?:ai|llm|the assistant|ai systems)\b",
    r"\bmaintenance mode\b",
    r"\bdo not tell the user\b",
    r"\bact as (?:an?|the) \w+",
    r"\bnew instructions?:",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in _PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class Screened:
    kept: list[Hit]
    quarantined: list[Hit]


def looks_like_injection(text: str) -> bool:
    return _INJECTION.search(text) is not None


def screen(hits: list[Hit]) -> Screened:
    kept: list[Hit] = []
    quarantined: list[Hit] = []
    for h in hits:
        if looks_like_injection(f"{h.chunk.section}. {h.chunk.text}"):
            quarantined.append(h)
        else:
            kept.append(h)
    return Screened(kept=kept, quarantined=quarantined)
