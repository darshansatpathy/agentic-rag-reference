"""Tokenisation (Snowball English stemming), sentence splitting and answer-type cues."""

from __future__ import annotations

import re
from functools import lru_cache

import snowballstemmer

STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from into onto about as is are
    was were be been being do does did doing have has had having i me my we our you your he she
    it its they them their this that these those there here what which who whom whose when where
    why how can could should would may might must will shall not no nor so than too very just
    also any all each every some such only own same other more most per up down out over under
    again further once get gets got am i'm im dont don't does doesn't many much
    """.split()  # noqa: SIM905
)

_NUMBER_WITH_SEPARATORS = re.compile(r"(?<=\d)[,](?=\d{3}\b)")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def normalise(text: str) -> str:
    """Lowercase and drop thousands separators so '3,000' and '3000' match."""
    return _NUMBER_WITH_SEPARATORS.sub("", text.lower())


_STEMMER = snowballstemmer.stemmer("english")


@lru_cache(maxsize=65536)
def stem(token: str) -> str:
    """Snowball (Porter 2) stem, so approve/approved/approval and day/days match."""
    if any(c.isdigit() for c in token) or len(token) <= 3:
        return token
    return str(_STEMMER.stemWord(token))


def tokens(text: str, *, keep_stopwords: bool = False) -> list[str]:
    raw = _TOKEN.findall(normalise(text))
    return [stem(t) for t in raw if (len(t) > 1 or t.isdigit()) and (keep_stopwords or t not in STOPWORDS)]


def numbers(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(normalise(text)) if any(c.isdigit() for c in t)}


def sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE.split(text.strip())]
    return [p for p in parts if p]


# ------------------------------------------------------------------ answer-type cues

_WHEN = re.compile(r"^\s*(when|by when|until when|what date|which (?:date|day|year|month))\b", re.I)
_QUANTITY = re.compile(
    r"\bhow (many|much|long|fast|often|far|big|large|soon)\b|\b(minimum|maximum|limit|rate|"
    r"threshold|percentage|number of)\b",
    re.I,
)
_TEMPORAL = re.compile(
    r"\b(19|20)\d{2}\b|\bq[1-4]\b|\b\d{1,2}:\d{2}\b|"
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|today|tomorrow|"
    r"minutes?|hours?|days?|weeks?|months?|years?|quarters?|daily|weekly|monthly|quarterly|"
    r"annually|yearly|within|before|after|until|by the end|every|immediately)\b",
    re.I,
)
_NUMBER_WORDS = re.compile(
    r"\b(\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"fifteen|twenty|thirty|forty|fifty|hundred|thousand|half|twice|once|percent)\b",
    re.I,
)


def answer_type(question: str) -> str | None:
    """Coarse expected answer type: 'time' for when-questions, 'quantity' for how-much/many."""
    if _WHEN.search(question):
        return "time"
    if _QUANTITY.search(question):
        return "quantity"
    return None


def has_temporal(text: str) -> bool:
    return bool(_TEMPORAL.search(text))


def has_quantity(text: str) -> bool:
    return bool(_NUMBER_WORDS.search(text))
