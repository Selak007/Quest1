"""
Stage 6 — Fuzzy text matching via RapidFuzz.

Matches the user-supplied target phrase against the forced-aligned
word list. Tolerant of ASR typos (e.g., "stagnatlon" vs "stagnation").

Design decisions:
- Match is performed at the WORD SEQUENCE level using a sliding window.
- "Earliest occurrence" wins when the phrase appears multiple times.
- Score is the RapidFuzz token_sort_ratio (0–100), normalised to 0–1.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from rapidfuzz import fuzz

from . import config
from .alignment import AlignedWord

logger = logging.getLogger(__name__)


def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class MatchResult:
    """Result of a successful fuzzy phrase match."""
    onset_s: float           # start time of the first matched word
    offset_s: float          # end time of the last matched word
    matched_text: str        # what the ASR actually said (may differ from target)
    target_text: str         # the original target phrase
    score: float             # RapidFuzz match quality 0–1
    matched_words: List[AlignedWord]  # the specific word objects matched


def find_phrase(
    target: str,
    aligned_words: List[AlignedWord],
    score_cutoff: float = None,
) -> Optional[MatchResult]:
    """
    Find the EARLIEST occurrence of *target* in *aligned_words*.

    Uses a sliding window of exactly len(target_words) words scored with
    RapidFuzz token_sort_ratio.  WhisperX gap-filling ensures words dropped
    by forced alignment (e.g. short standalone segments like "Victor...") are
    restored, so the exact-width window reliably finds cross-segment phrases.
    """
    if score_cutoff is None:
        score_cutoff = config.FUZZY_SCORE_CUTOFF

    if not aligned_words:
        logger.warning("No aligned words supplied — cannot match.")
        return None

    norm_target = _normalise(target)
    target_word_count = len(norm_target.split())

    if target_word_count > len(aligned_words):
        target_word_count = len(aligned_words)

    best: Optional[MatchResult] = None
    best_score: float = -1.0          # track true best, even below cutoff
    best_score_above_cutoff: float = -1.0

    for i in range(len(aligned_words) - target_word_count + 1):
        window = aligned_words[i : i + target_word_count]
        window_text = " ".join(w.word for w in window)
        norm_window = _normalise(window_text)

        score = fuzz.token_sort_ratio(norm_target, norm_window)
        if score > best_score:
            best_score = score

        if score >= score_cutoff and score > best_score_above_cutoff:
            best_score_above_cutoff = score
            best = MatchResult(
                onset_s=window[0].start,
                offset_s=window[-1].end,
                matched_text=window_text,
                target_text=target,
                score=score / 100.0,
                matched_words=window,
            )

    if best is None:
        logger.warning(
            "No match found for target=%r (best score seen: %.1f, cutoff: %.1f).",
            target,
            best_score,
            score_cutoff,
        )
    else:
        logger.info(
            "Match found: %r at %.3fs (score=%.2f).",
            best.matched_text,
            best.onset_s,
            best.score,
        )

    return best

