"""
Stage 8 — Composite confidence scoring.

Combines three independent signals into a single float in [0, 1]:

  text_score    (weight 0.50)
    RapidFuzz match ratio of target phrase vs. ASR-recognised text.
    Perfect match = 1.0; completely wrong = 0.0.

  asr_quality   (weight 0.30)
    Mean segment avg_logprob from Whisper, normalised to [0, 1].
    Whisper logprobs are typically in [-2, 0]; we clip and rescale.

  vad_agreement (weight 0.20)
    1.0  → VAD silence→speech transition within corroboration window
    0.5  → No VAD transition found near onset (ambiguous)
    0.3  → VAD indicates continuous speech / no transition (mild disagreement)

Status categories:
  ≥ 0.80  → "high"
  ≥ 0.60  → "medium"
  ≥ 0.40  → "low"
  < 0.40  → "best_effort"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List

from . import config
from .asr import TranscriptSegment

logger = logging.getLogger(__name__)

# Logprob normalisation: Whisper good segments ≈ -0.5 to 0; bad ≈ < -1.5
_LOGPROB_BEST = -0.2   # maps to 1.0
_LOGPROB_WORST = -2.0  # maps to 0.0


@dataclass
class ConfidenceBreakdown:
    """Full breakdown of how the confidence score was computed."""
    text_score: float
    asr_quality: float
    vad_agreement: float
    composite: float
    status: str


def _normalise_logprob(avg_logprob: float) -> float:
    """Map avg_logprob from [_LOGPROB_WORST, _LOGPROB_BEST] → [0, 1]."""
    clipped = max(_LOGPROB_WORST, min(_LOGPROB_BEST, avg_logprob))
    return (clipped - _LOGPROB_WORST) / (_LOGPROB_BEST - _LOGPROB_WORST)


def _status(score: float) -> str:
    if score >= config.CONF_HIGH:
        return "high"
    if score >= config.CONF_MEDIUM:
        return "medium"
    if score >= config.CONF_LOW:
        return "low"
    return "best_effort"


def compute_confidence(
    text_match_score: float,          # RapidFuzz ratio 0–1
    matched_segments: List[TranscriptSegment],
    vad_agrees: bool,
    vad_transition_found: bool,
) -> ConfidenceBreakdown:
    """
    Compute composite confidence from the three independent signals.

    Parameters
    ----------
    text_match_score:
        RapidFuzz score from matcher.py, already normalised to [0, 1].
    matched_segments:
        The Whisper TranscriptSegment(s) that contain the matched phrase,
        used to derive ASR quality from avg_logprob.
    vad_agrees:
        True if a VAD silence→speech transition was found within
        config.VAD_CORROBORATION_WINDOW_S of the onset.
    vad_transition_found:
        True if ANY VAD transition was found (even if outside the window).
        Used to distinguish "no transition at all" from "wrong transition".

    Returns
    -------
    ConfidenceBreakdown
    """
    # ── Text score ────────────────────────────────────────────────────────────
    text_score = float(text_match_score)

    # ── ASR quality ───────────────────────────────────────────────────────────
    if matched_segments:
        mean_lp = sum(s.avg_logprob for s in matched_segments) / len(matched_segments)
        asr_quality = _normalise_logprob(mean_lp)
    else:
        asr_quality = 0.5  # no segments to judge → neutral

    # ── VAD agreement ─────────────────────────────────────────────────────────
    if vad_agrees:
        vad_agreement = 1.0
    elif not vad_transition_found:
        # VAD didn't see a transition at all — music? overlapping speech?
        vad_agreement = 0.5
    else:
        # A VAD transition exists but it's far from our onset → mild disagreement
        vad_agreement = 0.3

    # ── Composite ─────────────────────────────────────────────────────────────
    composite = (
        config.CONF_WEIGHT_TEXT * text_score
        + config.CONF_WEIGHT_ASR * asr_quality
        + config.CONF_WEIGHT_VAD * vad_agreement
    )
    composite = round(max(0.0, min(1.0, composite)), 4)

    status = _status(composite)

    logger.info(
        "Confidence: text=%.2f asr=%.2f vad=%.2f → composite=%.3f (%s)",
        text_score,
        asr_quality,
        vad_agreement,
        composite,
        status,
    )

    return ConfidenceBreakdown(
        text_score=text_score,
        asr_quality=asr_quality,
        vad_agreement=vad_agreement,
        composite=composite,
        status=status,
    )
