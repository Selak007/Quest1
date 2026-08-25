"""
Shared pytest fixtures and helpers for the dialogue_locator test suite.
"""
from __future__ import annotations

import pytest
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_aligned_words(words_with_times: list[tuple[str, float, float]]):
    """
    Convenience factory.
    words_with_times: list of (word, start_s, end_s)
    """
    from dialogue_locator.alignment import AlignedWord
    return [
        AlignedWord(word=w, start=s, end=e, score=1.0)
        for w, s, e in words_with_times
    ]


def make_frame_pts(pts_values: list[float]):
    """
    Convenience factory: enumerate a list of PTS floats → List[FramePTS].
    """
    from dialogue_locator.frame_mapping import FramePTS
    return [FramePTS(frame_number=i, pts_s=v) for i, v in enumerate(pts_values)]


def make_transcript_segment(text: str, start: float, end: float, avg_logprob: float = -0.3):
    """
    Convenience factory for TranscriptSegment.
    """
    from dialogue_locator.asr import TranscriptSegment
    return TranscriptSegment(
        text=text,
        start=start,
        end=end,
        avg_logprob=avg_logprob,
        words=[],
    )


def make_speech_segment(start: float, end: float):
    from dialogue_locator.vad import SpeechSegment
    return SpeechSegment(start_s=start, end_s=end)
