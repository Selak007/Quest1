"""
Stage 5 — Forced Alignment via WhisperX (wav2vec2 CTC).

Refines the rough per-word timestamps from faster-whisper to ~20–40ms
accuracy by re-aligning the ASR transcript against the audio waveform.

WhisperX forced alignment is the PRIMARY timing source in v2.
No separate custom onset detector is needed — the aligner's output is
trusted directly as the word onset.

Fallback: if WhisperX cannot be imported (Python version constraint) or
errors during alignment, the module automatically falls back to
faster-whisper's own per-word timestamps (~50–100ms accuracy).
The rest of the pipeline is unaffected — AlignedWord objects are
produced identically by both paths.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from . import config
from .asr import TranscriptSegment, WordToken

logger = logging.getLogger(__name__)

_align_model = None
_align_metadata = None

# Try importing whisperx once at module load; record whether it's available
try:
    import whisperx as _whisperx_module
    _WHISPERX_AVAILABLE = True
    logger.debug("whisperx imported successfully.")
except Exception as _wx_import_err:
    _WHISPERX_AVAILABLE = False
    logger.warning(
        "whisperx not available (%s) — will use faster-whisper word timestamps "
        "as fallback (still accurate to ~50–100ms).",
        _wx_import_err,
    )


@dataclass
class AlignedWord:
    """A word with a precision-aligned onset/offset."""
    word: str
    start: float   # seconds
    end: float     # seconds
    score: float   # alignment confidence (0–1)
    source: str = "whisperx"  # "whisperx" | "faster-whisper"


def _load_align_model(language_code: str = "en", device: str = "cpu"):
    """Lazy-load the WhisperX alignment model."""
    global _align_model, _align_metadata
    if _align_model is not None:
        return
    logger.info("Loading WhisperX alignment model for language='%s' …", language_code)
    _align_model, _align_metadata = _whisperx_module.load_align_model(
        language_code=language_code,
        device=device,
    )
    logger.info("WhisperX alignment model loaded.")


def _align_with_whisperx(
    wav_path: Path,
    transcript_segments: List[TranscriptSegment],
    language: str,
    device: str,
    win_start: float = 0.0,
    win_end: Optional[float] = None,
) -> List[AlignedWord]:
    """Primary path: WhisperX forced alignment on sliced audio."""
    _load_align_model(language_code=language, device=device)

    import soundfile as sf
    sr = 16000
    if win_end is not None:
        start_sample = int(win_start * sr)
        frames_to_read = int((win_end - win_start) * sr)
        try:
            audio_array, _ = sf.read(str(wav_path), start=start_sample, frames=frames_to_read, dtype="float32")
            shift = win_start
        except Exception as read_err:
            logger.warning(
                "WhisperX sliced audio read failed (%s) - falling back to full audio.",
                read_err,
            )
            audio_array, _ = sf.read(str(wav_path), dtype="float32")
            shift = 0.0
    else:
        audio_array, _ = sf.read(str(wav_path), dtype="float32")
        shift = 0.0

    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    # Build wx_segments shifted relative to the slice start (0.0)
    wx_segments = [
        {
            "text": seg.text,
            "start": seg.start - shift,
            "end": seg.end - shift,
            "words": [
                {"word": w.word, "start": w.start - shift, "end": w.end - shift}
                for w in seg.words
            ],
        }
        for seg in transcript_segments
    ]

    logger.info("Running WhisperX forced alignment on %d segment(s) …", len(wx_segments))
    result = _whisperx_module.align(
        wx_segments,
        _align_model,
        _align_metadata,
        audio_array,
        device,
        return_char_alignments=False,
    )

    aligned_words: List[AlignedWord] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []):
            start = w.get("start")
            end = w.get("end")
            if start is None or end is None:
                continue
            aligned_words.append(
                AlignedWord(
                    word=w.get("word", "").strip(),
                    start=float(start) + shift,
                    end=float(end) + shift,
                    score=float(w.get("score", 0.0)),
                    source="whisperx",
                )
            )

    aligned_words.sort(key=lambda w: w.start)
    logger.info("WhisperX alignment produced %d words.", len(aligned_words))
    return aligned_words


def _align_with_faster_whisper(
    transcript_segments: List[TranscriptSegment],
) -> List[AlignedWord]:
    """
    Fallback path: use per-word timestamps already produced by faster-whisper.

    faster-whisper word timestamps are obtained from the CTC decoder and are
    typically accurate to ~50–100ms — sufficient for most use cases.
    """
    logger.info(
        "Building faster-whisper word timestamp list (%d segment(s)) …",
        len(transcript_segments),
    )
    aligned_words: List[AlignedWord] = []
    for seg in transcript_segments:
        for w in seg.words:
            if w.start is None or w.end is None:
                continue
            aligned_words.append(
                AlignedWord(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    score=w.probability,
                    source="faster-whisper",
                )
            )
    aligned_words.sort(key=lambda w: w.start)
    logger.info("Fallback produced %d words from faster-whisper.", len(aligned_words))
    return aligned_words


def _fill_gaps(
    whisperx_words: List[AlignedWord],
    fw_words: List[AlignedWord],
    gap_threshold_s: float = 0.3,
) -> List[AlignedWord]:
    """
    Supplement WhisperX aligned words with faster-whisper words that fall in
    time gaps — i.e. spans of >= gap_threshold_s with no WhisperX entry.

    This recovers words that WhisperX silently drops (e.g. very short segments,
    ellipsis-heavy words, or segments where backtracking fails).
    """
    if not fw_words:
        return whisperx_words

    merged = list(whisperx_words)

    for fw_word in fw_words:
        # Check if any WhisperX word overlaps with this faster-whisper word
        overlaps = any(
            abs(wx.start - fw_word.start) < gap_threshold_s
            for wx in whisperx_words
        )
        if not overlaps:
            merged.append(fw_word)
            logger.debug(
                "Gap-filled word %r at %.3fs from faster-whisper.",
                fw_word.word, fw_word.start,
            )

    merged.sort(key=lambda w: w.start)
    added = len(merged) - len(whisperx_words)
    if added:
        logger.info(
            "Gap-filled %d word(s) from faster-whisper into WhisperX output.",
            added,
        )
    return merged


def align_transcript(
    wav_path: Path,
    transcript_segments: List[TranscriptSegment],
    language: str = None,
    device: str = None,
    win_start: float = 0.0,
    win_end: Optional[float] = None,
) -> List[AlignedWord]:
    """
    Run alignment on the transcript, using WhisperX if available, otherwise
    falling back to faster-whisper word timestamps.

    When WhisperX succeeds, any words it silently dropped (short segments,
    alignment backtrack failures) are recovered from faster-whisper timestamps
    via gap-filling, so the returned word list is always complete.

    Parameters
    ----------
    wav_path:
        Path to the mono 16 kHz WAV file.
    transcript_segments:
        Segments from faster-whisper (Stage 4).
    language:
        Language code (default: from config).
    device:
        'cpu' or 'cuda' (default: from config).
    win_start:
        Optional start sample time for sliced audio alignment.
    win_end:
        Optional end sample time for sliced audio alignment.

    Returns
    -------
    List[AlignedWord]
        All words in chronological order with aligned timestamps.
        Each AlignedWord.source indicates which backend produced it.
    """
    if not transcript_segments:
        logger.warning("No transcript segments to align — returning empty list.")
        return []

    lang = language or config.WHISPER_LANGUAGE
    dev = device or config.WHISPER_DEVICE

    # Always build the faster-whisper word list — used for gap-filling
    fw_words = _align_with_faster_whisper(transcript_segments)

    if _WHISPERX_AVAILABLE:
        try:
            wx_words = _align_with_whisperx(
                wav_path,
                transcript_segments,
                lang,
                dev,
                win_start=win_start,
                win_end=win_end,
            )
            # Fill any gaps WhisperX dropped with faster-whisper words
            return _fill_gaps(wx_words, fw_words)
        except Exception as exc:
            logger.warning(
                "WhisperX alignment failed (%s) — using faster-whisper word "
                "timestamps only.",
                exc,
            )

    return fw_words
