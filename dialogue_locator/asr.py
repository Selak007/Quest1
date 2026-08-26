"""
Stage 4 — Two-phase Automatic Speech Recognition.

PHASE 1 — SCAN  (tiny model, greedy, no word-timestamps)
  • Only processes VAD segments whose duration ≥ min_seg_duration_s
    (segments too short to contain the full target phrase are skipped).
  • Produces rough segment-level timestamps for fuzzy matching.
  • Logs every segment: [SCAN] N/M  start→end (Xs) : "text"

PHASE 2 — REFINE  (small model, beam=5, word-timestamps)
  • Runs only on a ±window_s slice of audio around the rough match.
  • Produces word-level timestamps for WhisperX alignment.
  • Logs every segment: [REFINE] N/M  start→end : "text"

This two-phase design cuts wall-clock time from ~40 min to ~3-5 min
for a 40-minute video on CPU.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import config
from .vad import SpeechSegment

logger = logging.getLogger(__name__)

# ── Per-model caches (separate so tiny stays in memory during scan) ───────────
_scan_model = None   # tiny / greedy
_refine_model = None  # small / beam


@dataclass
class WordToken:
    word: str
    start: float        # rough onset (seconds)
    end: float          # rough offset (seconds)
    probability: float  # per-word confidence from Whisper


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    avg_logprob: float
    words: List[WordToken] = field(default_factory=list)


# ── Model loaders ─────────────────────────────────────────────────────────────

def _load_scan_model():
    """Lazy-load the tiny model used for the fast scan pass."""
    global _scan_model
    if _scan_model is not None:
        return
    model_name = config.WHISPER_SCAN_MODEL
    logger.info(
        "[SCAN] Loading faster-whisper model '%s' (greedy, no word-timestamps) …",
        model_name,
    )
    from faster_whisper import WhisperModel
    _scan_model = WhisperModel(
        model_name,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    logger.info("[SCAN] Model '%s' loaded.", model_name)


def _load_refine_model():
    """Lazy-load the small model used for the precise refine pass."""
    global _refine_model
    if _refine_model is not None:
        return
    model_name = config.WHISPER_REFINE_MODEL
    logger.info(
        "[REFINE] Loading faster-whisper model '%s' (beam=5, word-timestamps) …",
        model_name,
    )
    from faster_whisper import WhisperModel
    _refine_model = WhisperModel(
        model_name,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    logger.info("[REFINE] Model '%s' loaded.", model_name)


def unload_scan_model():
    """Free the scan model from memory before loading the refine model."""
    global _scan_model
    if _scan_model is not None:
        logger.info("[SCAN] Unloading scan model to free RAM …")
        del _scan_model
        _scan_model = None
        try:
            import gc
            gc.collect()
        except Exception:
            pass


# ── Phase 1: SCAN ─────────────────────────────────────────────────────────────

def transcribe_scan(
    wav_path: Path,
    speech_segments: List[SpeechSegment],
    min_seg_duration_s: float = None,
) -> List[TranscriptSegment]:
    """
    Phase 1: Fast scan over VAD-filtered audio.

    Uses the tiny model with greedy decoding and no word-timestamps.
    Skips any segment whose duration < min_seg_duration_s — segments
    that are too short to contain the full target phrase at all.

    Parameters
    ----------
    wav_path:
        Path to mono 16 kHz WAV.
    speech_segments:
        Merged VAD segments from vad.py.
    min_seg_duration_s:
        Minimum segment duration to run ASR on. Segments shorter than
        this are skipped. Defaults to config.SCAN_MIN_SEG_DURATION_S.

    Returns
    -------
    List[TranscriptSegment]
        Segment-level transcripts (no per-word timestamps).
    """
    _load_scan_model()

    if min_seg_duration_s is None:
        min_seg_duration_s = config.SCAN_MIN_SEG_DURATION_S

    # Filter: only segments long enough to possibly contain the target phrase
    eligible = [s for s in speech_segments if s.duration_s >= min_seg_duration_s]
    skipped = len(speech_segments) - len(eligible)
    logger.info(
        "[SCAN] %d/%d VAD segments eligible (≥%.1fs); skipped %d too-short segments.",
        len(eligible), len(speech_segments), min_seg_duration_s, skipped,
    )

    if not eligible:
        logger.warning("[SCAN] No eligible segments — returning empty transcript.")
        return []

    # Retrieve audio duration for percentage display
    try:
        import soundfile as sf
        duration_s = sf.info(str(wav_path)).duration
    except Exception:
        duration_s = 0.0

    # Build flat clip_timestamps for faster-whisper
    flat_clips: list = []
    for seg in eligible:
        flat_clips.extend([seg.start_s, seg.end_s])

    kwargs = {
        "language": config.WHISPER_LANGUAGE,
        "word_timestamps": False,   # not needed in scan pass
        "beam_size": 1,             # greedy decoding — fastest
        "vad_filter": False,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,  # no context bleeding between clips
        "clip_timestamps": flat_clips,
    }

    logger.info(
        "[SCAN] Transcribing %d segments on '%s' (greedy, no word-timestamps) …",
        len(eligible), config.WHISPER_SCAN_MODEL,
    )
    segments_gen, _ = _scan_model.transcribe(str(wav_path), **kwargs)

    results: List[TranscriptSegment] = []
    seg_index = 0
    for seg in segments_gen:
        seg_index += 1
        pct = (seg.end / duration_s * 100.0) if duration_s > 0 else 0.0
        logger.info(
            "[SCAN] %d/%d  %.2fs→%.2fs (%.1fs long, %.1f%% done): %r",
            seg_index, len(eligible),
            seg.start, seg.end, seg.end - seg.start,
            pct,
            seg.text.strip(),
        )
        results.append(TranscriptSegment(
            text=seg.text.strip(),
            start=seg.start,
            end=seg.end,
            avg_logprob=seg.avg_logprob,
            words=[],  # no word timestamps in scan pass
        ))

    logger.info(
        "[SCAN] Complete — %d segment(s) transcribed.",
        len(results),
    )
    return results


# ── Phase 2: REFINE ───────────────────────────────────────────────────────────

def transcribe_refine(
    wav_path: Path,
    onset_s: float,
    window_s: float = None,
    win_start: float = None,
    win_end: float = None,
) -> List[TranscriptSegment]:
    """
    Phase 2: Precise re-transcription on a tight window around the match.

    Loads and transcribes only the sliced audio array for maximum speed.

    Parameters
    ----------
    wav_path:
        Path to mono 16 kHz WAV.
    onset_s:
        Rough onset from the scan pass (seconds).
    window_s:
        Half-width of the refine window (default: config.REFINE_WINDOW_S).
        A ±window_s clip is extracted, so total clip = 2×window_s.
    win_start:
        Optional pre-calculated window start sample time.
    win_end:
        Optional pre-calculated window end sample time.

    Returns
    -------
    List[TranscriptSegment]
        Segments with per-word timestamps for WhisperX alignment.
    """
    _load_refine_model()

    if window_s is None:
        window_s = config.REFINE_WINDOW_S

    if win_start is None or win_end is None:
        # Retrieve audio duration for clamping
        try:
            import soundfile as sf
            duration_s = sf.info(str(wav_path)).duration
        except Exception:
            duration_s = float("inf")
        win_start = max(0.0, onset_s - window_s)
        win_end   = min(duration_s, onset_s + window_s)

    logger.info(
        "[REFINE] Window: %.2fs → %.2fs (%.1fs total) around rough onset %.2fs",
        win_start, win_end, win_end - win_start, onset_s,
    )

    # Slice the audio file to process ONLY the 60s window
    sr = 16000
    start_sample = int(win_start * sr)
    frames_to_read = int((win_end - win_start) * sr)

    import soundfile as sf
    try:
        audio_slice, _ = sf.read(str(wav_path), start=start_sample, frames=frames_to_read, dtype="float32")
        if audio_slice.ndim > 1:
            audio_slice = audio_slice[:, 0]
    except Exception as read_err:
        logger.warning(
            "[REFINE] Sliced read failed (%s) - falling back to full audio path.",
            read_err,
        )
        audio_slice = str(wav_path)

    kwargs = {
        "language": config.WHISPER_LANGUAGE,
        "word_timestamps": True,    # need word-level for WhisperX
        "beam_size": 5,             # beam search for accuracy
        "vad_filter": False,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
    }

    # If we fell back to the full WAV path, pass clip_timestamps
    if isinstance(audio_slice, str):
        kwargs["clip_timestamps"] = [win_start, win_end]

    logger.info(
        "[REFINE] Transcribing window with '%s' (beam=5, word-timestamps) …",
        config.WHISPER_REFINE_MODEL,
    )
    segments_gen, _ = _refine_model.transcribe(audio_slice, **kwargs)

    results: List[TranscriptSegment] = []
    # If we sliced the audio array, we need to shift the returned timestamps back to absolute time
    shift = 0.0 if isinstance(audio_slice, str) else win_start

    for seg in segments_gen:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(WordToken(
                    word=w.word.strip(),
                    start=w.start + shift,
                    end=w.end + shift,
                    probability=w.probability,
                ))
        results.append(TranscriptSegment(
            text=seg.text.strip(),
            start=seg.start + shift,
            end=seg.end + shift,
            avg_logprob=seg.avg_logprob,
            words=words,
        ))
        logger.info(
            "[REFINE] %d  %.2fs→%.2fs: %r",
            len(results),
            seg.start + shift, seg.end + shift,
            seg.text.strip(),
        )

    logger.info(
        "[REFINE] Complete — %d segment(s), %d word(s).",
        len(results),
        sum(len(s.words) for s in results),
    )
    return results


# ── Legacy single-pass transcribe (kept for backward-compat / unit tests) ─────

_whisper_model = None  # legacy model cache


def transcribe(
    wav_path: Path,
    speech_segments: Optional[List[SpeechSegment]] = None,
) -> List[TranscriptSegment]:
    """
    [LEGACY] Single-pass transcription — used by unit tests.
    New code should use transcribe_scan() + transcribe_refine() instead.
    """
    global _whisper_model
    if _whisper_model is None:
        logger.info(
            "Loading faster-whisper model '%s' on device='%s' compute_type='%s' …",
            config.WHISPER_MODEL, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE,
        )
        from faster_whisper import WhisperModel
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        logger.info("faster-whisper model loaded.")

    kwargs: dict = {
        "language": config.WHISPER_LANGUAGE,
        "word_timestamps": True,
        "vad_filter": False,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "beam_size": 5,
    }

    if speech_segments:
        flat = []
        for seg in speech_segments:
            flat.extend([seg.start_s, seg.end_s])
        kwargs["clip_timestamps"] = flat

    try:
        import soundfile as sf
        duration_s = sf.info(str(wav_path)).duration
    except Exception:
        duration_s = 0.0

    logger.info("Transcribing audio (this may take a while for long videos) …")
    segments_gen, _ = _whisper_model.transcribe(str(wav_path), **kwargs)

    results: List[TranscriptSegment] = []
    for seg in segments_gen:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(WordToken(
                    word=w.word.strip(),
                    start=w.start,
                    end=w.end,
                    probability=w.probability,
                ))
        results.append(TranscriptSegment(
            text=seg.text.strip(),
            start=seg.start,
            end=seg.end,
            avg_logprob=seg.avg_logprob,
            words=words,
        ))
        pct = (seg.end / duration_s * 100.0) if duration_s > 0 else 0.0
        logger.info(
            "Transcribed segment %d (starts at %.2fs, %.1f%% complete): %r",
            len(results), seg.start, pct, seg.text.strip()
        )

    total_words = sum(len(s.words) for s in results)
    logger.info("Transcription complete: %d segment(s), %d word(s).", len(results), total_words)
    return results
