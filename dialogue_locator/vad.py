"""
Stage 3 — Voice Activity Detection via Silero VAD.

Produces a list of (start_s, end_s) speech segments from the WAV file.
These are used for two purposes:
  1. Narrow the ASR search space (skip pure silence / music intros).
  2. Later, as an independent corroboration signal for the confidence score.

Uses silero-vad 5/6 modern API (load_silero_vad / get_speech_timestamps).
Audio is loaded via soundfile + torch (bypasses the broken torchaudio 2.9+
torchcodec path that doesn't support Python 3.14).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import soundfile as sf
import torch

from . import config

logger = logging.getLogger(__name__)

# Module-level cache so the model is only loaded once per process
_vad_model = None


@dataclass
class SpeechSegment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def __repr__(self) -> str:
        return f"SpeechSegment({self.start_s:.3f}s - {self.end_s:.3f}s)"


def _load_wav_as_tensor(wav_path: Path, sampling_rate: int = 16000) -> torch.Tensor:
    """
    Load a mono WAV file into a 1-D float32 torch tensor.
    Uses soundfile (backed by libsndfile) — no torchaudio/torchcodec needed.
    """
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    # Handle stereo: average channels
    if data.ndim > 1:
        data = data.mean(axis=1)
    wav = torch.from_numpy(data)
    # Resample if necessary (our extracted WAVs are already 16kHz, but be safe)
    if sr != sampling_rate:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
        wav = resampler(wav)
    return wav


def _load_vad_model():
    """Lazy-load Silero VAD (only runs once)."""
    global _vad_model
    if _vad_model is not None:
        return
    logger.info("Loading Silero VAD model ...")
    from silero_vad import load_silero_vad
    _vad_model = load_silero_vad(onnx=False)
    logger.info("Silero VAD loaded.")


def detect_speech_segments(wav_path: Path) -> List[SpeechSegment]:
    """
    Run Silero VAD on *wav_path* and return a list of SpeechSegment objects.

    Parameters
    ----------
    wav_path:
        Path to a mono 16 kHz WAV file.

    Returns
    -------
    List[SpeechSegment]
        Sorted list of detected speech intervals (may be empty for silent audio).
    """
    _load_vad_model()

    logger.info("Running VAD on %s ...", wav_path.name)
    wav = _load_wav_as_tensor(wav_path, sampling_rate=16000)

    from silero_vad import get_speech_timestamps
    speech_timestamps = get_speech_timestamps(
        wav,
        _vad_model,
        threshold=config.VAD_THRESHOLD,
        sampling_rate=16000,
        min_speech_duration_ms=config.VAD_MIN_SPEECH_DURATION_MS,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_DURATION_MS,
        return_seconds=True,
    )

    segments = [
        SpeechSegment(start_s=ts["start"], end_s=ts["end"])
        for ts in speech_timestamps
    ]
    logger.info("VAD found %d raw speech segment(s).", len(segments))

    segments = merge_speech_segments(segments)
    logger.info("VAD merged to %d segment(s) (gap_s≤%.1fs).", len(segments), config.VAD_MERGE_GAP_S)
    return segments


def merge_speech_segments(
    segments: List[SpeechSegment],
    gap_s: float = None,
) -> List[SpeechSegment]:
    """
    Merge consecutive speech segments whose inter-segment gap is ≤ *gap_s*.

    Silero VAD returns many short fragments (274 for a 21-min video).
    Passing all of them as clip_timestamps makes faster-whisper process each
    individually, multiplying CPU overhead by ~10x.

    Merging adjacent segments with small gaps gives fewer, longer clips that
    Whisper handles efficiently without losing any speech content.

    Parameters
    ----------
    segments:
        Sorted list of SpeechSegment from Silero VAD.
    gap_s:
        Maximum silence gap to bridge (defaults to config.VAD_MERGE_GAP_S).

    Returns
    -------
    List[SpeechSegment]
        Merged segments — typically 20-60 for a full episode vs 200-300 raw.
    """
    if gap_s is None:
        gap_s = config.VAD_MERGE_GAP_S
    if not segments:
        return segments

    merged: List[SpeechSegment] = [SpeechSegment(segments[0].start_s, segments[0].end_s)]
    for seg in segments[1:]:
        if seg.start_s - merged[-1].end_s <= gap_s:
            merged[-1] = SpeechSegment(merged[-1].start_s, seg.end_s)
        else:
            merged.append(SpeechSegment(seg.start_s, seg.end_s))
    return merged



def find_nearest_vad_transition(
    segments: List[SpeechSegment],
    onset_s: float,
    window_s: float = None,
) -> tuple[bool, float | None]:
    """
    Check whether a silence->speech transition falls near *onset_s*.

    Returns
    -------
    (agrees: bool, nearest_transition_s: float | None)
        agrees = True  -> a transition is within *window_s* of onset_s
        nearest_transition_s -> the exact transition time (or None if none found)
    """
    if window_s is None:
        window_s = config.VAD_CORROBORATION_WINDOW_S

    nearest: float | None = None
    min_dist = float("inf")

    for seg in segments:
        dist = abs(seg.start_s - onset_s)
        if dist < min_dist:
            min_dist = dist
            nearest = seg.start_s

    agrees = min_dist <= window_s
    return agrees, nearest
