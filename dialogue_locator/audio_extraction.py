"""
Stage 2 — Audio extraction via ffmpeg.
Converts the video's audio track to a mono 16 kHz WAV file,
which is the format required by both Silero VAD and Whisper/WhisperX.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    """
    Extract audio from *video_path* and write a mono 16 kHz WAV file.

    Parameters
    ----------
    video_path:
        Path to the downloaded video file.
    output_dir:
        Directory to write the WAV file into.

    Returns
    -------
    Path
        Absolute path to the WAV file.

    Raises
    ------
    RuntimeError
        If ffmpeg exits with a non-zero return code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / (video_path.stem + "_audio.wav")

    if wav_path.exists():
        logger.info("Audio already extracted at %s — skipping.", wav_path)
        return wav_path

    cmd = [
        config.FFMPEG_BIN,
        "-y",                      # overwrite without asking
        "-i", str(video_path),
        "-vn",                     # drop video stream
        "-ac", "1",                # mono
        "-ar", "16000",            # 16 kHz sample rate
        "-acodec", "pcm_s16le",    # 16-bit PCM little-endian
        str(wav_path),
    ]
    logger.info("Extracting audio: %s → %s", video_path.name, wav_path.name)
    logger.debug("Command: %s", " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio extraction failed (exit {proc.returncode}):\n{proc.stderr}"
        )

    logger.info("Audio written: %s (%.1f MB)", wav_path, wav_path.stat().st_size / 1e6)
    return wav_path
