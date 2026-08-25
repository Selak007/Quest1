"""
Stage 9 — Frame image extraction via ffmpeg.

Seeks to the exact frame number (using PTS timestamp) and saves it as
a JPEG file. Uses the container-read PTS rather than a computed seek
time to ensure we land on the correct frame even in variable-fps streams.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


def extract_frame(
    video_path: Path,
    pts_s: float,
    frame_number: int,
    output_dir: Path,
) -> Path:
    """
    Extract the video frame at *pts_s* seconds and save as a JPEG.

    Parameters
    ----------
    video_path:
        Path to the video file.
    pts_s:
        The exact presentation timestamp (seconds) of the target frame.
        We seek here — more accurate than specifying a frame number.
    frame_number:
        Used only for the output filename.
    output_dir:
        Directory to write the frame image into.

    Returns
    -------
    Path
        Path to the saved JPEG file.

    Raises
    ------
    RuntimeError
        If ffmpeg fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"frame_{frame_number:07d}.{config.FRAME_OUTPUT_FORMAT}"

    if out_path.exists():
        logger.info("Frame image already exists: %s", out_path)
        return out_path

    cmd = [
        config.FFMPEG_BIN,
        "-y",
        "-ss", f"{pts_s:.6f}",       # seek to PTS — accurate for keyframe-indexed containers
        "-i", str(video_path),
        "-frames:v", "1",             # extract exactly one frame
        "-q:v", str(100 - config.FRAME_OUTPUT_QUALITY),  # ffmpeg quality (2 = near-lossless)
        "-update", "1",               # required for single-frame output
        str(out_path),
    ]
    logger.info("Extracting frame %d at PTS %.3fs …", frame_number, pts_s)
    logger.debug("Command: %s", " ".join(cmd))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg frame extraction failed (exit {proc.returncode}):\n{proc.stderr}"
        )

    if not out_path.exists():
        raise RuntimeError(
            f"ffmpeg reported success but frame file not found: {out_path}"
        )

    size_kb = out_path.stat().st_size / 1024
    logger.info("Frame saved: %s (%.1f KB)", out_path, size_kb)
    return out_path
