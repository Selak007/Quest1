"""
Stage 7 — Frame mapping via real container PTS values.

Architecture v2 mandates reading ACTUAL per-frame presentation timestamps
from the container (via ffprobe JSON output) rather than computing
frame numbers from an assumed constant frame rate.

This is critical because:
- Dropped/duplicated frames break t * fps arithmetic.
- PAL/NTSC streams use fractional frame rates (25/1, 30000/1001, etc.).
- Container-read PTS is authoritative.

The frame boundary test is interval containment (§4 of the spec):
    Frame N is correct iff PTS(N) <= onset_time < PTS(N+1)
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List

from . import config

logger = logging.getLogger(__name__)


@dataclass
class FramePTS:
    """A single video frame's presentation timestamp."""
    frame_number: int    # 0-indexed frame counter in the container
    pts_s: float         # presentation timestamp in seconds


@dataclass
class FrameLocalization:
    """Result of interval-containment frame lookup."""
    frame_number: int
    pts_s: float         # PTS of the located frame
    next_pts_s: float    # PTS of the next frame (defines the window upper bound)


def _ffprobe_frames(video_path: Path, extra_args: list) -> List[FramePTS]:
    """
    Run ffprobe and return a list of FramePTS.

    extra_args is inserted between the flags and the file path — use it to
    pass -read_intervals for windowed reads.

    NOTE: frame_number is computed from pts_s × fps so it is always an
    absolute index from the start of the video, regardless of where ffprobe
    started reading.
    """
    cmd = [
        config.FFPROBE_BIN,
        "-v", "error",
        "-print_format", "json",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_pts_time",
    ] + extra_args + [str(video_path)]

    logger.debug("ffprobe command: %s", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffprobe failed (exit {proc.returncode}):\n{proc.stderr}"
        )

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ffprobe output is not valid JSON: {exc}") from exc

    result: List[FramePTS] = []
    for f in data.get("frames", []):
        ts_str = f.get("best_effort_timestamp_time") or f.get("pkt_pts_time")
        if ts_str is None or ts_str == "N/A":
            continue
        # frame_number is filled in by the caller once fps is known
        result.append(FramePTS(frame_number=-1, pts_s=float(ts_str)))

    result.sort(key=lambda f: f.pts_s)
    return result


def get_video_fps(video_path: Path) -> float:
    """
    Return the average frame rate of the first video stream as a float.
    Falls back to 24.0 if it cannot be determined.
    """
    cmd = [
        config.FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-print_format", "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.warning("Could not read FPS from %s — defaulting to 24.0", video_path.name)
        return 24.0
    try:
        data = json.loads(proc.stdout)
        rate_str = data["streams"][0]["avg_frame_rate"]  # e.g. "24000/1001" or "24/1"
        num, den = rate_str.split("/")
        fps = float(num) / float(den)
        logger.debug("Video FPS: %.4f (%s)", fps, rate_str)
        return fps
    except Exception as exc:
        logger.warning("Failed to parse FPS (%s) — defaulting to 24.0", exc)
        return 24.0


def read_frame_pts(
    video_path: Path,
    onset_s: float,
    window_s: float = None,
) -> List[FramePTS]:
    """
    Use ffprobe to read real per-frame PTS values in a window around *onset_s*.

    Parameters
    ----------
    video_path:
        Path to the video file.
    onset_s:
        The candidate onset time in seconds.
    window_s:
        How many seconds either side of onset_s to read.
        Defaults to config.PTS_READ_WINDOW_S.

    Returns
    -------
    List[FramePTS]
        Sorted list of frames in the read window.

    Raises
    ------
    RuntimeError
        If ffprobe fails or returns no frames.
    """
    if window_s is None:
        window_s = config.PTS_READ_WINDOW_S

    start_t = max(0.0, onset_s - window_s)
    duration = 2 * window_s

    # Get real FPS to compute absolute frame numbers from PTS
    fps = get_video_fps(video_path)

    # -read_intervals: "START%+DURATION" (absolute seconds, no leading %)
    interval = f"{start_t:.3f}%+{duration:.3f}"
    frame_pts_list = _ffprobe_frames(video_path, ["-read_intervals", interval])

    if not frame_pts_list:
        logger.warning(
            "Windowed ffprobe returned no frames — falling back to full-file scan."
        )
        frame_pts_list = _ffprobe_frames(video_path, [])

    if not frame_pts_list:
        raise RuntimeError(
            f"ffprobe returned no frames for {video_path.name}"
        )

    # Assign absolute frame numbers: abs_frame = round(pts_s * fps)
    for f in frame_pts_list:
        f.frame_number = round(f.pts_s * fps)

    logger.info(
        "Read %d frame PTS values around onset %.3fs (fps=%.4f).",
        len(frame_pts_list),
        onset_s,
        fps,
    )
    return frame_pts_list


def locate_frame(onset_s: float, frame_pts_list: List[FramePTS]) -> FrameLocalization:
    """
    Perform interval-containment search to find the frame N such that:
        PTS(N) <= onset_s < PTS(N+1)

    Parameters
    ----------
    onset_s:
        The forced-alignment word onset in seconds.
    frame_pts_list:
        Sorted list of FramePTS from read_frame_pts().

    Returns
    -------
    FrameLocalization
        The matching frame.

    Notes
    -----
    Edge cases:
    - If onset_s < first frame's PTS → returns frame 0.
    - If onset_s >= last frame's PTS → returns last frame.
    """
    if not frame_pts_list:
        raise ValueError("frame_pts_list is empty — cannot locate frame.")

    if onset_s < frame_pts_list[0].pts_s:
        logger.warning(
            "Onset %.3fs is before first frame PTS %.3fs — returning frame 0.",
            onset_s,
            frame_pts_list[0].pts_s,
        )
        f = frame_pts_list[0]
        next_pts = frame_pts_list[1].pts_s if len(frame_pts_list) > 1 else f.pts_s + (1.0 / 25)
        return FrameLocalization(f.frame_number, f.pts_s, next_pts)

    # Scan through windows
    for i in range(len(frame_pts_list) - 1):
        f_n = frame_pts_list[i]
        f_n1 = frame_pts_list[i + 1]
        if f_n.pts_s <= onset_s < f_n1.pts_s:
            logger.info(
                "Interval containment: onset %.3fs in [%.3f, %.3f) -> frame %d",
                onset_s,
                f_n.pts_s,
                f_n1.pts_s,
                f_n.frame_number,
            )
            return FrameLocalization(f_n.frame_number, f_n.pts_s, f_n1.pts_s)

    # onset_s >= last frame's PTS → return last frame
    f = frame_pts_list[-1]
    prev_gap = (
        (f.pts_s - frame_pts_list[-2].pts_s) if len(frame_pts_list) > 1 else (1.0 / 25)
    )
    logger.warning(
        "Onset %.3fs is at/after last frame PTS %.3fs — returning last frame.",
        onset_s,
        f.pts_s,
    )
    return FrameLocalization(f.frame_number, f.pts_s, f.pts_s + prev_gap)
