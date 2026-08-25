"""
Stage 1 — Video ingestion via yt-dlp.
Downloads the best available single-file video format.
Returns the local path to the downloaded file.
Skips re-download if a matching file already exists in output_dir.
"""
from __future__ import annotations

import logging
import subprocess
import sys
import json
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)


def _ytdlp_cmd() -> list[str]:
    """
    Return the command prefix to invoke yt-dlp.
    Prefers the installed Python module (always available in venv)
    over a bare binary on PATH.
    """
    return [sys.executable, "-m", "yt_dlp"]


def download_video(url: str, output_dir: Path) -> Path:
    """
    Download *url* into *output_dir* using yt-dlp.

    Returns
    -------
    Path
        Absolute path to the downloaded video file.

    Raises
    ------
    RuntimeError
        If yt-dlp exits with a non-zero return code.
    """
    # Common flags to avoid bot-detection and TLS issues (needed for ok.ru etc.)
    # --legacy-server-connect: forces TLS 1.2 — Python 3.14 strict SSL rejects
    # ok.ru/vkuser.net servers that don't fully support TLS 1.3.
    _BROWSER_ARGS = [
        "--user-agent",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "--add-header", "Accept-Language:en-US,en;q=0.9",
        "--legacy-server-connect",
        "--no-check-certificate",
        "--sleep-interval", "2",
    ]

    # Pre-network cache check: parse video ID from URL and check if file exists
    video_id = None
    try:
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url)
        if "youtube.com" in parsed.netloc:
            qs = parse_qs(parsed.query)
            if "v" in qs:
                video_id = qs["v"][0]
        elif "youtu.be" in parsed.netloc:
            video_id = parsed.path.strip("/")
        elif "ok.ru" in parsed.netloc:
            parts = parsed.path.strip("/").split("/")
            if parts and parts[-1].isdigit():
                video_id = parts[-1]
    except Exception as parse_err:
        logger.debug("Failed to extract ID from URL: %s", parse_err)

    if video_id:
        # Check if output_dir contains {video_id}.mp4
        local_path = output_dir / f"{video_id}.mp4"
        if local_path.exists():
            logger.info("Video already cached at %s (resolved from URL) — skipping network probe.", local_path)
            return local_path

    template = str(output_dir / "%(id)s.%(ext)s")
    probe_cmd = [
        *_ytdlp_cmd(),
        "--get-filename",
        "-o", template,
        "--no-playlist",
        *_BROWSER_ARGS,
        url,
    ]
    try:
        result = subprocess.run(
            probe_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        predicted_path = Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("Could not probe filename: %s — will attempt download anyway", exc)
        predicted_path = None

    if predicted_path and predicted_path.exists():
        logger.info("Video already cached at %s — skipping download.", predicted_path)
        return predicted_path

    # Perform the actual download
    dl_cmd = [
        *_ytdlp_cmd(),
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", template,
        "--no-playlist",
        "--newline",
        "--socket-timeout", "60",
        "--retries", "10",
        "--fragment-retries", "10",
        *_BROWSER_ARGS,
        url,
    ]
    logger.info("Downloading: %s", url)
    logger.debug("Command: %s", " ".join(dl_cmd))

    proc = subprocess.run(dl_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed (exit {proc.returncode}):\n{proc.stderr}"
        )

    # Re-probe to find the actual output path
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    video_path = Path(result.stdout.strip())

    if not video_path.exists():
        # Fallback: pick the newest mp4 in output_dir
        candidates = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            raise RuntimeError("yt-dlp finished but no video file was found.")
        video_path = candidates[-1]

    logger.info("Downloaded to: %s", video_path)
    return video_path
