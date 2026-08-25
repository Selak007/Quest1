"""
Central configuration: model names, thresholds, paths.
Override any value by setting the matching environment variable.
"""
import os
from pathlib import Path

# ── Whisper / WhisperX ────────────────────────────────────────────────────────
# Two-phase model config:
#   SCAN  model  → fast full-file pass  (tiny, greedy, no word-timestamps)
#   REFINE model → precise window pass  (small, beam=5, word-timestamps)
WHISPER_SCAN_MODEL: str   = os.getenv("WHISPER_SCAN_MODEL",   "tiny")
WHISPER_REFINE_MODEL: str = os.getenv("WHISPER_REFINE_MODEL", "small")
# Legacy single-pass model (used by unit tests and the old --model CLI flag)
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "medium")
WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "en")
WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # cpu-friendly

# ── Two-phase ASR knobs ───────────────────────────────────────────────────────
# Scan: minimum VAD segment duration to run ASR on.
# A 7-word phrase at normal speech (~3 wps) takes ≥ 2.0s.
# Segments shorter than this cannot contain the full phrase → skip.
SCAN_MIN_SEG_DURATION_S: float = float(os.getenv("SCAN_MIN_SEG_DURATION_S", "2.0"))

# Refine: half-width of the window around the rough scan match (seconds).
# Total refine clip = 2 × REFINE_WINDOW_S (default: ±30s = 60s window).
REFINE_WINDOW_S: float = float(os.getenv("REFINE_WINDOW_S", "30.0"))

# ── Silero VAD ────────────────────────────────────────────────────────────────
VAD_THRESHOLD: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SPEECH_DURATION_MS: int = int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "250"))
VAD_MIN_SILENCE_DURATION_MS: int = int(os.getenv("VAD_MIN_SILENCE_DURATION_MS", "100"))
# Merge nearby VAD segments: gaps ≤ this value (seconds) are bridged.
# Reduces 200-300 raw fragments to 20-60 longer clips → 4-10x faster ASR.
VAD_MERGE_GAP_S: float = float(os.getenv("VAD_MERGE_GAP_S", "1.5"))


# ── Fuzzy matching ────────────────────────────────────────────────────────────
FUZZY_SCORE_CUTOFF: float = float(os.getenv("FUZZY_SCORE_CUTOFF", "60.0"))

# ── VAD corroboration window (seconds around onset) ───────────────────────────
VAD_CORROBORATION_WINDOW_S: float = float(os.getenv("VAD_CORROBORATION_WINDOW_S", "1.5"))

# ── Confidence weights (must sum to 1.0) ─────────────────────────────────────
CONF_WEIGHT_TEXT: float = 0.50
CONF_WEIGHT_ASR: float = 0.30
CONF_WEIGHT_VAD: float = 0.20

# ── Confidence status thresholds ─────────────────────────────────────────────
CONF_HIGH: float = 0.80
CONF_MEDIUM: float = 0.60
CONF_LOW: float = 0.40
# below CONF_LOW → "best_effort"

# ── Frame extraction ──────────────────────────────────────────────────────────
FRAME_OUTPUT_FORMAT: str = "jpg"
FRAME_OUTPUT_QUALITY: int = 95  # JPEG quality (1-100)

# ── PTS read window: how many seconds around the onset to read PTS for ────────
PTS_READ_WINDOW_S: float = float(os.getenv("PTS_READ_WINDOW_S", "2.0"))

# ── ffprobe / ffmpeg binary names (override if not on PATH) ──────────────────
FFPROBE_BIN: str = os.getenv("FFPROBE_BIN", "ffprobe")
FFMPEG_BIN: str = os.getenv("FFMPEG_BIN", "ffmpeg")
YTDLP_BIN: str = os.getenv("YTDLP_BIN", "yt-dlp")

# ── Default output directory ──────────────────────────────────────────────────
DEFAULT_OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", "./output"))
