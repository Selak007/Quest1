"""
Top-level pipeline orchestrator.

Runs all stages in order and returns a LocalizationResult dataclass.
Each stage is logged with timing information so you can see which
step is the bottleneck.

Low-confidence behaviour (§7 of the spec):
  - Always returns a best-effort result.
  - status field reflects confidence level.
  - If strict=True, raises LowConfidenceError when status == "best_effort".
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config
from .ingestion import download_video
from .audio_extraction import extract_audio
from .vad import detect_speech_segments, find_nearest_vad_transition
from .asr import transcribe_scan, transcribe_refine, unload_scan_model
from .alignment import align_transcript
from .matcher import find_phrase
from .frame_mapping import read_frame_pts, locate_frame
from .confidence import compute_confidence
from .frame_extraction import extract_frame

logger = logging.getLogger(__name__)


class LowConfidenceError(RuntimeError):
    """Raised in strict mode when the result confidence is below the low threshold."""


@dataclass
class LocalizationResult:
    """Complete output from the dialogue localization pipeline."""
    # ── Core result ───────────────────────────────────────────────────────────
    timestamp_s: float
    timestamp_fmt: str          # HH:MM:SS.mmm
    frame_number: int
    dialogue_text: str          # user-supplied target phrase
    matched_text: str           # what ASR actually said (may have minor typos)
    confidence: float           # composite score 0–1
    status: str                 # "high" | "medium" | "low" | "best_effort"

    # ── Detailed breakdown ────────────────────────────────────────────────────
    text_score: float = 0.0
    asr_quality: float = 0.0
    vad_agreement: float = 0.0
    vad_transition_s: Optional[float] = None
    pts_window_start: float = 0.0
    pts_window_end: float = 0.0

    # ── Paths ─────────────────────────────────────────────────────────────────
    frame_image_path: Optional[str] = None
    video_path: Optional[str] = None
    audio_path: Optional[str] = None


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _stage(name: str):
    """Simple context manager that logs elapsed time for a stage."""
    class _Stage:
        def __enter__(self):
            self.t0 = time.perf_counter()
            logger.info("━━ Stage: %s", name)
            return self
        def __exit__(self, *_):
            elapsed = time.perf_counter() - self.t0
            logger.info("   └─ done in %.2fs", elapsed)
    return _Stage()


def run_pipeline(
    url: str,
    target: str,
    output_dir: Path = None,
    extract_frame_image: bool = True,
    strict: bool = False,
    whisper_model: str = None,
) -> LocalizationResult:
    """
    Run the full spoken-dialogue → exact-frame pipeline.

    Parameters
    ----------
    url:
        Video URL (any format supported by yt-dlp: YouTube, ok.ru, etc.)
    target:
        The spoken line to locate, e.g. "My mind rebels at stagnation".
    output_dir:
        Where to write video, audio, and frame files.
        Defaults to config.DEFAULT_OUTPUT_DIR.
    extract_frame_image:
        If True (default), save the located frame as a JPEG.
    strict:
        If True, raises LowConfidenceError when confidence < CONF_LOW.
    whisper_model:
        Override config.WHISPER_MODEL for this call.

    Returns
    -------
    LocalizationResult

    Raises
    ------
    LowConfidenceError
        Only in strict=True mode when confidence is below threshold.
    ValueError
        If no phrase match is found in the transcript.
    """
    if output_dir is None:
        output_dir = config.DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)

    if whisper_model:
        config.WHISPER_MODEL = whisper_model

    t_total = time.perf_counter()
    logger.info("═══ Dialogue Localization Pipeline ═══")
    logger.info("  URL   : %s", url)
    logger.info("  Target: %r", target)

    # ── Stage 1: Download video ───────────────────────────────────────────────
    with _stage("Video Download"):
        video_path = download_video(url, output_dir / "video")

    # ── Stage 2: Extract audio ────────────────────────────────────────────────
    with _stage("Audio Extraction"):
        wav_path = extract_audio(video_path, output_dir / "audio")

    # ── Stage 3: Silero VAD ───────────────────────────────────────────────────
    with _stage("VAD"):
        speech_segs = detect_speech_segments(wav_path)

    # ── Stage 4a: SCAN — tiny model, greedy, no word-timestamps ─────────────
    # Only runs on VAD segments ≥ SCAN_MIN_SEG_DURATION_S; skips the rest.
    with _stage("ASR Phase 1: SCAN (tiny, greedy)"):
        scan_segs = transcribe_scan(wav_path, speech_segs)

    # ── Stage 4b: Fuzzy match on scan output (rough onset) ───────────────────
    with _stage("Fuzzy Phrase Match (scan)"):
        # align_transcript not needed for scan — use faster-whisper segment times
        from .alignment import AlignedWord
        # Build pseudo-AlignedWord list from scan segment timestamps
        scan_words: list = []
        for seg in scan_segs:
            # Split segment text into rough per-word tokens
            tokens = seg.text.split()
            if not tokens:
                continue
            seg_dur = seg.end - seg.start
            tok_dur = seg_dur / len(tokens)
            for i, tok in enumerate(tokens):
                scan_words.append(AlignedWord(
                    word=tok,
                    start=seg.start + i * tok_dur,
                    end=seg.start + (i + 1) * tok_dur,
                    score=max(0.0, seg.avg_logprob + 1.0),  # rough proxy
                    source="scan",
                ))
        match_scan = find_phrase(target, scan_words)

    if match_scan is None:
        raise ValueError(
            f"Target phrase not found in transcript (scan pass): {target!r}\n"
            f"Try lowering FUZZY_SCORE_CUTOFF (currently {config.FUZZY_SCORE_CUTOFF})."
        )

    rough_onset_s = match_scan.onset_s
    logger.info(
        "[SCAN] Rough onset: %.3fs  matched: %r  score: %.2f",
        rough_onset_s, match_scan.matched_text, match_scan.score,
    )

    # ── Stage 4c: Free scan model RAM before loading refine model ────────────
    unload_scan_model()

    # ── Stage 4d: REFINE — small model, beam, word-timestamps on ±window ──────
    with _stage("ASR Phase 2: REFINE (small, beam, word-timestamps)"):
        refine_segs = transcribe_refine(wav_path, onset_s=rough_onset_s)

    # ── Stage 5: Forced alignment (WhisperX) on refine window only ───────────
    with _stage("Forced Alignment (WhisperX)"):
        aligned_words = align_transcript(wav_path, refine_segs)

    # ── Stage 6: Fuzzy match on precisely aligned words ───────────────────────
    with _stage("Fuzzy Phrase Match (refine)"):
        match = find_phrase(target, aligned_words)

    if match is None:
        # Fall back to scan match if refine window missed the phrase
        logger.warning(
            "Refine pass did not find phrase — falling back to scan-pass onset."
        )
        from .alignment import AlignedWord as _AW
        match = match_scan
        # Rebuild aligned_words from refine_segs so confidence calc works
        aligned_words = scan_words

    onset_s = match.onset_s
    logger.info("Onset: %.3f s  matched: %r", onset_s, match.matched_text)

    # ── Stage 7: VAD corroboration ────────────────────────────────────────────
    with _stage("VAD Corroboration"):
        vad_agrees, vad_transition_s = find_nearest_vad_transition(speech_segs, onset_s)
        vad_transition_found = vad_transition_s is not None
        logger.info(
            "VAD corroboration: agrees=%s transition_at=%s",
            vad_agrees,
            f"{vad_transition_s:.3f}s" if vad_transition_s else "none",
        )

    # ── Stage 8: Read real frame PTS ─────────────────────────────────────────
    with _stage("Frame PTS Read (ffprobe)"):
        frame_pts_list = read_frame_pts(video_path, onset_s)

    # ── Stage 9: Interval containment → locate frame ─────────────────────────
    with _stage("Frame Interval Containment"):
        frame_loc = locate_frame(onset_s, frame_pts_list)

    # ── Stage 10: Confidence scoring ─────────────────────────────────────────
    with _stage("Confidence Scoring"):
        # Find refine-pass ASR segments that overlap the match window
        matched_asr_segs = [
            s for s in refine_segs
            if s.start <= match.offset_s and s.end >= onset_s
        ]
        conf = compute_confidence(
            text_match_score=match.score,
            matched_segments=matched_asr_segs,
            vad_agrees=vad_agrees,
            vad_transition_found=vad_transition_found,
        )


    # ── Stage 11: Extract frame image ────────────────────────────────────────
    frame_image_path: Optional[str] = None
    if extract_frame_image:
        with _stage("Frame Extraction"):
            try:
                img_path = extract_frame(
                    video_path,
                    pts_s=frame_loc.pts_s,
                    frame_number=frame_loc.frame_number,
                    output_dir=output_dir / "frames",
                )
                frame_image_path = str(img_path)
            except RuntimeError as exc:
                logger.error("Frame extraction failed: %s", exc)

    elapsed = time.perf_counter() - t_total
    logger.info("═══ Pipeline complete in %.1fs ═══", elapsed)

    result = LocalizationResult(
        timestamp_s=onset_s,
        timestamp_fmt=_fmt_time(onset_s),
        frame_number=frame_loc.frame_number,
        dialogue_text=target,
        matched_text=match.matched_text,
        confidence=conf.composite,
        status=conf.status,
        text_score=conf.text_score,
        asr_quality=conf.asr_quality,
        vad_agreement=conf.vad_agreement,
        vad_transition_s=vad_transition_s,
        pts_window_start=frame_loc.pts_s,
        pts_window_end=frame_loc.next_pts_s,
        frame_image_path=frame_image_path,
        video_path=str(video_path),
        audio_path=str(wav_path),
    )

    if strict and conf.status == "best_effort":
        raise LowConfidenceError(
            f"Confidence {conf.composite:.3f} is below minimum threshold "
            f"({config.CONF_LOW}). Result may be unreliable.\n"
            f"  Matched text: {match.matched_text!r}\n"
            f"  Text score  : {conf.text_score:.2f}\n"
            f"  ASR quality : {conf.asr_quality:.2f}\n"
            f"  VAD agrees  : {vad_agrees}"
        )

    return result
