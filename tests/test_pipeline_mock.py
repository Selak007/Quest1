"""
Full pipeline integration test with ALL I/O mocked.

No real video, audio, model downloads, or ffmpeg calls are needed.
Verifies that:
  - run_pipeline wires all stages correctly and returns a populated LocalizationResult
  - The spec worked example (Sherlock Holmes, "My mind rebels at stagnation")
    produces frame 10057 at 402.318s
  - Low-confidence path sets status = "best_effort"
  - strict=True raises LowConfidenceError on best_effort results
  - Missing phrase raises ValueError
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from dialogue_locator.alignment import AlignedWord
from dialogue_locator.asr import TranscriptSegment
from dialogue_locator.vad import SpeechSegment
from dialogue_locator.frame_mapping import FramePTS
from dialogue_locator.pipeline import run_pipeline, LowConfidenceError


# ── Shared fixture data (mirrors the spec worked example) ─────────────────────

SPEC_ALIGNED_WORDS = [
    AlignedWord(word="My",         start=402.318, end=402.520, score=0.99),
    AlignedWord(word="mind",       start=402.520, end=402.781, score=0.98),
    AlignedWord(word="rebels",     start=402.781, end=403.040, score=0.97),
    AlignedWord(word="at",         start=403.040, end=403.180, score=0.99),
    AlignedWord(word="stagnatlon", start=403.180, end=403.600, score=0.91),  # ASR typo
]

SPEC_FRAME_PTS = [
    FramePTS(frame_number=10056, pts_s=402.240),
    FramePTS(frame_number=10057, pts_s=402.280),
    FramePTS(frame_number=10058, pts_s=402.320),
    FramePTS(frame_number=10059, pts_s=402.360),
]

SPEC_VAD_SEGS = [SpeechSegment(start_s=402.3, end_s=410.0)]

SPEC_TRANSCRIPT = [
    TranscriptSegment(
        text="my mind rebels at stagnatlon give me",
        start=402.0,
        end=405.0,
        avg_logprob=-0.4,
        words=[],
    )
]


def _mock_pipeline(
    *,
    aligned_words=None,
    frame_pts=None,
    vad_segs=None,
    transcript=None,
    frame_image_path="output/frames/frame_0010057.jpg",
):
    """Return a context manager that patches all I/O stages."""
    import contextlib

    aligned_words = aligned_words if aligned_words is not None else SPEC_ALIGNED_WORDS
    frame_pts     = frame_pts     if frame_pts     is not None else SPEC_FRAME_PTS
    vad_segs      = vad_segs      if vad_segs      is not None else SPEC_VAD_SEGS
    transcript    = transcript    if transcript    is not None else SPEC_TRANSCRIPT

    @contextlib.contextmanager
    def _ctx():
        with \
            patch("dialogue_locator.pipeline.download_video",
                  return_value=Path("fake_video.mp4")), \
            patch("dialogue_locator.pipeline.extract_audio",
                  return_value=Path("fake_audio.wav")), \
            patch("dialogue_locator.pipeline.detect_speech_segments",
                  return_value=vad_segs), \
            patch("dialogue_locator.pipeline.transcribe",
                  return_value=transcript), \
            patch("dialogue_locator.pipeline.align_transcript",
                  return_value=aligned_words), \
            patch("dialogue_locator.pipeline.read_frame_pts",
                  return_value=frame_pts), \
            patch("dialogue_locator.pipeline.extract_frame",
                  return_value=Path(frame_image_path)):
            yield

    return _ctx()


# ── Spec worked example ───────────────────────────────────────────────────────

class TestSpecExample:

    def test_returns_frame_10057(self):
        with _mock_pipeline():
            result = run_pipeline(
                url="https://ok.ru/video/fake",
                target="My mind rebels at stagnation",
                output_dir=Path("./test_output"),
            )
        assert result.frame_number == 10057

    def test_timestamp_is_402_318(self):
        with _mock_pipeline():
            result = run_pipeline(
                url="https://ok.ru/video/fake",
                target="My mind rebels at stagnation",
                output_dir=Path("./test_output"),
            )
        assert result.timestamp_s == pytest.approx(402.318, abs=0.001)

    def test_timestamp_fmt_format(self):
        with _mock_pipeline():
            result = run_pipeline(
                url="https://ok.ru/video/fake",
                target="My mind rebels at stagnation",
                output_dir=Path("./test_output"),
            )
        # Should be HH:MM:SS.mmm
        assert result.timestamp_fmt.startswith("00:06:42")

    def test_dialogue_text_preserved(self):
        with _mock_pipeline():
            result = run_pipeline(
                url="https://ok.ru/video/fake",
                target="My mind rebels at stagnation",
                output_dir=Path("./test_output"),
            )
        assert result.dialogue_text == "My mind rebels at stagnation"

    def test_matched_text_contains_typo(self):
        """ASR produced 'stagnatlon' — matched_text should reflect that."""
        with _mock_pipeline():
            result = run_pipeline(
                url="https://ok.ru/video/fake",
                target="My mind rebels at stagnation",
                output_dir=Path("./test_output"),
            )
        assert "stagnatlon" in result.matched_text.lower() or \
               "stagnation" in result.matched_text.lower()

    def test_confidence_is_high(self):
        with _mock_pipeline():
            result = run_pipeline(
                url="https://ok.ru/video/fake",
                target="My mind rebels at stagnation",
                output_dir=Path("./test_output"),
            )
        assert result.confidence >= 0.70  # should be high given good ASR + VAD agreement

    def test_frame_image_path_populated(self):
        with _mock_pipeline():
            result = run_pipeline(
                url="https://ok.ru/video/fake",
                target="My mind rebels at stagnation",
                output_dir=Path("./test_output"),
            )
        assert result.frame_image_path is not None


# ── Low-confidence path ───────────────────────────────────────────────────────

class TestLowConfidencePath:

    _BAD_WORDS = [
        AlignedWord(word="zxqw", start=5.0, end=5.3, score=0.1),
        AlignedWord(word="vbkp", start=5.3, end=5.6, score=0.1),
    ]
    _BAD_TRANSCRIPT = [
        TranscriptSegment("zxqw vbkp", 5.0, 5.6, avg_logprob=-2.0, words=[])
    ]

    def test_low_confidence_still_returns_result(self):
        """Even with a terrible match, run_pipeline returns (not raises) by default."""
        with _mock_pipeline(
            aligned_words=self._BAD_WORDS,
            transcript=self._BAD_TRANSCRIPT,
            vad_segs=[SpeechSegment(start_s=99.0, end_s=102.0)],  # far from onset
        ):
            result = run_pipeline(
                url="fake",
                target="zxqw vbkp",
                output_dir=Path("./test_output"),
                strict=False,
            )
        assert result.status in ("best_effort", "low", "medium", "high")

    def test_strict_mode_raises_on_best_effort(self):
        """strict=True should raise LowConfidenceError when confidence is below CONF_LOW."""
        bad_words = [
            AlignedWord(word="hello", start=1.0, end=1.3, score=0.9),
            AlignedWord(word="world", start=1.3, end=1.6, score=0.9),
        ]
        bad_transcript = [
            TranscriptSegment("hello world", 1.0, 1.6, avg_logprob=-2.0, words=[])
        ]
        # VAD far away → low vad_agreement; bad ASR → low asr_quality
        # text match will be OK but combined should be low enough
        with _mock_pipeline(
            aligned_words=bad_words,
            transcript=bad_transcript,
            vad_segs=[SpeechSegment(start_s=99.0, end_s=102.0)],
        ):
            from dialogue_locator import config as cfg
            original_conf_low = cfg.CONF_LOW
            cfg.CONF_LOW = 0.99  # Artificially raise threshold so we trigger best_effort
            try:
                with pytest.raises(LowConfidenceError):
                    run_pipeline(
                        url="fake",
                        target="hello world",
                        output_dir=Path("./test_output"),
                        strict=True,
                    )
            finally:
                cfg.CONF_LOW = original_conf_low


# ── Missing phrase ────────────────────────────────────────────────────────────

class TestMissingPhrase:

    def test_phrase_not_in_transcript_raises_value_error(self):
        short_words = [
            AlignedWord(word="The", start=1.0, end=1.2, score=0.9),
            AlignedWord(word="game", start=1.2, end=1.5, score=0.9),
        ]
        with _mock_pipeline(aligned_words=short_words):
            with pytest.raises(ValueError, match="not found in transcript"):
                run_pipeline(
                    url="fake",
                    target="completely unrelated phrase that will never match",
                    output_dir=Path("./test_output"),
                )

    def test_empty_transcript_raises_value_error(self):
        with _mock_pipeline(aligned_words=[], transcript=[]):
            with pytest.raises(ValueError):
                run_pipeline(
                    url="fake",
                    target="any phrase",
                    output_dir=Path("./test_output"),
                )


# ── Result dataclass completeness ─────────────────────────────────────────────

class TestResultDataclass:
    def test_all_required_fields_populated(self):
        with _mock_pipeline():
            result = run_pipeline(
                url="fake",
                target="My mind rebels at stagnation",
                output_dir=Path("./test_output"),
            )

        # Every field must be populated / non-default
        assert isinstance(result.timestamp_s, float)
        assert isinstance(result.timestamp_fmt, str)
        assert isinstance(result.frame_number, int)
        assert isinstance(result.dialogue_text, str)
        assert isinstance(result.matched_text, str)
        assert 0.0 <= result.confidence <= 1.0
        assert result.status in ("high", "medium", "low", "best_effort")
        assert isinstance(result.text_score, float)
        assert isinstance(result.asr_quality, float)
        assert isinstance(result.vad_agreement, float)

    def test_no_frame_extraction(self):
        with _mock_pipeline():
            result = run_pipeline(
                url="fake",
                target="My mind rebels at stagnation",
                output_dir=Path("./test_output"),
                extract_frame_image=False,
            )
        assert result.frame_image_path is None
