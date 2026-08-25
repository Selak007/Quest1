"""
Unit tests for dialogue_locator/vad.py

Tests cover the VAD corroboration logic (find_nearest_vad_transition),
which is pure Python and testable without loading the actual model.

The detect_speech_segments function requires torch + model download,
so it is tested with a mock to keep unit tests fast and offline.
"""
import pytest
from unittest.mock import patch, MagicMock
from tests.conftest import make_speech_segment
from dialogue_locator.vad import find_nearest_vad_transition, SpeechSegment


class TestFindNearestVadTransition:

    def test_transition_within_window_agrees(self):
        segs = [make_speech_segment(100.0, 105.0)]
        agrees, ts = find_nearest_vad_transition(segs, onset_s=100.5, window_s=1.5)
        assert agrees is True
        assert ts == pytest.approx(100.0, abs=1e-6)

    def test_transition_outside_window_disagrees(self):
        segs = [make_speech_segment(50.0, 55.0)]
        agrees, ts = find_nearest_vad_transition(segs, onset_s=100.0, window_s=1.5)
        assert agrees is False
        assert ts == pytest.approx(50.0, abs=1e-6)

    def test_empty_segments_no_transition(self):
        agrees, ts = find_nearest_vad_transition([], onset_s=10.0, window_s=1.5)
        assert agrees is False
        assert ts is None

    def test_exact_boundary_of_window_agrees(self):
        """Onset exactly at window boundary → agrees (<=, not <)."""
        segs = [make_speech_segment(10.0, 15.0)]
        agrees, ts = find_nearest_vad_transition(segs, onset_s=11.5, window_s=1.5)
        # distance = abs(10.0 - 11.5) = 1.5 → exactly at boundary → agrees
        assert agrees is True

    def test_picks_closest_of_multiple_segments(self):
        segs = [
            make_speech_segment(1.0, 3.0),    # transition at 1.0 — far from onset
            make_speech_segment(10.0, 12.0),  # transition at 10.0 — close
            make_speech_segment(20.0, 22.0),  # transition at 20.0 — far
        ]
        agrees, ts = find_nearest_vad_transition(segs, onset_s=10.2, window_s=1.5)
        assert agrees is True
        assert ts == pytest.approx(10.0, abs=1e-6)

    def test_default_window_used_when_none(self):
        """When window_s=None, config.VAD_CORROBORATION_WINDOW_S is used."""
        from dialogue_locator import config
        segs = [make_speech_segment(5.0, 8.0)]
        # onset 4.0, config window is 1.5 → distance 1.0 → should agree
        agrees, _ = find_nearest_vad_transition(segs, onset_s=4.0, window_s=None)
        assert agrees is True


class TestDetectSpeechSegmentsMocked:
    """
    Smoke tests for detect_speech_segments using mocked Silero model.
    These tests verify the function's output shape/type without any
    actual torch/model loading.
    """

    def test_returns_list_of_speech_segments(self):
        mock_model = MagicMock()
        mock_timestamps = [
            {"start": 1.0, "end": 5.0},
            {"start": 10.0, "end": 15.0},
        ]

        with patch("dialogue_locator.vad._load_vad_model"), \
             patch("dialogue_locator.vad._vad_model", mock_model), \
             patch("silero_vad.get_speech_timestamps",
                   return_value=mock_timestamps), \
             patch("dialogue_locator.vad._load_wav_as_tensor",
                   return_value=MagicMock()):

            # Re-import inside the patch context so module-level references are patched
            from dialogue_locator.vad import detect_speech_segments
            from pathlib import Path
            result = detect_speech_segments(Path("fake.wav"))

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].start_s == pytest.approx(1.0)
        assert result[1].end_s == pytest.approx(15.0)

    def test_empty_audio_returns_empty_list(self):
        with patch("dialogue_locator.vad._load_vad_model"), \
             patch("dialogue_locator.vad._vad_model", MagicMock()), \
             patch("silero_vad.get_speech_timestamps", return_value=[]), \
             patch("dialogue_locator.vad._load_wav_as_tensor",
                   return_value=MagicMock()):

            from dialogue_locator.vad import detect_speech_segments
            from pathlib import Path
            result = detect_speech_segments(Path("silent.wav"))

        assert result == []


class TestSpeechSegmentDataclass:
    def test_duration(self):
        seg = SpeechSegment(start_s=2.0, end_s=5.0)
        assert seg.duration_s == pytest.approx(3.0)

    def test_repr(self):
        seg = SpeechSegment(start_s=1.5, end_s=3.0)
        assert "1.500" in repr(seg)
        assert "3.000" in repr(seg)
