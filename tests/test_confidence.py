"""
Unit tests for dialogue_locator/confidence.py

Tests cover:
  - All signals perfect → composite ≈ 1.0, status = "high"
  - VAD disagreement → lower score
  - Poor text match → lower score
  - ASR low quality (bad logprob) → lower score
  - All signals at floor → "best_effort"
  - Status boundary thresholds
  - No matched segments → asr_quality defaults to 0.5 (neutral)
"""
import pytest
from tests.conftest import make_transcript_segment
from dialogue_locator.confidence import compute_confidence, ConfidenceBreakdown, _normalise_logprob
from dialogue_locator import config


class TestNormaliseLogprob:
    def test_best_logprob_maps_to_one(self):
        assert _normalise_logprob(-0.2) == pytest.approx(1.0, abs=0.01)

    def test_worst_logprob_maps_to_zero(self):
        assert _normalise_logprob(-2.0) == pytest.approx(0.0, abs=0.01)

    def test_middle_logprob(self):
        val = _normalise_logprob(-1.1)
        assert 0.0 < val < 1.0

    def test_clipping_above_best(self):
        # logprob above _LOGPROB_BEST should clamp to 1.0
        assert _normalise_logprob(0.0) == pytest.approx(1.0, abs=0.01)

    def test_clipping_below_worst(self):
        assert _normalise_logprob(-99.0) == pytest.approx(0.0, abs=0.01)


class TestComputeConfidence:

    def test_perfect_signals_high_confidence(self):
        seg = make_transcript_segment("text", 0.0, 1.0, avg_logprob=-0.2)
        result = compute_confidence(
            text_match_score=1.0,
            matched_segments=[seg],
            vad_agrees=True,
            vad_transition_found=True,
        )
        assert result.composite >= config.CONF_HIGH
        assert result.status == "high"

    def test_vad_disagrees_lowers_score(self):
        seg = make_transcript_segment("text", 0.0, 1.0, avg_logprob=-0.2)

        agrees = compute_confidence(1.0, [seg], vad_agrees=True, vad_transition_found=True)
        disagrees = compute_confidence(1.0, [seg], vad_agrees=False, vad_transition_found=True)

        assert disagrees.composite < agrees.composite
        assert disagrees.vad_agreement == pytest.approx(0.3)

    def test_no_vad_transition_found_is_neutral(self):
        seg = make_transcript_segment("text", 0.0, 1.0, avg_logprob=-0.2)
        result = compute_confidence(1.0, [seg], vad_agrees=False, vad_transition_found=False)
        assert result.vad_agreement == pytest.approx(0.5)

    def test_poor_text_match_lowers_score(self):
        seg = make_transcript_segment("text", 0.0, 1.0, avg_logprob=-0.2)

        good = compute_confidence(1.0, [seg], vad_agrees=True, vad_transition_found=True)
        poor = compute_confidence(0.4, [seg], vad_agrees=True, vad_transition_found=True)

        assert poor.composite < good.composite

    def test_poor_asr_quality_lowers_score(self):
        good_seg = make_transcript_segment("text", 0.0, 1.0, avg_logprob=-0.2)
        bad_seg  = make_transcript_segment("text", 0.0, 1.0, avg_logprob=-2.0)

        good = compute_confidence(1.0, [good_seg], vad_agrees=True, vad_transition_found=True)
        bad  = compute_confidence(1.0, [bad_seg],  vad_agrees=True, vad_transition_found=True)

        assert bad.composite < good.composite

    def test_all_signals_floor_is_best_effort(self):
        bad_seg = make_transcript_segment("text", 0.0, 1.0, avg_logprob=-2.0)
        result = compute_confidence(
            text_match_score=0.0,
            matched_segments=[bad_seg],
            vad_agrees=False,
            vad_transition_found=False,
        )
        assert result.status == "best_effort"

    def test_no_matched_segments_uses_neutral_asr(self):
        result = compute_confidence(
            text_match_score=1.0,
            matched_segments=[],
            vad_agrees=True,
            vad_transition_found=True,
        )
        assert result.asr_quality == pytest.approx(0.5)
        # Should still be at least medium confidence
        assert result.composite >= config.CONF_MEDIUM

    def test_composite_is_clamped_to_zero_one(self):
        # Even with extreme inputs, should not go outside [0, 1]
        bad_seg = make_transcript_segment("x", 0.0, 1.0, avg_logprob=-99.0)
        result = compute_confidence(0.0, [bad_seg], False, False)
        assert 0.0 <= result.composite <= 1.0

    def test_composite_clamp_ceiling(self):
        good_seg = make_transcript_segment("x", 0.0, 1.0, avg_logprob=99.0)
        result = compute_confidence(1.0, [good_seg], True, True)
        assert result.composite <= 1.0

    def test_weights_sum_to_one(self):
        from dialogue_locator.config import CONF_WEIGHT_TEXT, CONF_WEIGHT_ASR, CONF_WEIGHT_VAD
        total = CONF_WEIGHT_TEXT + CONF_WEIGHT_ASR + CONF_WEIGHT_VAD
        assert total == pytest.approx(1.0, abs=1e-9)


class TestStatusThresholds:
    """Verify status labels align with the configured thresholds."""

    def _result_with_composite(self, composite: float) -> ConfidenceBreakdown:
        """Build a ConfidenceBreakdown whose composite is approximately *composite*."""
        # Tune text_match_score to hit the desired composite:
        # composite ≈ 0.5*text + 0.3*0.5 + 0.2*1.0 = 0.5*text + 0.35
        # → text = (composite - 0.35) / 0.5
        text_score = max(0.0, min(1.0, (composite - 0.35) / 0.5))
        seg = make_transcript_segment("x", 0.0, 1.0, avg_logprob=-1.1)  # neutral asr
        return compute_confidence(text_score, [seg], vad_agrees=True, vad_transition_found=True)

    def test_high_threshold(self):
        result = self._result_with_composite(config.CONF_HIGH + 0.05)
        assert result.status == "high"

    def test_medium_threshold(self):
        result = self._result_with_composite(config.CONF_MEDIUM + 0.02)
        assert result.status in ("medium", "high")

    def test_low_threshold(self):
        r = compute_confidence(0.0, [], False, False)
        assert r.status in ("low", "best_effort")

    def test_best_effort_label(self):
        bad_seg = make_transcript_segment("x", 0.0, 1.0, avg_logprob=-2.0)
        result = compute_confidence(0.0, [bad_seg], False, False)
        assert result.status == "best_effort"
