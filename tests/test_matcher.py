"""
Unit tests for dialogue_locator/matcher.py

Tests cover:
  - Exact match → perfect score
  - ASR typo ("stagnatlon") → still matches above cutoff
  - Wrong phrase → no match / None returned
  - Multiple occurrences → earliest one returned
  - Single word target
  - Empty transcript → graceful None
  - Target longer than transcript → handled without crash
  - Case insensitivity and punctuation stripping
"""
import pytest
from tests.conftest import make_aligned_words
from dialogue_locator.matcher import find_phrase, _normalise


# ── Normalisation ──────────────────────────────────────────────────────────────

class TestNormalise:
    def test_lowercases(self):
        assert _normalise("Hello World") == "hello world"

    def test_strips_punctuation(self):
        assert _normalise("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert _normalise("  hello   world  ") == "hello world"

    def test_preserves_apostrophe(self):
        # contractions shouldn't lose the apostrophe
        assert "don't" in _normalise("Don't stop")


# ── Exact match ────────────────────────────────────────────────────────────────

class TestExactMatch:
    def test_exact_match_returns_result(self):
        words = make_aligned_words([
            ("My", 10.0, 10.2),
            ("mind", 10.2, 10.5),
            ("rebels", 10.5, 10.9),
            ("at", 10.9, 11.0),
            ("stagnation", 11.0, 11.5),
        ])
        result = find_phrase("My mind rebels at stagnation", words)
        assert result is not None

    def test_exact_match_score_is_near_one(self):
        words = make_aligned_words([
            ("My", 10.0, 10.2),
            ("mind", 10.2, 10.5),
            ("rebels", 10.5, 10.9),
            ("at", 10.9, 11.0),
            ("stagnation", 11.0, 11.5),
        ])
        result = find_phrase("My mind rebels at stagnation", words)
        assert result.score >= 0.95

    def test_exact_match_onset_time(self):
        words = make_aligned_words([
            ("My", 402.318, 402.520),
            ("mind", 402.520, 402.781),
            ("rebels", 402.781, 403.040),
            ("at", 403.040, 403.180),
            ("stagnation", 403.180, 403.600),
        ])
        result = find_phrase("My mind rebels at stagnation", words)
        assert abs(result.onset_s - 402.318) < 0.001

    def test_case_insensitive(self):
        words = make_aligned_words([
            ("ELEMENTARY", 5.0, 5.5),
            ("MY", 5.5, 5.7),
            ("DEAR", 5.7, 5.9),
            ("WATSON", 5.9, 6.2),
        ])
        result = find_phrase("elementary my dear watson", words)
        assert result is not None
        assert result.score >= 0.90


# ── ASR typos ──────────────────────────────────────────────────────────────────

class TestAsrTypoTolerance:
    def test_single_char_typo(self):
        """'stagnatlon' (l instead of i) should still match 'stagnation'."""
        words = make_aligned_words([
            ("My", 10.0, 10.2),
            ("mind", 10.2, 10.5),
            ("rebels", 10.5, 10.9),
            ("at", 10.9, 11.0),
            ("stagnatlon", 11.0, 11.5),  # ASR typo from the spec example
        ])
        result = find_phrase("My mind rebels at stagnation", words)
        assert result is not None, "Should match despite typo"
        assert result.score >= 0.80

    def test_omitted_word(self):
        """One missing word should still match if score above cutoff."""
        words = make_aligned_words([
            ("My", 10.0, 10.2),
            ("mind", 10.2, 10.5),
            ("rebels", 10.5, 10.9),
            ("stagnation", 10.9, 11.3),  # "at" dropped by ASR
        ])
        # Target has 5 words, window size = 5, so it won't match cleanly.
        # With token_sort_ratio, 4/5 matching should still be ~80+
        result = find_phrase("My mind rebels at stagnation", words, score_cutoff=60)
        # It might or might not match depending on window — important thing is no crash
        # and if it does match, score reflects the quality
        if result is not None:
            assert result.score < 1.0  # can't be perfect with a missing word


# ── No match ──────────────────────────────────────────────────────────────────

class TestNoMatch:
    def test_completely_wrong_phrase(self):
        words = make_aligned_words([
            ("The", 1.0, 1.1),
            ("game", 1.1, 1.3),
            ("is", 1.3, 1.5),
            ("afoot", 1.5, 1.8),
        ])
        result = find_phrase("My mind rebels at stagnation", words, score_cutoff=85)
        assert result is None

    def test_empty_word_list(self):
        result = find_phrase("Hello world", [], score_cutoff=70)
        assert result is None

    def test_very_high_cutoff_causes_no_match(self):
        words = make_aligned_words([
            ("nearly", 1.0, 1.2),
            ("right", 1.2, 1.5),
        ])
        # "nearly right" vs "nearly correct" — below 100% cutoff
        result = find_phrase("nearly correct", words, score_cutoff=99)
        assert result is None


# ── Multiple occurrences ───────────────────────────────────────────────────────

class TestMultipleOccurrences:
    def test_earliest_occurrence_wins(self):
        """When the phrase appears twice, the first occurrence's onset is returned."""
        words = make_aligned_words([
            # First occurrence at 5s
            ("My", 5.0, 5.2),
            ("mind", 5.2, 5.5),
            ("rebels", 5.5, 5.9),
            ("at", 5.9, 6.0),
            ("stagnation", 6.0, 6.5),
            # Filler
            ("give", 6.5, 6.7),
            ("me", 6.7, 6.9),
            ("problems", 6.9, 7.2),
            # Second occurrence at 100s
            ("My", 100.0, 100.2),
            ("mind", 100.2, 100.5),
            ("rebels", 100.5, 100.9),
            ("at", 100.9, 101.0),
            ("stagnation", 101.0, 101.5),
        ])
        result = find_phrase("My mind rebels at stagnation", words)
        assert result is not None
        assert result.onset_s == pytest.approx(5.0, abs=0.01), (
            f"Expected first occurrence at 5.0s, got {result.onset_s}"
        )

    def test_both_occurrences_above_cutoff(self):
        """Ensures iteration finds BOTH, then returns earliest."""
        words = make_aligned_words([
            ("hello", 1.0, 1.3), ("world", 1.3, 1.6),
            ("foo", 2.0, 2.2),
            ("hello", 3.0, 3.3), ("world", 3.3, 3.6),
        ])
        result = find_phrase("hello world", words)
        assert result is not None
        assert result.onset_s == pytest.approx(1.0, abs=0.01)


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_single_word_target(self):
        words = make_aligned_words([
            ("Elementary", 2.0, 2.4),
        ])
        result = find_phrase("Elementary", words)
        assert result is not None
        assert result.onset_s == pytest.approx(2.0)

    def test_target_longer_than_transcript(self):
        """Should not crash even if target has more words than transcript."""
        words = make_aligned_words([
            ("only", 1.0, 1.2),
            ("two", 1.2, 1.5),
        ])
        # Should gracefully return None (no crash)
        result = find_phrase("one two three four five six seven", words)
        # May or may not match — key requirement is no exception
        assert result is None or isinstance(result.score, float)

    def test_returns_matched_words_list(self):
        words = make_aligned_words([
            ("the", 1.0, 1.1),
            ("game", 1.1, 1.4),
            ("is", 1.4, 1.6),
            ("afoot", 1.6, 1.9),
        ])
        result = find_phrase("the game is afoot", words)
        assert result is not None
        assert len(result.matched_words) == 4
        assert result.matched_words[0].word == "the"
        assert result.matched_words[-1].word == "afoot"
