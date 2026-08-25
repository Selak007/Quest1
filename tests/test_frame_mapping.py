"""
Unit tests for dialogue_locator/frame_mapping.py

Tests cover:
  - locate_frame: onset at exact PTS boundary
  - locate_frame: onset strictly between two frames
  - locate_frame: onset before first frame (edge case → frame 0)
  - locate_frame: onset after last frame (edge case → last frame)
  - locate_frame: non-uniform frame spacing (simulates dropped/duplicated frames)
  - locate_frame: single-frame list
  - locate_frame: the worked example from the spec (frame 10057)
"""
import pytest
from tests.conftest import make_frame_pts
from dialogue_locator.frame_mapping import locate_frame, FramePTS


class TestLocateFrame:

    # ── Basic containment ──────────────────────────────────────────────────────

    def test_onset_at_exact_pts_boundary(self):
        """Onset exactly equals PTS(N) → should return frame N, not N-1."""
        pts = make_frame_pts([1.000, 1.040, 1.080, 1.120, 1.160])
        result = locate_frame(1.040, pts)
        assert result.frame_number == 1
        assert result.pts_s == pytest.approx(1.040, abs=1e-6)

    def test_onset_strictly_inside_window(self):
        """Onset between PTS(N) and PTS(N+1) → frame N."""
        pts = make_frame_pts([1.000, 1.040, 1.080, 1.120])
        result = locate_frame(1.055, pts)
        assert result.frame_number == 1  # [1.040, 1.080)

    def test_onset_just_before_next_boundary(self):
        """Onset at PTS(N+1) - ε → still frame N (not N+1)."""
        pts = make_frame_pts([1.000, 1.040, 1.080])
        result = locate_frame(1.0799, pts)
        assert result.frame_number == 1

    def test_onset_at_next_boundary_is_next_frame(self):
        """Onset at exactly PTS(N+1) → returns frame N+1."""
        pts = make_frame_pts([1.000, 1.040, 1.080, 1.120])
        result = locate_frame(1.080, pts)
        assert result.frame_number == 2

    # ── Spec worked example (§5 of the architecture doc) ─────────────────────

    def test_spec_example_frame_10057(self):
        """
        From the spec:
          frame 10056 -> 402.240
          frame 10057 -> 402.280
          frame 10058 -> 402.320
          frame 10059 -> 402.360
          onset = 402.318
          Expected: frame 10057  (window [402.280, 402.320))
        """
        pts = [
            FramePTS(frame_number=10056, pts_s=402.240),
            FramePTS(frame_number=10057, pts_s=402.280),
            FramePTS(frame_number=10058, pts_s=402.320),
            FramePTS(frame_number=10059, pts_s=402.360),
        ]
        result = locate_frame(402.318, pts)
        assert result.frame_number == 10057
        assert result.pts_s == pytest.approx(402.280, abs=1e-6)
        assert result.next_pts_s == pytest.approx(402.320, abs=1e-6)

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_onset_before_first_frame(self):
        """Onset before any frame → return frame 0, no crash."""
        pts = make_frame_pts([5.000, 5.040, 5.080])
        result = locate_frame(4.500, pts)
        assert result.frame_number == 0

    def test_onset_after_last_frame(self):
        """Onset at or past last PTS → return last frame, no crash."""
        pts = make_frame_pts([1.000, 1.040, 1.080])
        result = locate_frame(99.0, pts)
        assert result.frame_number == 2

    def test_onset_equals_last_pts(self):
        """Onset exactly equal to last PTS → return last frame."""
        pts = make_frame_pts([1.000, 1.040, 1.080])
        result = locate_frame(1.080, pts)
        assert result.frame_number == 2

    def test_single_frame_list(self):
        """Only one frame in list → return it without indexing error."""
        pts = [FramePTS(frame_number=42, pts_s=10.0)]
        result = locate_frame(10.5, pts)
        assert result.frame_number == 42

    # ── Non-uniform spacing (dropped/duplicated frames) ───────────────────────

    def test_non_uniform_spacing_large_gap(self):
        """
        Simulates a dropped frame: gap between frame 2 and frame 3 is double-size.
        onset falls inside the large gap → should map to frame 2.
        """
        pts = [
            FramePTS(frame_number=0, pts_s=0.000),
            FramePTS(frame_number=1, pts_s=0.040),
            FramePTS(frame_number=2, pts_s=0.080),
            # Frame 3 is at 0.160 instead of 0.120 — one frame dropped
            FramePTS(frame_number=3, pts_s=0.160),
            FramePTS(frame_number=4, pts_s=0.200),
        ]
        # onset at 0.100 is inside [0.080, 0.160) → frame 2
        result = locate_frame(0.100, pts)
        assert result.frame_number == 2

    def test_non_uniform_spacing_duplicate_pts(self):
        """
        Two frames with same PTS (duplicated frame).
        onset at that PTS → lands on first of the pair.
        """
        pts = [
            FramePTS(frame_number=0, pts_s=1.000),
            FramePTS(frame_number=1, pts_s=1.040),
            FramePTS(frame_number=2, pts_s=1.040),  # duplicate
            FramePTS(frame_number=3, pts_s=1.080),
        ]
        # onset at 1.040 → frame 1 (first to satisfy PTS(N) <= 1.040 < PTS(N+1))
        result = locate_frame(1.040, pts)
        assert result.frame_number in (1, 2)  # either is acceptable for a duplicate

    # ── Window bounds exposed ─────────────────────────────────────────────────

    def test_next_pts_is_correct(self):
        pts = make_frame_pts([0.000, 0.040, 0.080, 0.120])
        result = locate_frame(0.050, pts)
        assert result.pts_s == pytest.approx(0.040, abs=1e-6)
        assert result.next_pts_s == pytest.approx(0.080, abs=1e-6)

    def test_frame_number_preserved(self):
        """Non-sequential frame numbers (container numbering) are preserved."""
        pts = [
            FramePTS(frame_number=999, pts_s=10.000),
            FramePTS(frame_number=1000, pts_s=10.040),
            FramePTS(frame_number=1001, pts_s=10.080),
        ]
        result = locate_frame(10.050, pts)
        assert result.frame_number == 1000
