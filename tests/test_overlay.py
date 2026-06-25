"""Tests for the floating overlay's pure styling logic (no tkinter needed)."""

from client.overlay import STYLES, style_for


def test_known_statuses_have_distinct_colors():
    idle = style_for("idle")
    recording = style_for("recording")
    processing = style_for("processing")
    assert idle != recording != processing
    # Each maps to a (glyph, hex-color) pair.
    for glyph, color in (idle, recording, processing):
        assert isinstance(glyph, str) and color.startswith("#")


def test_recording_is_red():
    assert style_for("recording") == STYLES["recording"]
    assert style_for("recording")[1] == "#c0392b"


def test_unknown_status_falls_back_to_idle():
    assert style_for("bogus") == STYLES["idle"]
