"""Tests for the empty-transcript classifier (`describe_no_speech`)."""

from client.diagnostics import SILENT_FLOOR, describe_no_speech

THRESHOLD = 0.01  # backend's default silence gate


def test_no_level_gives_generic_message():
    assert describe_no_speech(None, None) == "No speech detected."


def test_silent_mic_points_at_permission():
    msg = describe_no_speech(0.001, THRESHOLD)
    assert "silent" in msg
    assert "permission" in msg.lower()
    assert "miccheck" in msg


def test_quiet_speech_below_threshold():
    msg = describe_no_speech(0.008, THRESHOLD)
    assert "quiet" in msg
    assert "0.010" in msg  # the threshold is shown
    assert "closer" in msg.lower()


def test_audible_but_empty():
    msg = describe_no_speech(0.04, THRESHOLD)
    assert "audible" in msg
    assert "empty" in msg


def test_boundary_at_silent_floor():
    # Exactly at the floor is no longer "silent" — it's classified as quiet.
    assert "silent" in describe_no_speech(SILENT_FLOOR - 0.0001, THRESHOLD)
    assert "quiet" in describe_no_speech(SILENT_FLOOR, THRESHOLD)


def test_above_floor_without_threshold_is_audible():
    # No threshold known → can't call it "quiet", so fall through to audible.
    msg = describe_no_speech(0.008, None)
    assert "audible" in msg
