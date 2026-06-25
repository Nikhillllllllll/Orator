"""Turn an empty transcription into an actionable explanation.

An empty result has three very different causes that otherwise look identical to
the user: a silent mic (permission / wrong device), speech too quiet to clear
the backend's silence gate, or audible audio that still transcribed to nothing.
Given the measured audio level (RMS, 0..1) and the backend's silence threshold,
say which one it was.
"""

# Below this, the signal is indistinguishable from a dead or muted mic.
SILENT_FLOOR = 0.005


def describe_no_speech(rms: float | None, threshold: float | None) -> str:
    """Build the user-facing message for an empty transcript."""
    base = "No speech detected"
    if rms is None:
        return f"{base}."
    if rms < SILENT_FLOOR:
        return (
            f"{base} — the mic was silent (level {rms:.3f}). Check microphone "
            "permission and the selected input device "
            "(run: uv run python client/miccheck.py)."
        )
    if threshold is not None and rms < threshold:
        return (
            f"{base} — audio was very quiet (level {rms:.3f}, below the "
            f"{threshold:.3f} detection threshold). Move closer or raise input gain."
        )
    return (
        f"{base} — audio was audible (level {rms:.3f}) but came back empty. "
        "Likely background noise or an ASR hiccup; try again."
    )
