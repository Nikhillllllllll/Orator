"""Tests for live streaming: WAV framing and the rolling-buffer transcriber."""

import asyncio
import wave

import numpy as np
import pytest

from backend.asr import ASRResult
from backend.streaming import StreamingTranscriber
from backend.utils import pcm_to_wav


def _silence_pcm(seconds: float, sample_rate: int = 16000) -> bytes:
    return (np.zeros(int(seconds * sample_rate), dtype=np.int16)).tobytes()


def _speech_pcm(seconds: float, sample_rate: int = 16000) -> bytes:
    # Loud-enough samples to read as "speech" by the energy VAD.
    return (np.full(int(seconds * sample_rate), 8000, dtype=np.int16)).tobytes()


def test_pcm_to_wav_roundtrips():
    pcm = (np.arange(16000, dtype=np.int16)).tobytes()
    wav = pcm_to_wav(pcm, sample_rate=16000)
    import io

    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.readframes(wf.getnframes()) == pcm


@pytest.mark.asyncio
async def test_finalize_transcribes_buffer():
    calls = []

    async def fake_transcribe(wav: bytes) -> ASRResult:
        calls.append(len(wav))
        return ASRResult(text="hello world", confidence=0.9)

    st = StreamingTranscriber(fake_transcribe, sample_rate=16000)
    st.feed(_speech_pcm(1.0))
    result = await st.finalize()

    assert result.text == "hello world"
    assert len(calls) == 1
    assert st.last_wav is not None


@pytest.mark.asyncio
async def test_silence_is_not_transcribed():
    """Silent audio must not hit the ASR (Whisper hallucinates on silence)."""
    async def fake_transcribe(wav: bytes) -> ASRResult:
        raise AssertionError("should not transcribe silence")

    st = StreamingTranscriber(fake_transcribe, sample_rate=16000)
    st.feed(_silence_pcm(2.0))
    result = await st.finalize()
    assert result.text == ""


@pytest.mark.asyncio
async def test_finalize_with_no_audio_is_empty():
    async def fake_transcribe(wav: bytes) -> ASRResult:
        raise AssertionError("should not transcribe empty buffer")

    st = StreamingTranscriber(fake_transcribe, sample_rate=16000)
    result = await st.finalize()
    assert result.text == ""


@pytest.mark.asyncio
async def test_tick_loop_emits_partials_as_audio_grows():
    counter = {"n": 0}

    async def fake_transcribe(wav: bytes) -> ASRResult:
        counter["n"] += 1
        return ASRResult(text=f"partial {counter['n']}")

    partials: list[str] = []

    async def send_partial(result: ASRResult):
        partials.append(result.text)

    st = StreamingTranscriber(
        fake_transcribe, sample_rate=16000, interval_s=0.02, min_new_s=0.1
    )
    stop = asyncio.Event()
    ticker = asyncio.create_task(st.tick_loop(send_partial, stop))

    # Feed audio in bursts larger than min_new so each tick re-transcribes.
    for _ in range(3):
        st.feed(_speech_pcm(0.2))
        await asyncio.sleep(0.05)

    stop.set()
    ticker.cancel()
    try:
        await ticker
    except asyncio.CancelledError:
        pass

    assert len(partials) >= 1
    assert partials[-1].startswith("partial")


@pytest.mark.asyncio
async def test_segment_commits_on_pause():
    """A speech segment followed by enough silence should be frozen (committed),
    so later ticks no longer re-transcribe it."""
    counter = {"n": 0}

    async def fake_transcribe(wav: bytes) -> ASRResult:
        counter["n"] += 1
        return ASRResult(text=f"seg{counter['n']}")

    partials: list[str] = []

    async def send_partial(result: ASRResult):
        partials.append(result.text)

    st = StreamingTranscriber(
        fake_transcribe,
        sample_rate=16000,
        interval_s=0.03,
        min_new_s=0.1,
        commit_silence_s=0.4,
        min_segment_s=0.8,
    )
    stop = asyncio.Event()
    ticker = asyncio.create_task(st.tick_loop(send_partial, stop))

    # 1.2s of speech, then 0.6s of silence -> should trigger a commit.
    st.feed(_speech_pcm(1.2))
    st.feed(_silence_pcm(0.6))
    await asyncio.sleep(0.15)

    stop.set()
    ticker.cancel()
    try:
        await ticker
    except asyncio.CancelledError:
        pass

    assert st.committed_text != ""           # something was committed
    assert st.committed_seconds > 0          # the commit offset advanced


@pytest.mark.asyncio
async def test_tick_loop_skips_when_no_new_audio():
    async def fake_transcribe(wav: bytes) -> ASRResult:
        return ASRResult(text="x")

    sent = []

    async def send_partial(result: ASRResult):
        sent.append(result.text)

    st = StreamingTranscriber(
        fake_transcribe, sample_rate=16000, interval_s=0.02, min_new_s=0.1
    )
    stop = asyncio.Event()
    ticker = asyncio.create_task(st.tick_loop(send_partial, stop))
    # No audio fed at all — nothing should be sent.
    await asyncio.sleep(0.1)
    stop.set()
    ticker.cancel()
    try:
        await ticker
    except asyncio.CancelledError:
        pass

    assert sent == []
