import asyncio
import io
import time
import wave

import numpy as np

MIME_MAP = {
    b"\x1a\x45\xdf\xa3": "audio/webm",
    b"RIFF": "audio/wav",
    b"ID3": "audio/mp3",
    b"\xff\xfb": "audio/mp3",
    b"\xff\xf3": "audio/mp3",
    b"OggS": "audio/ogg",
    b"fLaC": "audio/flac",
}


def detect_audio_format(data: bytes) -> str:
    for magic, mime in MIME_MAP.items():
        if data[: len(magic)] == magic:
            return mime
    if len(data) > 4 and data[4:8] == b"ftyp":
        return "audio/mp4"
    return "audio/wav"


async def convert_to_wav(audio_data: bytes, source_format: str) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-i",
        "pipe:0",
        "-f",
        "wav",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "pipe:1",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=audio_data)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {stderr.decode()}")
    return stdout


def pcm_to_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit little-endian PCM samples in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Parse 16-bit PCM WAV bytes into a mono float32 array and its sample rate."""
    with wave.open(io.BytesIO(data), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"unsupported sample width: {sample_width * 8}-bit")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sample_rate


async def decode_to_float32_mono16k(audio_data: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes to a mono float32 array at 16 kHz.

    Fast path: 16 kHz mono 16-bit WAV is parsed directly. Anything else is
    routed through ffmpeg (handles webm/ogg/mp3/mp4 and resampling).
    """
    mime = detect_audio_format(audio_data)
    if mime == "audio/wav":
        try:
            audio, sample_rate = _read_wav(audio_data)
            if sample_rate == 16000:
                return audio
        except (wave.Error, ValueError):
            pass  # fall through to ffmpeg
    wav = await convert_to_wav(audio_data, mime)
    audio, _ = _read_wav(wav)
    return audio


def audio_stats(audio_data: bytes) -> dict[str, float] | None:
    """Cheap voice characteristics from 16-bit WAV (duration, loudness).

    Returns None for non-WAV input (no ffmpeg here — keep it free for every
    request). Used to make cleanup voice-aware (e.g. detect whispering).
    """
    try:
        audio, sample_rate = _read_wav(audio_data)
    except (wave.Error, ValueError, EOFError):
        return None
    if audio.size == 0 or sample_rate <= 0:
        return None
    return {
        "duration": round(audio.size / sample_rate, 3),
        "rms": round(float(np.sqrt(np.mean(audio**2))), 4),
        "peak": round(float(np.max(np.abs(audio))), 4),
    }


class Timer:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._current = name
        self._start = time.perf_counter()

    def stop(self) -> float:
        elapsed = round(time.perf_counter() - self._start, 3)
        self.stages[self._current] = elapsed
        return elapsed
