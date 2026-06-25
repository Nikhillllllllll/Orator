"""Live streaming ASR with VAD-segmented committing.

True streaming Whisper APIs don't exist for the cloud providers we use, and the
local engines are batch. So we approximate streaming by re-transcribing the
accumulated audio on a cadence and emitting each result as a partial.

The naive version re-transcribes the *entire* utterance every tick — O(n²) over
its length. Instead we split the audio at natural pauses: once a segment is
followed by enough silence it is **committed** (transcribed one final time and
frozen), and only the still-growing *active* tail is re-transcribed thereafter.
That bounds the work to the current segment and stops earlier text from
flickering. Silence is detected by RMS energy, so it needs no extra dependency
and works identically for every provider (cloud or local).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

import numpy as np

from backend.asr import ASRResult
from backend.utils import pcm_to_wav

logger = logging.getLogger(__name__)


def _rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


class StreamingTranscriber:
    def __init__(
        self,
        transcribe: Callable[[bytes], Awaitable[ASRResult]],
        sample_rate: int = 16000,
        interval_s: float = 1.2,
        min_new_s: float = 0.8,
        commit_silence_s: float = 0.6,
        min_segment_s: float = 1.0,
        silence_rms: float = 0.01,
    ) -> None:
        self._transcribe = transcribe          # async (wav_bytes) -> ASRResult
        self.sample_rate = sample_rate
        self.interval_s = interval_s
        self._bytes_per_s = sample_rate * 2     # int16 mono
        self._min_new_bytes = int(min_new_s * self._bytes_per_s)
        self._commit_silence_s = commit_silence_s
        self._min_segment_bytes = int(min_segment_s * self._bytes_per_s)
        self._silence_rms = silence_rms

        self._buffer = bytearray()
        self._commit_offset = 0                 # bytes already committed (frozen)
        self._active_transcribed_len = 0        # active bytes covered by last tick
        self.committed_text = ""
        self._busy = False
        self.last_result = ASRResult(text="")
        self.last_wav: bytes | None = None

    # ---- input ----

    def feed(self, pcm: bytes) -> None:
        self._buffer.extend(pcm)

    def has_audio(self) -> bool:
        return len(self._buffer) > 0

    @property
    def buffered_seconds(self) -> float:
        return len(self._buffer) / self._bytes_per_s

    @property
    def committed_seconds(self) -> float:
        return self._commit_offset / self._bytes_per_s

    # ---- helpers ----

    def _active(self) -> bytes:
        return bytes(self._buffer[self._commit_offset:])

    def _combine(self, active_text: str) -> str:
        return " ".join(p for p in (self.committed_text, active_text) if p).strip()

    def _trailing_silence(self, region: bytes) -> bool:
        window = int(self._commit_silence_s * self._bytes_per_s)
        if len(region) < window:
            return False
        return _rms(region[-window:]) < self._silence_rms

    async def _transcribe_active(self) -> tuple[ASRResult | None, int]:
        snap_len = len(self._buffer)
        active = bytes(self._buffer[self._commit_offset:snap_len])
        if not active:
            return None, snap_len
        if _rms(active) < self._silence_rms:
            # Whisper hallucinates ("thank you", "thanks for watching") on silence.
            # Skip the call entirely so silent audio yields no text.
            self._active_transcribed_len = len(active)
            return None, snap_len
        self._busy = True
        try:
            self.last_result = await self._transcribe(pcm_to_wav(active, self.sample_rate))
            self._active_transcribed_len = len(active)
            return self.last_result, snap_len
        finally:
            self._busy = False

    # ---- streaming loop ----

    async def tick_loop(
        self, send_partial: Callable[[ASRResult], Awaitable[None]], stop: asyncio.Event
    ) -> None:
        try:
            while not stop.is_set():
                await asyncio.sleep(self.interval_s)
                active_len = len(self._buffer) - self._commit_offset
                if self._busy or (active_len - self._active_transcribed_len) < self._min_new_bytes:
                    continue
                try:
                    result, snap_len = await self._transcribe_active()
                except Exception:  # noqa: BLE001 — a failed partial shouldn't kill the stream
                    logger.exception("Streaming partial transcription failed")
                    continue
                if result is None:
                    continue

                active_text = result.text.strip()
                region = bytes(self._buffer[self._commit_offset:snap_len])
                committed_now = (
                    bool(active_text)
                    and len(region) >= self._min_segment_bytes
                    and self._trailing_silence(region)
                )
                if committed_now:
                    self.committed_text = self._combine(active_text)
                    self._commit_offset = snap_len
                    self._active_transcribed_len = 0
                    partial_text = self.committed_text
                    logger.debug("Committed segment @ %.1fs: %s", snap_len / self._bytes_per_s, active_text)
                else:
                    partial_text = self._combine(active_text)

                if partial_text:
                    await send_partial(
                        ASRResult(
                            text=partial_text,
                            confidence=result.confidence,
                            language=result.language,
                        )
                    )
        except asyncio.CancelledError:
            pass

    async def finalize(self) -> ASRResult:
        """Transcribe the remaining active tail once and return the full transcript."""
        while self._busy:  # let an in-flight tick finish to avoid overlapping model calls
            await asyncio.sleep(0.02)
        if self._active():
            result, _ = await self._transcribe_active()
            if result is not None and result.text.strip():
                self.committed_text = self._combine(result.text.strip())
                self._commit_offset = len(self._buffer)
        self.last_wav = pcm_to_wav(bytes(self._buffer), self.sample_rate) if self._buffer else None
        return ASRResult(
            text=self.committed_text,
            confidence=self.last_result.confidence,
            language=self.last_result.language,
        )
