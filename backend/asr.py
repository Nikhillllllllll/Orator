import asyncio
import logging
import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from backend.config import settings
from backend.utils import decode_to_float32_mono16k, detect_audio_format
from backend.vocabulary import asr_bias_prompt

logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """Transcript plus optional voice-aware signals (when the provider reports them)."""

    text: str
    confidence: float | None = None      # 0..1; None if the provider doesn't report it
    language: str | None = None
    no_speech_prob: float | None = None


def _confidence_from_logprobs(logprobs: list[float]) -> float | None:
    if not logprobs:
        return None
    return round(float(np.exp(np.mean(logprobs))), 3)


class ASRProvider(ABC):
    @abstractmethod
    async def transcribe(
        self, audio_data: bytes, terms: list[str] | None = None, language: str | None = None
    ) -> ASRResult: ...


# ─────────────────────────── Cloud providers ───────────────────────────


class GroqWhisperASR(ASRProvider):
    def __init__(self) -> None:
        from groq import AsyncGroq

        self.client = AsyncGroq(api_key=settings.groq_api_key)

    async def transcribe(
        self, audio_data: bytes, terms: list[str] | None = None, language: str | None = None
    ) -> ASRResult:
        mime = detect_audio_format(audio_data)
        ext = mime.split("/")[-1]
        if ext == "mpeg":
            ext = "mp3"
        kwargs: dict = {
            "model": "whisper-large-v3-turbo",
            "file": (f"audio.{ext}", audio_data),
            "response_format": "text",
        }
        if terms and (bias := asr_bias_prompt(terms)):
            kwargs["prompt"] = bias
        if language:
            kwargs["language"] = language
        response = await self.client.audio.transcriptions.create(**kwargs)
        return ASRResult(text=str(response).strip(), language=language)


class DeepgramASR(ASRProvider):
    def __init__(self) -> None:
        from deepgram import DeepgramClient

        self.client = DeepgramClient(settings.deepgram_api_key)

    async def transcribe(
        self, audio_data: bytes, terms: list[str] | None = None, language: str | None = None
    ) -> ASRResult:
        from deepgram import PrerecordedOptions

        mime = detect_audio_format(audio_data)
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            language=language or "en",
            keywords=list(terms) if terms else None,
        )
        response = await self.client.listen.asyncrest.v("1").transcribe_file(
            {"buffer": audio_data, "mimetype": mime},
            options,
        )
        alt = response.results.channels[0].alternatives[0]
        return ASRResult(
            text=alt.transcript,
            confidence=getattr(alt, "confidence", None),
            language=language or "en",
        )


# ─────────────────────────── Local providers ───────────────────────────


class FasterWhisperASR(ASRProvider):
    """Offline Whisper via CTranslate2. Runs on CPU or CUDA, cross-platform."""

    def __init__(self) -> None:
        from faster_whisper import WhisperModel

        device = settings.local_asr_device
        compute_type = settings.local_asr_compute_type
        if compute_type == "auto":
            compute_type = "int8" if device in ("cpu", "auto") else "float16"
        logger.info(
            "Loading faster-whisper '%s' (device=%s, compute=%s)…",
            settings.local_asr_model, device, compute_type,
        )
        self.model = WhisperModel(
            settings.local_asr_model, device=device, compute_type=compute_type
        )

    async def transcribe(
        self, audio_data: bytes, terms: list[str] | None = None, language: str | None = None
    ) -> ASRResult:
        audio = await decode_to_float32_mono16k(audio_data)
        return await asyncio.to_thread(self._run, audio, terms, language)

    def _run(self, audio: np.ndarray, terms: list[str] | None, language: str | None) -> ASRResult:
        lang = language or (settings.local_asr_language or None)
        segments, info = self.model.transcribe(
            audio,
            language=lang,
            initial_prompt=asr_bias_prompt(terms or []) or None,
            vad_filter=True,
            beam_size=5,
        )
        texts: list[str] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        for seg in segments:
            texts.append(seg.text)
            logprobs.append(seg.avg_logprob)
            no_speech.append(seg.no_speech_prob)
        return ASRResult(
            text=" ".join(t.strip() for t in texts).strip(),
            confidence=_confidence_from_logprobs(logprobs),
            language=info.language,
            no_speech_prob=round(float(np.mean(no_speech)), 3) if no_speech else None,
        )


class MLXWhisperASR(ASRProvider):
    """Offline Whisper optimized for Apple Silicon via MLX."""

    def __init__(self) -> None:
        import mlx_whisper  # noqa: F401 — fail fast if not installed

        self.model = settings.mlx_whisper_model
        logger.info("Using MLX Whisper '%s'", self.model)

    async def transcribe(
        self, audio_data: bytes, terms: list[str] | None = None, language: str | None = None
    ) -> ASRResult:
        audio = await decode_to_float32_mono16k(audio_data)
        return await asyncio.to_thread(self._run, audio, terms, language)

    def _run(self, audio: np.ndarray, terms: list[str] | None, language: str | None) -> ASRResult:
        import mlx_whisper

        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model,
            initial_prompt=asr_bias_prompt(terms or []) or None,
            language=language or (settings.local_asr_language or None),
        )
        segments = result.get("segments") or []
        logprobs = [s["avg_logprob"] for s in segments if s.get("avg_logprob") is not None]
        no_speech = [s["no_speech_prob"] for s in segments if s.get("no_speech_prob") is not None]
        return ASRResult(
            text=result.get("text", "").strip(),
            confidence=_confidence_from_logprobs(logprobs),
            language=result.get("language"),
            no_speech_prob=round(float(np.mean(no_speech)), 3) if no_speech else None,
        )


# ─────────────────────────── Selection ───────────────────────────


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _make_local_asr(preference: str) -> ASRProvider:
    """Pick a local backend. 'local'/'whisper' auto-select MLX on Apple Silicon."""
    if preference == "faster-whisper":
        return FasterWhisperASR()
    if preference == "mlx":
        return MLXWhisperASR()
    # auto
    if _is_apple_silicon():
        try:
            return MLXWhisperASR()
        except Exception as e:  # noqa: BLE001 — MLX missing/unsupported, degrade gracefully
            logger.warning("MLX Whisper unavailable (%s); using faster-whisper", e)
    return FasterWhisperASR()


def get_asr_provider() -> ASRProvider:
    provider = settings.asr_provider.lower()

    if provider in ("local", "whisper", "faster-whisper", "mlx"):
        logger.info("Using local ASR (%s)", provider)
        return _make_local_asr(provider)
    if provider == "groq" and settings.groq_api_key:
        logger.info("Using Groq Whisper ASR")
        return GroqWhisperASR()
    if provider == "deepgram" and settings.deepgram_api_key:
        logger.info("Using Deepgram ASR")
        return DeepgramASR()

    # Fall back to any configured cloud key, then to fully-local.
    if settings.groq_api_key:
        logger.info("Falling back to Groq Whisper ASR")
        return GroqWhisperASR()
    if settings.deepgram_api_key:
        logger.info("Falling back to Deepgram ASR")
        return DeepgramASR()
    logger.info("No ASR key configured — falling back to local Whisper")
    return _make_local_asr("local")
