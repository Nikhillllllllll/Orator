import logging

from backend.asr import ASRProvider, ASRResult, get_asr_provider
from backend.llm import LLMProvider, get_llm_provider, is_command_response
from backend.models import (
    CommandResponse,
    RawTranscribeResponse,
    TranscribeMetadata,
    TranscribeResponse,
)
from backend.utils import Timer, audio_stats
from backend.vocabulary import load_dictionary, merge_terms

logger = logging.getLogger(__name__)

_asr: ASRProvider | None = None
_llm: LLMProvider | None = None


def get_asr() -> ASRProvider:
    global _asr
    if _asr is None:
        _asr = get_asr_provider()
    return _asr


def get_llm() -> LLMProvider:
    global _llm
    if _llm is None:
        _llm = get_llm_provider()
    return _llm


def resolve_terms(metadata: TranscribeMetadata) -> list[str]:
    """Server-side dictionary merged with any per-request vocabulary."""
    return merge_terms(load_dictionary(), metadata.dictionary)


def describe_voice(stats: dict[str, float] | None, word_count: int, result: ASRResult) -> str:
    """Turn cheap audio stats + ASR signals into a short voice-characteristics hint."""
    parts: list[str] = []
    if stats:
        duration = stats["duration"]
        rms = stats["rms"]
        if duration > 0 and word_count > 2:
            wps = word_count / duration
            if wps > 3.2:
                parts.append("fast pace")
            elif wps < 1.3:
                parts.append("slow, deliberate pace")
        if rms < 0.015:
            parts.append("very quiet (likely whispered — keep it discreet)")
        elif rms > 0.25:
            parts.append("loud / emphatic")
        parts.append(f"~{duration:.1f}s")
    if result.confidence is not None and result.confidence < 0.55:
        parts.append("low ASR confidence (transcript may contain errors)")
    return "; ".join(parts)


async def _run_asr(
    audio_data: bytes, metadata: TranscribeMetadata, timer: Timer
) -> tuple[ASRResult, str]:
    """Run ASR with vocabulary biasing and derive a voice-characteristics hint."""
    terms = resolve_terms(metadata)
    timer.start("asr")
    result = await get_asr().transcribe(audio_data, terms=terms, language=metadata.language)
    asr_time = timer.stop()
    logger.info("ASR completed in %.3fs: %s", asr_time, result.text[:100])

    word_count = len(result.text.split())
    voice_hint = describe_voice(audio_stats(audio_data), word_count, result)
    if voice_hint:
        logger.info("Voice: %s", voice_hint)
    return result, voice_hint


async def transcribe_raw(
    audio_data: bytes, metadata: TranscribeMetadata | None = None
) -> RawTranscribeResponse:
    meta = metadata or TranscribeMetadata()
    timer = Timer()
    result, voice_hint = await _run_asr(audio_data, meta, timer)
    return RawTranscribeResponse(
        text=result.text,
        timing=timer.stages,
        confidence=result.confidence,
        language=result.language,
        voice_hint=voice_hint,
    )


async def cleanup_transcript(
    transcript: str, metadata: TranscribeMetadata, voice_hint: str = ""
) -> tuple[str, dict[str, float]]:
    timer = Timer()
    timer.start("llm")
    cleaned = await get_llm().cleanup(
        transcript=transcript,
        app_context=metadata.app_context,
        screen_text=metadata.screen_text,
        user_style=metadata.user_style,
        dictionary=resolve_terms(metadata),
        voice_hint=voice_hint,
    )
    llm_time = timer.stop()
    logger.info("LLM cleanup completed in %.3fs", llm_time)
    return cleaned, timer.stages


async def transcribe_and_clean(
    audio_data: bytes, metadata: TranscribeMetadata
) -> TranscribeResponse | CommandResponse:
    timer = Timer()
    result, voice_hint = await _run_asr(audio_data, metadata, timer)

    if not result.text.strip():
        return TranscribeResponse(
            text="",
            raw_transcript="",
            app_context=metadata.app_context,
            timing=timer.stages,
            confidence=result.confidence,
            language=result.language,
            voice_hint=voice_hint,
        )

    timer.start("llm")
    cleaned = await get_llm().cleanup(
        transcript=result.text,
        app_context=metadata.app_context,
        screen_text=metadata.screen_text,
        user_style=metadata.user_style,
        dictionary=resolve_terms(metadata),
        voice_hint=voice_hint,
    )
    llm_time = timer.stop()
    logger.info("LLM cleanup completed in %.3fs", llm_time)

    command = is_command_response(cleaned)
    if command:
        return CommandResponse(**command)

    total = sum(timer.stages.values())
    timer.stages["total"] = round(total, 3)
    logger.info("Pipeline total: %.3fs | Stages: %s", total, timer.stages)

    return TranscribeResponse(
        text=cleaned,
        raw_transcript=result.text,
        app_context=metadata.app_context,
        timing=timer.stages,
        confidence=result.confidence,
        language=result.language,
        voice_hint=voice_hint,
    )
