import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from backend.config import settings
from backend.llm import is_command_response
from backend.models import TranscribeMetadata
from backend.pipeline import (
    cleanup_transcript,
    describe_voice,
    get_asr,
    resolve_terms,
    transcribe_and_clean,
    transcribe_raw,
)
from backend.streaming import StreamingTranscriber
from backend.utils import audio_stats
from backend.vocabulary import _resolve_path, load_dictionary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)

app = FastAPI(title="Wisper", description="Voice dictation backend — ASR + LLM cleanup")

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def get_config():
    """Active providers and runtime config — handy for clients and debugging."""
    return {
        "asr_provider": settings.asr_provider,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.resolved_llm_model,
        "local_asr_model": settings.local_asr_model,
        "mlx_whisper_model": settings.mlx_whisper_model,
        "ollama_model": settings.ollama_model,
        "dictionary_terms": len(load_dictionary()),
    }


class DictionaryUpdate(BaseModel):
    terms: list[str]


@app.get("/api/dictionary")
async def get_dictionary():
    return {"terms": load_dictionary()}


@app.post("/api/dictionary")
async def add_dictionary_terms(update: DictionaryUpdate):
    """Append new custom-vocabulary terms (case-insensitive de-dup)."""
    existing = load_dictionary()
    seen = {t.lower() for t in existing}
    added = [t.strip() for t in update.terms if t.strip() and t.strip().lower() not in seen]
    if added:
        path = _resolve_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            if path.exists() and path.stat().st_size > 0:
                f.write("\n")
            f.write("\n".join(added))
    return {"added": added, "terms": load_dictionary()}


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    metadata: str = Form("{}"),
):
    logger = logging.getLogger("api")
    try:
        meta = TranscribeMetadata.model_validate_json(metadata)
    except Exception:
        meta = TranscribeMetadata()

    audio_data = await audio.read()
    if not audio_data:
        return JSONResponse(status_code=400, content={"error": "Empty audio file"})

    try:
        result = await transcribe_and_clean(audio_data, meta)
        return result
    except Exception as e:
        logger.exception("Transcribe failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/transcribe/raw")
async def transcribe_raw_endpoint(audio: UploadFile = File(...)):
    logger = logging.getLogger("api")
    audio_data = await audio.read()
    if not audio_data:
        return JSONResponse(status_code=400, content={"error": "Empty audio file"})

    try:
        result = await transcribe_raw(audio_data)
        return result
    except Exception as e:
        logger.exception("Raw transcribe failed")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.websocket("/ws/transcribe")
async def websocket_transcribe(ws: WebSocket):
    await ws.accept()
    logger = logging.getLogger("ws")

    try:
        cfg = json.loads(await ws.receive_text())
        meta = TranscribeMetadata.model_validate(cfg)
    except Exception:
        cfg = {}
        meta = TranscribeMetadata()

    mode = cfg.get("mode", "buffered")
    logger.info("WS session: app=%s mode=%s", meta.app_context, mode)

    if mode == "stream":
        await _handle_streaming(ws, cfg, meta)
    else:
        await _handle_buffered(ws, meta)


async def _finalize_and_send(
    ws: WebSocket, meta: TranscribeMetadata, raw_text: str, voice_hint: str, prior_timing: dict
) -> None:
    """Run LLM cleanup on the final transcript and send the terminal message."""
    logger = logging.getLogger("ws")

    if not raw_text.strip():
        await ws.send_json({
            "stage": "final", "text": "", "raw_transcript": "",
            "app_context": meta.app_context, "timing": prior_timing,
        })
        await ws.close()
        return

    try:
        cleaned, llm_timing = await cleanup_transcript(raw_text, meta, voice_hint=voice_hint)
    except Exception:
        logger.exception("LLM cleanup failed, falling back to raw transcript")
        await ws.send_json({
            "stage": "final", "text": raw_text, "raw_transcript": raw_text,
            "app_context": meta.app_context, "timing": prior_timing,
        })
        await ws.close()
        return

    all_timing = {**prior_timing, **llm_timing}
    all_timing["total"] = round(sum(all_timing.values()), 3)

    command = is_command_response(cleaned)
    if command:
        await ws.send_json({"stage": "final", **command})
    else:
        await ws.send_json({
            "stage": "final", "text": cleaned, "raw_transcript": raw_text,
            "app_context": meta.app_context, "timing": all_timing, "voice_hint": voice_hint,
        })
    await ws.close()


async def _receive_until_end(ws: WebSocket, on_bytes) -> None:
    """Pump incoming audio frames into on_bytes until END. Raises
    WebSocketDisconnect if the client drops mid-stream."""
    while True:
        message = await ws.receive()
        if message.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect()
        if message.get("bytes"):
            on_bytes(message["bytes"])
        elif message.get("text") is not None:
            text = message["text"]
            if text == "END":
                return
            try:
                if json.loads(text).get("action") == "end":
                    return
            except (json.JSONDecodeError, AttributeError):
                pass


async def _handle_buffered(ws: WebSocket, meta: TranscribeMetadata) -> None:
    """Legacy path: buffer the whole utterance, transcribe once, then clean."""
    logger = logging.getLogger("ws")
    audio_chunks: list[bytes] = []
    try:
        await _receive_until_end(ws, audio_chunks.append)
    except WebSocketDisconnect:
        logger.info("WS client disconnected during recording")
        return

    if not audio_chunks:
        await ws.send_json({"error": "No audio received"})
        await ws.close()
        return

    audio_data = b"".join(audio_chunks)
    logger.info("WS received %d bytes of audio", len(audio_data))

    try:
        raw_result = await transcribe_raw(audio_data, meta)
    except Exception as e:
        logger.exception("ASR failed")
        await ws.send_json({"error": f"ASR failed: {e}"})
        await ws.close()
        return

    await ws.send_json({
        "stage": "partial",
        "text": raw_result.text,
        "raw_transcript": raw_result.text,
        "timing": raw_result.timing,
        "confidence": raw_result.confidence,
        "language": raw_result.language,
        "voice_hint": raw_result.voice_hint,
    })
    await _finalize_and_send(ws, meta, raw_result.text, raw_result.voice_hint, raw_result.timing)


async def _handle_streaming(ws: WebSocket, cfg: dict, meta: TranscribeMetadata) -> None:
    """Streaming path: re-transcribe the rolling buffer live, then clean on END."""
    logger = logging.getLogger("ws")
    sample_rate = int(cfg.get("sample_rate", 16000))
    terms = resolve_terms(meta)

    async def transcribe(wav: bytes):
        return await get_asr().transcribe(wav, terms=terms, language=meta.language)

    st = StreamingTranscriber(
        transcribe,
        sample_rate=sample_rate,
        interval_s=settings.stream_interval_s,
        min_new_s=settings.stream_min_new_s,
        commit_silence_s=settings.stream_commit_silence_s,
    )
    stop = asyncio.Event()

    async def send_partial(result):
        await ws.send_json({
            "stage": "partial", "final": False,
            "text": result.text, "raw_transcript": result.text,
            "confidence": result.confidence, "language": result.language,
        })

    ticker = asyncio.create_task(st.tick_loop(send_partial, stop))
    disconnected = False
    try:
        await _receive_until_end(ws, st.feed)
    except WebSocketDisconnect:
        disconnected = True
        logger.info("WS client disconnected during streaming")
    finally:
        stop.set()
        ticker.cancel()
        try:
            await ticker
        except asyncio.CancelledError:
            pass

    if disconnected:
        return
    if not st.has_audio():
        await ws.send_json({"error": "No audio received"})
        await ws.close()
        return

    try:
        result = await st.finalize()
    except Exception as e:
        logger.exception("ASR failed")
        await ws.send_json({"error": f"ASR failed: {e}"})
        await ws.close()
        return

    stats = audio_stats(st.last_wav) if st.last_wav else None
    voice_hint = describe_voice(stats, len(result.text.split()), result)
    logger.info("Stream final (%.1fs audio): %s", st.buffered_seconds, result.text[:100])
    await _finalize_and_send(ws, meta, result.text, voice_hint, {})
