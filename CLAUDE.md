# Wisper — Voice Dictation Backend

## Quick Start

```bash
cp .env.example .env   # fill in API keys (or leave blank to run fully offline)
uv run uvicorn backend.main:app --reload --port 8000
```

### Fully offline (local models)

```bash
uv sync --extra local --extra mlx        # MLX installs only on Apple Silicon
ollama serve && ollama pull llama3.2     # local LLM
# in .env:  ASR_PROVIDER=local   LLM_PROVIDER=ollama
```

No cloud keys? The provider selection auto-falls back to local Whisper + Ollama,
so the pipeline still works offline.

## Commands

- **Dictate (default)**: `uv run dictate` — `client/launcher.py` starts the backend in the background, runs the hotkey client, and stops the backend on exit. Reuses an already-running backend.
- **Run server only**: `uv run uvicorn backend.main:app --reload --port 8000`
- **Run tests**: `uv run pytest tests/ -v`
- **Run single test**: `uv run pytest tests/test_llm.py::test_name -v`
- **Install local ASR**: `uv sync --extra local --extra mlx`

## Architecture

```
audio → ASR → LLM cleanup → cleaned text (or command JSON)
        │      │
        │      └ Groq / OpenAI / Gemini / Anthropic / Ollama (local)
        └ Groq Whisper / Deepgram / faster-whisper (local) / MLX Whisper (Apple Silicon)
```

- `backend/config.py` — Settings from env vars (cloud + local providers, vocabulary path)
- `backend/asr.py` — ASR providers + `ASRResult` (text, confidence, language); cloud & local, term biasing
- `backend/llm.py` — LLM cleanup; strong context/voice-aware prompt; OpenAI-compatible base shared by Groq/OpenAI/Ollama
- `backend/vocabulary.py` — Custom dictionary load/merge; biases ASR and corrects LLM output
- `backend/pipeline.py` — Orchestrates ASR → LLM with timing, vocabulary, and voice-hint derivation
- `backend/streaming.py` — `StreamingTranscriber`: VAD-segmented streaming (commit on pause, re-transcribe only the active tail)
- `backend/main.py` — FastAPI endpoints (`/api/transcribe`, `/ws/transcribe`, `/api/config`, `/api/dictionary`)
- `backend/models.py` — Pydantic request/response models
- `backend/utils.py` — Audio format detection, WAV decode/stats, PCM→WAV, timing, ffmpeg

## WebSocket protocol (`/ws/transcribe`)

First message is a JSON config; then audio frames; then `"END"`.

- **Buffered** (default, back-compat): config without `mode`; send one WAV blob; get one `partial` (raw) + one `final` (cleaned).
- **Streaming** (`{"mode": "stream", "sample_rate": 16000}`): send raw int16 PCM chunks live; get repeated `partial` messages (`{"stage":"partial","final":false,...}`), then one `final` after `END`. Cadence tuned via `STREAM_INTERVAL_S` / `STREAM_MIN_NEW_S` / `STREAM_COMMIT_SILENCE_S`.

Streaming is **VAD-segmented**: a segment followed by `STREAM_COMMIT_SILENCE_S` of silence is committed (frozen, never re-transcribed), and only the growing active tail is re-transcribed each tick. This bounds ASR cost on long dictation (O(segment²) not O(utterance²)) and keeps the committed prefix from flickering. The desktop client uses streaming; cleanup (LLM) still runs once, on the final transcript.

## Key features

- **Local-first option**: faster-whisper (portable) + MLX Whisper (Apple Silicon, auto-selected) + Ollama. Works with zero API keys.
- **Live streaming**: rolling-buffer re-transcription emits partial text while you speak — works with any provider, cloud or local (no native streaming API required).
- **Custom vocabulary**: `dictionary.txt` (or per-request) biases ASR decoding *and* tells the LLM how to spell names/jargon. Managed via `GET/POST /api/dictionary`.
- **Voice-aware cleanup**: cheap audio stats (duration, loudness) + ASR confidence produce a voice hint (e.g. "whispered", "fast pace") fed to the prompt.
- **Strong prompt**: few-shot examples, disfluency/self-correction handling, spoken-punctuation conversion, per-app formatting profiles, and command mode.

## Conventions

- Async throughout, type hints everywhere; local model inference runs via `asyncio.to_thread`
- Pydantic models for all schemas
- Groq is the default cloud provider (fastest latency); local providers are lazy-imported and degrade gracefully
- Log timing for every pipeline stage
- Providers select by env var with sensible fallbacks (cloud key present → cloud; else → local)
