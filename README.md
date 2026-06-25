# Orator

Voice dictation backend that pipes audio through ASR and LLM cleanup to produce polished text. Works fully offline or with cloud providers.

```
audio -> ASR -> LLM cleanup -> cleaned text (or command JSON)
          |      |
          |      +-- Groq / OpenAI / Gemini / Anthropic / Ollama (local)
          +-- Groq Whisper / Deepgram / faster-whisper (local) / MLX Whisper (Apple Silicon)
```

## Features

- **Local-first** — faster-whisper + MLX Whisper + Ollama. Zero API keys required.
- **Live streaming** — rolling-buffer re-transcription emits partial text while you speak, works with any provider.
- **Custom vocabulary** — `dictionary.txt` biases ASR decoding and tells the LLM how to spell names, jargon, and acronyms.
- **Voice-aware cleanup** — audio stats (duration, loudness, pace) feed the LLM prompt for better context.
- **Multiple providers** — swap ASR and LLM providers via env vars. Groq is the default for lowest latency.

## Quick Start

### Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- [ffmpeg](https://ffmpeg.org/) (audio decoding) — `brew install ffmpeg` on macOS
- **One API key** (free) — or nothing at all if you run fully offline (see below)

### Install and run

```bash
git clone https://github.com/Nikhillllllllll/Orator.git
cd Orator
cp .env.example .env          # then add your API key (next step)
uv sync
uv run dictate                # starts the backend + dictation in one command
```

**Add an API key.** Open `.env` and paste a key into one line. The fastest is a
free **Groq** key from [console.groq.com](https://console.groq.com/keys) (used
for both speech-to-text and cleanup by default):

```
GROQ_API_KEY=gsk_your_key_here
```

Prefer no keys at all? Skip this and jump to [Fully offline](#fully-offline-local-models).

`uv run dictate` ensures the backend is running, then starts the desktop client.
Press the hotkey (**Left Ctrl + Left Shift**) to start recording, press it again
to stop; cleaned text is pasted at your cursor.

The backend runs as a **persistent, detached service** — it keeps running when
you quit the client or close the terminal, and is reused next time. Manage it
explicitly:

```bash
uv run dictate --status   # is the backend running?
uv run dictate --stop     # stop the backend
```

Other knobs: `ORATOR_DEBUG=1` surfaces backend logs and otherwise-silent client
errors. Logs are written to `logs/backend.log` and `logs/client.log`.

**Permissions** (first run): grant **Microphone** access, and on macOS also
**Accessibility** (System Settings → Privacy & Security → Accessibility) so
auto-paste works — otherwise text is left on the clipboard to paste manually.
On Linux, install a clipboard helper (`wl-clipboard`, `xclip`, or `xsel`).

### Backend only (API + web UI)

Prefer to run the server by itself — for the browser UI at
`http://localhost:8000`, or to host the API for remote clients:

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

### Fully offline (local models)

```bash
uv sync --extra local --extra mlx        # MLX installs only on Apple Silicon

# Start Ollama and pull a model
ollama serve
ollama pull llama3.2
```

Set in `.env`:

```
ASR_PROVIDER=local
LLM_PROVIDER=ollama
```

No cloud keys needed — the pipeline auto-falls back to local Whisper + Ollama.

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the full list.

| Variable | Default | Description |
|---|---|---|
| `ASR_PROVIDER` | `groq` | `groq` \| `deepgram` \| `local` \| `faster-whisper` \| `mlx` |
| `LLM_PROVIDER` | `groq` | `groq` \| `openai` \| `anthropic` \| `gemini` \| `ollama` |
| `LLM_MODEL` | per-provider default | Override the LLM model name |
| `DICTIONARY_PATH` | `dictionary.txt` | Path to custom vocabulary file |
| `LOCAL_ASR_MODEL` | `small.en` | Whisper model size for local ASR |

## API

### REST

**`POST /api/transcribe`** — Upload audio, get cleaned transcript.

```bash
curl -X POST http://localhost:8000/api/transcribe \
  -F "audio=@recording.wav"
```

### WebSocket

**`/ws/transcribe`** — Real-time streaming transcription.

- **Buffered mode** (default): send one WAV blob, get a `partial` (raw) + `final` (cleaned) message.
- **Streaming mode**: send `{"mode": "stream", "sample_rate": 16000}` as the first message, then raw PCM chunks, then `"END"`. Get repeated `partial` messages as you speak, then one `final`.

### Other endpoints

- `GET /api/config` — Current server configuration
- `GET /api/dictionary` — Current custom vocabulary
- `POST /api/dictionary` — Update custom vocabulary

## Custom Vocabulary

Create a `dictionary.txt` with one term per line:

```
Kubernetes
PostgreSQL
FastAPI
```

These terms bias ASR decoding and are included in the LLM cleanup prompt so names and jargon are spelled correctly.

## Project Structure

```
backend/
  main.py          — FastAPI endpoints (REST + WebSocket)
  config.py        — Settings from env vars
  asr.py           — ASR providers (cloud + local)
  llm.py           — LLM cleanup with voice-aware prompts
  pipeline.py      — Orchestrates ASR -> LLM with timing and vocabulary
  streaming.py     — VAD-segmented streaming transcriber
  vocabulary.py    — Custom dictionary loading and merging
  models.py        — Pydantic request/response schemas
  utils.py         — Audio format detection, WAV decode, timing
tests/             — pytest test suite
client/            — Cross-platform desktop client (macOS, Windows, Linux)
  dictate.py       — Hotkey listener + streaming dictation (platform-neutral)
  platforms.py     — Per-OS clipboard, paste, app/tab detection
  toast.py         — Floating status HUD
dictionary.txt     — Custom vocabulary (names, jargon, acronyms)
```

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Run a single test
uv run pytest tests/test_llm.py::test_name -v

# Run server with auto-reload
uv run uvicorn backend.main:app --reload --port 8000
```

## License

[MIT](LICENSE)
