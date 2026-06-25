from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "extra": "ignore"}

    # --- Cloud API keys ---
    deepgram_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""

    # --- Provider selection ---
    # asr_provider: groq | deepgram | local | faster-whisper | mlx
    # llm_provider: groq | openai | anthropic | gemini | ollama
    asr_provider: str = "groq"
    llm_provider: str = "groq"
    llm_model: str = ""

    # --- Local ASR (offline Whisper) ---
    # faster-whisper model size: tiny[.en] base[.en] small[.en] medium[.en] large-v3 distil-large-v3 ...
    local_asr_model: str = "small.en"
    local_asr_device: str = "auto"          # auto | cpu | cuda
    local_asr_compute_type: str = "auto"    # auto | int8 | int8_float16 | float16 | float32
    local_asr_language: str = "en"          # ISO code, or "" for auto-detect
    # MLX (Apple Silicon) Whisper repo
    mlx_whisper_model: str = "mlx-community/whisper-large-v3-turbo"

    # --- Local LLM (Ollama / any OpenAI-compatible server) ---
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.2"

    # --- Custom vocabulary / dictionary ---
    # Newline-separated terms; biases ASR and corrects LLM output. Relative paths resolve to repo root.
    dictionary_path: str = "dictionary.txt"

    # --- Live streaming (WS /ws/transcribe mode=stream) ---
    # How often to re-transcribe the active (uncommitted) tail, and the minimum
    # new audio required before spending another ASR call on a partial.
    stream_interval_s: float = 1.2
    stream_min_new_s: float = 0.8
    # A pause of this many seconds finalizes ("commits") the current segment so
    # it's never re-transcribed again — keeps cost bounded on long dictation.
    stream_commit_silence_s: float = 0.6
    # RMS (0..1) below which audio is treated as silence and ASR is skipped
    # (Whisper hallucinates on silence). Lower it for a quiet mic; raise it if
    # background noise is being transcribed. Override with STREAM_SILENCE_RMS.
    stream_silence_rms: float = 0.006

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    @property
    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        defaults = {
            "groq": "llama-3.3-70b-versatile",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-4-5-20251001",
            "gemini": "gemini-2.0-flash",
            "ollama": self.ollama_model,
        }
        return defaults.get(self.llm_provider, "llama-3.3-70b-versatile")


settings = Settings()
