from pydantic import BaseModel, Field


class TranscribeMetadata(BaseModel):
    app_context: str = "default"
    screen_text: str = ""
    user_style: str = "default"
    # Per-request custom vocabulary, merged with the server-side dictionary.
    dictionary: list[str] = Field(default_factory=list)
    # ISO language code to force ASR/cleanup; None = provider default / auto-detect.
    language: str | None = None


class TranscribeResponse(BaseModel):
    text: str
    raw_transcript: str
    app_context: str
    timing: dict[str, float]
    confidence: float | None = None
    language: str | None = None
    voice_hint: str = ""


class RawTranscribeResponse(BaseModel):
    text: str
    timing: dict[str, float]
    confidence: float | None = None
    language: str | None = None
    voice_hint: str = ""


class CommandResponse(BaseModel):
    command: str
    target: str | int | None = None
