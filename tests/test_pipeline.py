"""Pipeline tests — unit tests with mocked providers, integration test marker for live API."""

import pytest
from unittest.mock import AsyncMock, patch

from backend.asr import ASRResult
from backend.models import TranscribeMetadata
from backend.pipeline import transcribe_and_clean, transcribe_raw


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset cached providers between tests."""
    import backend.pipeline as p
    p._asr = None
    p._llm = None
    yield
    p._asr = None
    p._llm = None


@pytest.mark.asyncio
async def test_transcribe_raw_returns_transcript():
    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = ASRResult(text="hello world", confidence=0.9, language="en")

    with patch("backend.pipeline.get_asr", return_value=mock_asr):
        result = await transcribe_raw(b"fake audio data")

    assert result.text == "hello world"
    assert result.confidence == 0.9
    assert result.language == "en"
    assert "asr" in result.timing


@pytest.mark.asyncio
async def test_transcribe_and_clean_returns_cleaned():
    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = ASRResult(text="um hello uh world")

    mock_llm = AsyncMock()
    mock_llm.cleanup.return_value = "Hello world."

    with (
        patch("backend.pipeline.get_asr", return_value=mock_asr),
        patch("backend.pipeline.get_llm", return_value=mock_llm),
    ):
        result = await transcribe_and_clean(
            b"fake audio data",
            TranscribeMetadata(app_context="default"),
        )

    assert result.text == "Hello world."
    assert result.raw_transcript == "um hello uh world"
    assert "asr" in result.timing
    assert "llm" in result.timing


@pytest.mark.asyncio
async def test_transcribe_empty_transcript():
    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = ASRResult(text="  ")

    with patch("backend.pipeline.get_asr", return_value=mock_asr):
        result = await transcribe_and_clean(
            b"fake audio",
            TranscribeMetadata(),
        )

    assert result.text == ""


@pytest.mark.asyncio
async def test_transcribe_command_response():
    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = ASRResult(text="delete that")

    mock_llm = AsyncMock()
    mock_llm.cleanup.return_value = '{"command": "delete"}'

    with (
        patch("backend.pipeline.get_asr", return_value=mock_asr),
        patch("backend.pipeline.get_llm", return_value=mock_llm),
    ):
        result = await transcribe_and_clean(
            b"fake audio",
            TranscribeMetadata(),
        )

    assert result.command == "delete"


@pytest.mark.asyncio
async def test_llm_receives_correct_context():
    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = ASRResult(text="test transcript")

    mock_llm = AsyncMock()
    mock_llm.cleanup.return_value = "cleaned"

    meta = TranscribeMetadata(
        app_context="slack",
        screen_text="channel: #general",
        user_style="casual",
        dictionary=["Acme"],
    )

    with (
        patch("backend.pipeline.get_asr", return_value=mock_asr),
        patch("backend.pipeline.get_llm", return_value=mock_llm),
    ):
        await transcribe_and_clean(b"audio", meta)

    mock_llm.cleanup.assert_called_once()
    kwargs = mock_llm.cleanup.call_args.kwargs
    assert kwargs["transcript"] == "test transcript"
    assert kwargs["app_context"] == "slack"
    assert kwargs["screen_text"] == "channel: #general"
    assert kwargs["user_style"] == "casual"
    assert "Acme" in kwargs["dictionary"]
    assert "voice_hint" in kwargs
