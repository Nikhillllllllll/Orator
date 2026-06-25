"""HTTP + WebSocket endpoint tests using FastAPI's TestClient.

The pipeline (ASR/LLM) is mocked at the ``backend.main`` boundary so these
exercise request parsing, routing, the WebSocket protocol, and error handling
without touching real providers or the network.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.asr import ASRResult
from backend.config import settings
from backend.main import app
from backend.models import RawTranscribeResponse, TranscribeResponse

client = TestClient(app)

WAV_BYTES = b"RIFF....WAVEfmt " + b"\x00" * 64  # opaque; the pipeline is mocked
PCM_ONE_SECOND = (b"\x10\x27" * 16000)  # 16k int16 samples, clearly non-silent


@pytest.fixture
def temp_dictionary(tmp_path, monkeypatch):
    """Point the dictionary at a throwaway file so writes don't touch the repo."""
    dict_file = tmp_path / "dictionary.txt"
    dict_file.write_text("Acme\nKubernetes\n", encoding="utf-8")
    monkeypatch.setattr(settings, "dictionary_path", str(dict_file))
    return dict_file


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_config_reports_providers():
    resp = client.get("/api/config")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("asr_provider", "llm_provider", "llm_model", "dictionary_terms"):
        assert key in body
    assert isinstance(body["dictionary_terms"], int)


def test_get_dictionary(temp_dictionary):
    resp = client.get("/api/dictionary")
    assert resp.status_code == 200
    assert resp.json()["terms"] == ["Acme", "Kubernetes"]


def test_post_dictionary_appends_new_terms(temp_dictionary):
    resp = client.post("/api/dictionary", json={"terms": ["Postgres", "Acme"]})
    assert resp.status_code == 200
    body = resp.json()
    # "Acme" already exists (case-insensitive) → only "Postgres" is added.
    assert body["added"] == ["Postgres"]
    assert "Postgres" in body["terms"]
    assert temp_dictionary.read_text().count("Acme") == 1


def test_transcribe_rejects_empty_audio():
    resp = client.post(
        "/api/transcribe",
        files={"audio": ("clip.wav", b"", "audio/wav")},
        data={"metadata": "{}"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_transcribe_returns_cleaned_result():
    fake = TranscribeResponse(
        text="Hello world.",
        raw_transcript="um hello world",
        app_context="default",
        timing={"asr": 0.1, "llm": 0.2, "total": 0.3},
    )
    with patch("backend.main.transcribe_and_clean", AsyncMock(return_value=fake)) as mock:
        resp = client.post(
            "/api/transcribe",
            files={"audio": ("clip.wav", WAV_BYTES, "audio/wav")},
            data={"metadata": json.dumps({"app_context": "slack"})},
        )
    assert resp.status_code == 200
    assert resp.json()["text"] == "Hello world."
    # Metadata was parsed and forwarded to the pipeline.
    assert mock.call_args.args[1].app_context == "slack"


def test_transcribe_malformed_metadata_falls_back_to_defaults():
    fake = TranscribeResponse(
        text="hi", raw_transcript="hi", app_context="default", timing={}
    )
    with patch("backend.main.transcribe_and_clean", AsyncMock(return_value=fake)) as mock:
        resp = client.post(
            "/api/transcribe",
            files={"audio": ("clip.wav", WAV_BYTES, "audio/wav")},
            data={"metadata": "not-json{{"},
        )
    assert resp.status_code == 200
    assert mock.call_args.args[1].app_context == "default"


def test_transcribe_pipeline_error_returns_500():
    with patch("backend.main.transcribe_and_clean", AsyncMock(side_effect=RuntimeError("boom"))):
        resp = client.post(
            "/api/transcribe",
            files={"audio": ("clip.wav", WAV_BYTES, "audio/wav")},
            data={"metadata": "{}"},
        )
    assert resp.status_code == 500
    assert "boom" in resp.json()["error"]


def test_websocket_buffered_partial_then_final():
    raw = RawTranscribeResponse(
        text="um hello world",
        timing={"asr": 0.1},
        confidence=0.9,
        language="en",
        voice_hint="",
    )
    with (
        patch("backend.main.transcribe_raw", AsyncMock(return_value=raw)),
        patch("backend.main.cleanup_transcript", AsyncMock(return_value=("Hello world.", {"llm": 0.2}))),
    ):
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_text(json.dumps({"app_context": "default"}))  # buffered (no mode)
            ws.send_bytes(WAV_BYTES)
            ws.send_text("END")

            partial = ws.receive_json()
            assert partial["stage"] == "partial"
            assert partial["text"] == "um hello world"

            final = ws.receive_json()
            assert final["stage"] == "final"
            assert final["text"] == "Hello world."
            assert final["raw_transcript"] == "um hello world"


def test_websocket_streaming_emits_final():
    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = ASRResult(text="hello world", confidence=0.9, language="en")

    with (
        patch("backend.main.get_asr", return_value=mock_asr),
        patch("backend.main.cleanup_transcript", AsyncMock(return_value=("Hello world.", {"llm": 0.2}))),
    ):
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_text(json.dumps({"mode": "stream", "sample_rate": 16000}))
            ws.send_bytes(PCM_ONE_SECOND)
            ws.send_text("END")

            # Partials may or may not arrive depending on tick timing; drain until final.
            for _ in range(20):
                msg = ws.receive_json()
                if msg.get("stage") == "final":
                    assert msg["text"] == "Hello world."
                    break
            else:
                pytest.fail("never received a final message")


def test_websocket_streaming_empty_reports_audio_level():
    # ASR returns nothing despite real (non-silent) audio → the final must carry
    # the measured level + threshold so the client can explain why.
    mock_asr = AsyncMock()
    mock_asr.transcribe.return_value = ASRResult(text="", confidence=None, language=None)

    with patch("backend.main.get_asr", return_value=mock_asr):
        with client.websocket_connect("/ws/transcribe") as ws:
            ws.send_text(json.dumps({"mode": "stream", "sample_rate": 16000}))
            ws.send_bytes(PCM_ONE_SECOND)
            ws.send_text("END")

            for _ in range(20):
                msg = ws.receive_json()
                if msg.get("stage") == "final":
                    assert msg["text"] == ""
                    assert isinstance(msg["audio_rms"], float) and msg["audio_rms"] > 0
                    assert msg["silence_threshold"] == 0.01
                    break
            else:
                pytest.fail("never received a final message")


def test_websocket_no_audio_reports_error():
    with client.websocket_connect("/ws/transcribe") as ws:
        ws.send_text(json.dumps({"app_context": "default"}))
        ws.send_text("END")
        msg = ws.receive_json()
        assert "error" in msg
