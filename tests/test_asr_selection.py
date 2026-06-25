"""ASR provider selection & fallback — the logic behind 'works with zero keys'.

The provider classes are patched with sentinels so no real client is built;
we only assert which backend ``get_asr_provider`` picks for a given config.
"""

import pytest

import backend.asr as asr
from backend.asr import get_asr_provider


@pytest.fixture
def fake_providers(monkeypatch):
    """Replace each backend with a sentinel marker instead of a real client."""
    monkeypatch.setattr(asr, "GroqWhisperASR", lambda: "groq")
    monkeypatch.setattr(asr, "DeepgramASR", lambda: "deepgram")
    monkeypatch.setattr(asr, "_make_local_asr", lambda pref: f"local:{pref}")
    return monkeypatch


def configure(monkeypatch, *, provider, groq_key="", deepgram_key=""):
    monkeypatch.setattr(asr.settings, "asr_provider", provider)
    monkeypatch.setattr(asr.settings, "groq_api_key", groq_key)
    monkeypatch.setattr(asr.settings, "deepgram_api_key", deepgram_key)


@pytest.mark.parametrize("provider", ["local", "whisper", "faster-whisper", "mlx"])
def test_explicit_local_providers(fake_providers, provider):
    configure(fake_providers, provider=provider)
    assert get_asr_provider() == f"local:{provider}"


def test_groq_selected_when_key_present(fake_providers):
    configure(fake_providers, provider="groq", groq_key="gsk_test")
    assert get_asr_provider() == "groq"


def test_deepgram_selected_when_key_present(fake_providers):
    configure(fake_providers, provider="deepgram", deepgram_key="dg_test")
    assert get_asr_provider() == "deepgram"


def test_groq_without_key_falls_back_to_any_cloud_key(fake_providers):
    # Asked for Groq but no Groq key; a Deepgram key is available → use it.
    configure(fake_providers, provider="groq", deepgram_key="dg_test")
    assert get_asr_provider() == "deepgram"


def test_no_keys_falls_back_to_local(fake_providers):
    configure(fake_providers, provider="groq")
    assert get_asr_provider() == "local:local"


def test_unknown_provider_with_groq_key_uses_groq(fake_providers):
    configure(fake_providers, provider="something-else", groq_key="gsk_test")
    assert get_asr_provider() == "groq"


def test_unknown_provider_no_keys_uses_local(fake_providers):
    configure(fake_providers, provider="something-else")
    assert get_asr_provider() == "local:local"
