"""Tests for custom-vocabulary loading/merging and config resolution."""

import backend.vocabulary as vocab
from backend.config import Settings
from backend.utils import audio_stats
from backend.vocabulary import asr_bias_prompt, load_dictionary, merge_terms


def test_merge_terms_dedups_case_insensitively():
    assert merge_terms(["Anthropic", "groq"], ["GROQ", "Claude"]) == [
        "Anthropic",
        "groq",
        "Claude",
    ]


def test_merge_terms_strips_and_skips_blanks():
    assert merge_terms(["  Acme  ", ""], ["", "Foo"]) == ["Acme", "Foo"]


def test_asr_bias_prompt_empty():
    assert asr_bias_prompt([]) == ""


def test_asr_bias_prompt_lists_terms():
    prompt = asr_bias_prompt(["Kubernetes", "Grafana"])
    assert "Kubernetes" in prompt and "Grafana" in prompt


def test_load_dictionary_reads_file_and_ignores_comments(tmp_path, monkeypatch):
    f = tmp_path / "dict.txt"
    f.write_text("# a comment\nAnthropic\n\n  Groq  \n# another\n")
    monkeypatch.setattr(vocab.settings, "dictionary_path", str(f))
    assert load_dictionary() == ["Anthropic", "Groq"]


def test_load_dictionary_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(vocab.settings, "dictionary_path", str(tmp_path / "nope.txt"))
    assert load_dictionary() == []


def test_resolved_llm_model_ollama_uses_ollama_model():
    s = Settings(llm_provider="ollama", ollama_model="qwen2.5")
    assert s.resolved_llm_model == "qwen2.5"


def test_resolved_llm_model_override_wins():
    s = Settings(llm_provider="ollama", llm_model="custom-model")
    assert s.resolved_llm_model == "custom-model"


def test_audio_stats_rejects_non_wav():
    assert audio_stats(b"not a wav file at all") is None


def test_audio_stats_parses_wav():
    import io
    import wave

    import numpy as np

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        samples = (np.ones(16000, dtype=np.int16) * 1000)  # 1 second, constant tone
        wf.writeframes(samples.tobytes())

    stats = audio_stats(buf.getvalue())
    assert stats is not None
    assert abs(stats["duration"] - 1.0) < 0.01
    assert stats["rms"] > 0
