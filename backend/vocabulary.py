"""Custom vocabulary / dictionary.

Terms here serve two purposes:
1. Bias the ASR so they are transcribed correctly (Whisper ``initial_prompt`` /
   Groq ``prompt`` / Deepgram ``keywords``).
2. Tell the LLM how to spell names, jargon, and acronyms during cleanup.
"""

import logging
from pathlib import Path

from backend.config import settings

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path() -> Path:
    path = Path(settings.dictionary_path)
    return path if path.is_absolute() else _REPO_ROOT / path


def load_dictionary() -> list[str]:
    """Read the dictionary file. Blank lines and ``#`` comments are ignored."""
    path = _resolve_path()
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        logger.warning("Could not read dictionary %s: %s", path, e)
        return []
    return [s for line in lines if (s := line.strip()) and not s.startswith("#")]


def merge_terms(*term_lists: list[str]) -> list[str]:
    """Combine term lists, dropping case-insensitive duplicates, preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for terms in term_lists:
        for t in terms:
            t = t.strip()
            key = t.lower()
            if t and key not in seen:
                seen.add(key)
                out.append(t)
    return out


def asr_bias_prompt(terms: list[str]) -> str:
    """Whisper-style biasing prompt — listing terms nudges decoding toward them."""
    if not terms:
        return ""
    return "Glossary of terms that may appear: " + ", ".join(terms) + "."
