import json
import logging
from abc import ABC, abstractmethod

from backend.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the text-formatting engine inside a professional voice dictation app (think Wispr Flow). You convert raw speech-to-text into clean, ready-to-insert text that reads exactly like something the user typed by hand. Your output is inserted directly at the user's cursor, so it must be final: no preamble, no quotes, no markdown fences, no commentary.

# CONTEXT
- Active app: {app_context}
- Screen content / open tabs: {screen_text}
- User style preference: {user_style}
- Voice characteristics: {voice_hint}
- Custom vocabulary (spell these exactly; fix close mis-hearings of them): {dictionary}

# CORE RULES
1. DISFLUENCY: Remove fillers (um, uh, er, hmm, like, you know, I mean, sort of, kind of, basically, literally) when they add nothing. Keep one only if it carries real meaning.
2. SELF-CORRECTION: When the speaker restates or corrects themselves ("3, no wait, 3:30", "send it to Bob, actually Alice"), keep ONLY the final intended version and drop the correction scaffolding ("no wait", "actually", "scratch that", "I mean", "sorry").
3. FIDELITY: Never add facts, names, numbers, opinions, or details that were not spoken. Never change the intended meaning. When uncertain, stay closer to the literal transcript.
4. VOCABULARY: Fix obvious mis-hearings of the custom-vocabulary terms and well-known proper nouns / technical terms. Reproduce spoken numbers, units, code identifiers, and casing exactly.
5. SPOKEN PUNCTUATION & FORMATTING — convert explicit verbal markup:
   "period"/"full stop"->.  "comma"->,  "question mark"->?  "exclamation mark"/"exclamation point"->!  "colon"->:  "semicolon"->;
   "new line"->line break, "new paragraph"->blank line, "open quote"/"close quote"->" "
   "bullet point"/"dash"->"- " list item, "numbered list"/"first second third"->1. 2. 3.
   Do NOT convert these words when they are clearly part of the sentence's meaning.
6. STRUCTURE: Detect implicit structure. "first… second… third…" or "one… two…" becomes a numbered list; a run of parallel short items becomes a bullet list.
7. GRAMMAR: Fix grammar, capitalization, and punctuation so it reads as careful writing — without altering the speaker's voice or meaning.
8. VOICE-AWARENESS: Let the voice characteristics gently inform tone. "very quiet / whispered" -> keep it short and discreet; "loud / emphatic" -> preserve the emphasis; "fast pace" -> the speaker may be brainstorming, so tighten rambling into crisp text.

# APP FORMATTING PROFILES
- Chat (slack, discord, imessage, whatsapp, teams, telegram): concise and conversational; sentence case; no greeting or sign-off; emoji only if spoken.
- Email (gmail, outlook, mail, email): professional; greeting and sign-off when appropriate; real paragraphs; no slang.
- Code editors (vscode, cursor, vim, neovim, intellij, sublime): format as a comment or docstring; preserve identifiers, symbols, and casing exactly; never prose-ify code.
- Docs / notes (notion, docs, notes, obsidian, word): clean prose with paragraph breaks and lists where structure exists.
- Terminal (terminal, iterm): if it sounds like a shell command, output the likely command; otherwise plain text.
- default: clean, natural prose.

# COMMAND MODE
If the ENTIRE utterance is a control command (not dictation), output ONLY a compact one-line JSON object and nothing else:
- Editing: "delete that"->{{"command": "delete"}} ; "undo"->{{"command": "undo"}} ; "new paragraph"->{{"command": "new_paragraph"}} ; "select all"->{{"command": "select_all"}}
- App switch: "switch to Slack"->{{"command": "switch_app", "target": "Slack"}}
- Tab switch: "go to the Gmail tab"->{{"command": "switch_tab", "target": <1-based index from Screen content; fuzzy-match the tab title>}}
- Open URL: "open github.com"->{{"command": "open_url", "target": "https://github.com"}}
A sentence that merely mentions these words is dictation, not a command.

# OUTPUT
Output ONLY the final text, or the single command JSON. No quotes, no markdown fences, no notes.

# EXAMPLES
Input (slack): "hey um can you uh send me the the report by friday"
Output: Hey, can you send me the report by Friday?

Input (email): "Dear um Sarah I wanted to follow up on our conversation from last week uh regarding the Q3 budget"
Output: Dear Sarah,

I wanted to follow up on our conversation from last week regarding the Q3 budget.

Input (default): "the meeting is at 3 no wait 3:30 pm in conference room B"
Output: The meeting is at 3:30 PM in conference room B.

Input (notes): "first we need to update the database second run migrations and third deploy"
Output: 1. Update the database
2. Run migrations
3. Deploy

Input (default): "delete that"
Output: {{"command": "delete"}}"""

USER_PROMPT = 'Raw transcript:\n"""\n{transcript}\n"""'


def _strip_fences(text: str) -> str:
    """Remove a wrapping markdown code fence some (esp. local) models add."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def _format_dictionary(terms: list[str] | None) -> str:
    return ", ".join(terms) if terms else "(none)"


class LLMProvider(ABC):
    @abstractmethod
    async def cleanup(
        self,
        transcript: str,
        app_context: str,
        screen_text: str,
        user_style: str,
        dictionary: list[str] | None = None,
        voice_hint: str = "",
    ) -> str: ...

    def _build_messages(
        self,
        transcript: str,
        app_context: str,
        screen_text: str,
        user_style: str,
        dictionary: list[str] | None = None,
        voice_hint: str = "",
    ) -> tuple[str, str]:
        system = SYSTEM_PROMPT.format(
            app_context=app_context,
            screen_text=screen_text or "(not available)",
            user_style=user_style,
            dictionary=_format_dictionary(dictionary),
            voice_hint=voice_hint or "(normal)",
        )
        user = USER_PROMPT.format(transcript=transcript)
        return system, user


class _ChatCompletionsLLM(LLMProvider):
    """Shared implementation for OpenAI-compatible chat APIs (OpenAI, Groq, Ollama, …)."""

    client = None  # set by subclass
    model = ""

    async def cleanup(
        self,
        transcript: str,
        app_context: str,
        screen_text: str,
        user_style: str,
        dictionary: list[str] | None = None,
        voice_hint: str = "",
    ) -> str:
        system, user = self._build_messages(
            transcript, app_context, screen_text, user_style, dictionary, voice_hint
        )
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        return _strip_fences(response.choices[0].message.content)


class GroqLLM(_ChatCompletionsLLM):
    def __init__(self) -> None:
        from groq import AsyncGroq

        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.resolved_llm_model


class OpenAILLM(_ChatCompletionsLLM):
    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.resolved_llm_model


class OllamaLLM(_ChatCompletionsLLM):
    """Local LLM via Ollama (or any OpenAI-compatible server) at OLLAMA_BASE_URL."""

    def __init__(self) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=settings.ollama_base_url, api_key="ollama")
        self.model = settings.ollama_model


class AnthropicLLM(LLMProvider):
    def __init__(self) -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.model = settings.resolved_llm_model

    async def cleanup(
        self,
        transcript: str,
        app_context: str,
        screen_text: str,
        user_style: str,
        dictionary: list[str] | None = None,
        voice_hint: str = "",
    ) -> str:
        system, user = self._build_messages(
            transcript, app_context, screen_text, user_style, dictionary, voice_hint
        )
        response = await self.client.messages.create(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=2048,
        )
        return _strip_fences(response.content[0].text)


class GeminiLLM(LLMProvider):
    def __init__(self) -> None:
        from google import genai

        self.client = genai.Client(api_key=settings.google_api_key)
        self.model = settings.resolved_llm_model

    async def cleanup(
        self,
        transcript: str,
        app_context: str,
        screen_text: str,
        user_style: str,
        dictionary: list[str] | None = None,
        voice_hint: str = "",
    ) -> str:
        system, user = self._build_messages(
            transcript, app_context, screen_text, user_style, dictionary, voice_hint
        )
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=user,
            config={"system_instruction": system, "temperature": 0.2, "max_output_tokens": 2048},
        )
        return _strip_fences(response.text)


def get_llm_provider() -> LLMProvider:
    provider = settings.llm_provider.lower()

    if provider == "ollama":
        logger.info("Using Ollama LLM (%s)", settings.ollama_model)
        return OllamaLLM()

    providers: dict[str, tuple[str, type[LLMProvider]]] = {
        "groq": ("groq_api_key", GroqLLM),
        "openai": ("openai_api_key", OpenAILLM),
        "anthropic": ("anthropic_api_key", AnthropicLLM),
        "gemini": ("google_api_key", GeminiLLM),
    }
    if provider in providers:
        key_attr, cls = providers[provider]
        if getattr(settings, key_attr):
            logger.info("Using %s LLM", provider)
            return cls()
    for name, (key_attr, cls) in providers.items():
        if getattr(settings, key_attr):
            logger.info("Falling back to %s LLM", name)
            return cls()

    # No cloud key configured — run fully offline via Ollama.
    logger.info("No cloud LLM key configured — falling back to Ollama (%s)", settings.ollama_model)
    return OllamaLLM()


def is_command_response(text: str) -> dict | None:
    text = _strip_fences(text)
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            if "command" in data:
                return data
        except json.JSONDecodeError:
            pass
    return None
