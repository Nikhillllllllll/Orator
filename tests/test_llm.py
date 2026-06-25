"""Tests for LLM cleanup prompt — validates formatting across contexts."""

import pytest

from backend.llm import SYSTEM_PROMPT, USER_PROMPT, _strip_fences, is_command_response

# Prompt rendering tests

def test_system_prompt_renders_context():
    rendered = SYSTEM_PROMPT.format(
        app_context="slack",
        screen_text="some context",
        user_style="casual",
        dictionary="Anthropic, Groq",
        voice_hint="very quiet (likely whispered)",
    )
    assert "slack" in rendered
    assert "some context" in rendered
    assert "casual" in rendered
    assert "Anthropic, Groq" in rendered
    assert "whispered" in rendered


def test_user_prompt_renders_transcript():
    rendered = USER_PROMPT.format(transcript="hello world")
    assert "hello world" in rendered


# Fence stripping (local models sometimes wrap output in ``` fences)

def test_strip_fences_removes_code_block():
    assert _strip_fences('```\nHello world.\n```') == "Hello world."


def test_strip_fences_removes_language_block():
    assert _strip_fences('```json\n{"command": "delete"}\n```') == '{"command": "delete"}'


def test_strip_fences_leaves_plain_text():
    assert _strip_fences("Just text.") == "Just text."


def test_command_detection_with_fences():
    assert is_command_response('```json\n{"command": "undo"}\n```') == {"command": "undo"}


# Command detection tests

def test_command_detection_delete():
    assert is_command_response('{"command": "delete"}') == {"command": "delete"}


def test_command_detection_undo():
    assert is_command_response('{"command": "undo"}') == {"command": "undo"}


def test_command_detection_ignores_normal_text():
    assert is_command_response("The meeting is at 3 PM.") is None


def test_command_detection_ignores_partial_json():
    assert is_command_response('{"text": "hello"}') is None


def test_command_detection_ignores_bad_json():
    assert is_command_response("{bad json}") is None


# Test case catalog from the spec — used for integration tests against live LLM

CLEANUP_TEST_CASES = [
    {
        "raw": "hey um can you send me the uh the report by friday",
        "app_context": "slack",
        "should_contain": ["report", "friday"],
        "should_not_contain": ["um", "uh"],
    },
    {
        "raw": "Dear um Sarah I wanted to follow up on our conversation from last week uh regarding the Q3 budget",
        "app_context": "email",
        "should_contain": ["Dear Sarah", "Q3 budget"],
        "should_not_contain": ["um", "uh"],
    },
    {
        "raw": "the meeting is at 3 no wait 3:30 pm in conference room B",
        "app_context": "default",
        "should_contain": ["3:30", "conference room B"],
        "should_not_contain": ["no wait"],
    },
    {
        "raw": "first we need to update the database second we need to run migrations and third we need to deploy",
        "app_context": "notes",
        "should_contain": ["database", "migrations", "deploy"],
    },
    {
        "raw": "delete that",
        "app_context": "default",
        "expect_command": "delete",
    },
    {
        "raw": "I think we should like basically focus on three things revenue growth customer retention and like product quality you know",
        "app_context": "email",
        "should_contain": ["revenue growth", "customer retention", "product quality"],
        "should_not_contain": ["like basically", "you know"],
    },
    {
        "raw": "so basically I was thinking we could um maybe try to like refactor the authentication module because it's uh it's really messy right now",
        "app_context": "slack",
        "should_contain": ["refactor", "authentication"],
        "should_not_contain": ["basically", "um", "uh", "like"],
    },
    {
        "raw": "the function takes two parameters a string called name and an integer called age and it returns a boolean",
        "app_context": "vscode",
        "should_contain": ["string", "name", "integer", "age", "boolean"],
    },
    {
        "raw": "new paragraph",
        "app_context": "default",
        "expect_command": "new_paragraph",
    },
    {
        "raw": "hey what's up just wanted to check if you're free for lunch tomorrow around noon",
        "app_context": "imessage",
        "should_contain": ["lunch", "tomorrow", "noon"],
    },
]


@pytest.fixture
def test_cases():
    return CLEANUP_TEST_CASES
