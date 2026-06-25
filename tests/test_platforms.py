"""Tests for the client's cross-platform OS-integration layer."""

import client.platforms as platforms
from client.platforms import (
    LinuxPlatform,
    MacPlatform,
    Platform,
    WindowsPlatform,
    categorize_app,
    get_platform,
)


class TestCategorizeApp:
    def test_known_apps_map_to_categories(self):
        assert categorize_app("Slack") == "slack"
        assert categorize_app("Discord") == "discord"
        assert categorize_app("Messages") == "imessage"

    def test_email_aliases(self):
        assert categorize_app("Gmail") == "email"
        assert categorize_app("Microsoft Outlook") == "email"
        assert categorize_app("Mail") == "email"

    def test_editors_map_to_vscode(self):
        assert categorize_app("Visual Studio Code") == "vscode"
        assert categorize_app("Cursor") == "vscode"

    def test_more_specific_keyword_wins(self):
        # "google docs" must beat the broader "docs".
        assert categorize_app("Google Docs — Untitled") == "docs"

    def test_matching_is_case_insensitive(self):
        assert categorize_app("SLACK") == "slack"

    def test_unknown_and_empty_default(self):
        assert categorize_app("Some Random App") == "default"
        assert categorize_app("") == "default"
        assert categorize_app(None) == "default"


class TestGetPlatform:
    def test_selects_backend_by_sys_platform(self, monkeypatch):
        cases = {
            "darwin": MacPlatform,
            "win32": WindowsPlatform,
            "linux": LinuxPlatform,
            "linux2": LinuxPlatform,
        }
        for value, expected in cases.items():
            monkeypatch.setattr(platforms.sys, "platform", value)
            # Linux backend probes for a clipboard tool at construction.
            monkeypatch.setattr(platforms.shutil, "which", lambda _name: None)
            assert isinstance(get_platform(), expected)

    def test_unknown_platform_falls_back_to_noop(self, monkeypatch):
        monkeypatch.setattr(platforms.sys, "platform", "freebsd")
        plat = get_platform()
        assert type(plat) is Platform
        assert plat.active_app() == "default"
        assert plat.desktop_context() == ""
        assert plat.set_clipboard("x") is False


class TestCopyAndPaste:
    def test_copy_failure_short_circuits_paste(self, monkeypatch):
        plat = Platform()
        monkeypatch.setattr(plat, "set_clipboard", lambda _text: False)
        # paste() must not be reached if the clipboard could not be set.
        monkeypatch.setattr(
            plat, "paste", lambda: (_ for _ in ()).throw(AssertionError("paste called"))
        )
        assert plat.copy_and_paste("hello") is False

    def test_copy_then_paste_succeeds(self, monkeypatch):
        plat = Platform()
        calls = []
        monkeypatch.setattr(plat, "set_clipboard", lambda text: calls.append(text) or True)
        monkeypatch.setattr(plat, "paste", lambda: True)
        assert plat.copy_and_paste("hello") is True
        assert calls == ["hello"]


class TestMacPlatform:
    def test_active_app_categorizes_osascript_output(self, monkeypatch):
        class FakeProc:
            returncode = 0
            stdout = "Slack\n"

        monkeypatch.setattr(platforms, "_run", lambda *a, **k: FakeProc())
        assert MacPlatform().active_app() == "slack"

    def test_active_app_defaults_when_osascript_fails(self, monkeypatch):
        monkeypatch.setattr(platforms, "_run", lambda *a, **k: None)
        assert MacPlatform().active_app() == "default"


class TestLinuxPlatform:
    def test_no_clipboard_tool_reports_unavailable(self, monkeypatch):
        monkeypatch.setattr(platforms.shutil, "which", lambda _name: None)
        plat = LinuxPlatform()
        assert plat.input_permission_ok() is False
        assert plat.set_clipboard("x") is False
        assert "clipboard" in plat.permission_hint().lower()

    def test_prefers_wayland_then_xclip(self, monkeypatch):
        monkeypatch.setattr(platforms.shutil, "which", lambda name: name == "wl-copy")
        assert LinuxPlatform()._clip_cmd == ["wl-copy"]

        monkeypatch.setattr(platforms.shutil, "which", lambda name: name == "xclip")
        assert LinuxPlatform()._clip_cmd[0] == "xclip"
