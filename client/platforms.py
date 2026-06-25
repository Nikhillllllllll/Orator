"""Cross-platform OS integration for the Wisper dictation client.

The client needs five things from the host OS: read the foreground app, gather
optional desktop context (e.g. browser tabs), put text on the clipboard,
synthesize a paste keystroke, and activate/launch apps. Each backend implements
what it can for its platform; anything unsupported degrades to a safe default
(returns ``""`` / ``"default"`` / ``False``) instead of raising. That keeps the
core record → transcribe → paste loop working everywhere, with the niceties
(tab context, app switching) gated to the platforms that can do them.

``pynput`` is imported lazily inside :meth:`Platform.paste` so this module can
be imported on a headless box (CI, a server) where the keyboard backend has no
display to attach to.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import webbrowser

logger = logging.getLogger("wisper.client")

# Foreground app/window name (matched lowercase, substring) → dictation context.
# Order matters: more specific keys first so they win over broader ones.
_CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("slack", "slack"),
    ("gmail", "email"),
    ("outlook", "email"),
    ("mail", "email"),
    ("messages", "imessage"),
    ("discord", "discord"),
    ("cursor", "vscode"),
    ("code", "vscode"),
    ("notion", "notes"),
    ("notes", "notes"),
    ("google docs", "docs"),
    ("docs", "docs"),
]

# Friendly target → OS-level app name, for the "switch to X" command on macOS.
_MAC_APP_ALIASES = {
    "chrome": "Google Chrome",
    "imessage": "Messages",
    "messages": "Messages",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
    "iterm": "iTerm2",
    "terminal": "Terminal",
    "finder": "Finder",
}


def categorize_app(name: str) -> str:
    """Map a raw app/window name to a dictation context category."""
    name = (name or "").lower()
    for keyword, category in _CATEGORY_KEYWORDS:
        if keyword in name:
            return category
    return "default"


def _run(cmd: list[str], timeout: float = 3.0) -> subprocess.CompletedProcess | None:
    """Run a command, returning the completed process or ``None`` on any failure."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        logger.debug("Command failed: %s", cmd[0], exc_info=True)
        return None


class Platform:
    """Default no-op backend; a safe fallback for unrecognized systems."""

    name = "generic"
    paste_combo = "Ctrl+V"
    _paste_modifier = "ctrl"  # pynput Key attribute name; "cmd" on macOS

    # --- overridden per platform ------------------------------------------
    def active_app(self) -> str:
        return "default"

    def desktop_context(self) -> str:
        return ""

    def set_clipboard(self, text: str) -> bool:
        return False

    def activate_app(self, app: str) -> bool:
        return False

    def switch_browser_tab(self, index: int) -> bool:
        return False

    def input_permission_ok(self) -> bool:
        return True

    def permission_hint(self) -> str:
        return ""

    # --- shared across platforms ------------------------------------------
    def open_url(self, url: str) -> bool:
        try:
            return webbrowser.open(url)
        except Exception:
            logger.debug("Could not open URL: %s", url, exc_info=True)
            return False

    def paste(self) -> bool:
        """Synthesize the paste hotkey (clipboard contents are assumed set)."""
        try:
            from pynput.keyboard import Controller, Key

            kb = Controller()
            modifier = getattr(Key, self._paste_modifier)
            with kb.pressed(modifier):
                kb.press("v")
                kb.release("v")
            return True
        except Exception:
            logger.debug("Synthetic paste failed", exc_info=True)
            return False

    def copy_and_paste(self, text: str) -> bool:
        """Copy ``text`` and paste it at the cursor.

        Returns ``True`` if the text was both copied and pasted, ``False`` if it
        was at best copied (the caller should tell the user to paste manually).
        """
        if not self.set_clipboard(text):
            return False
        return self.paste()


class MacPlatform(Platform):
    name = "darwin"
    paste_combo = "Cmd+V"
    _paste_modifier = "cmd"

    def active_app(self) -> str:
        res = _run([
            "osascript", "-e",
            'tell application "System Events" to get name of first '
            "application process whose frontmost is true",
        ])
        if res and res.returncode == 0:
            return categorize_app(res.stdout.strip())
        return "default"

    def desktop_context(self) -> str:
        parts: list[str] = []

        apps = _run([
            "osascript", "-e",
            'tell application "System Events" to get name of every '
            "application process whose visible is true",
        ])
        if apps and apps.returncode == 0 and apps.stdout.strip():
            parts.append(f"Open apps: {apps.stdout.strip()}")

        # Chrome exposes a tab's text as `title`, Safari as `name`.
        for app, prop, label in (
            ("Google Chrome", "title", "Chrome tabs"),
            ("Safari", "name", "Safari tabs"),
        ):
            tabs = _run(["osascript", "-e", _BROWSER_TABS_SCRIPT.format(app=app, prop=prop)])
            if tabs and tabs.returncode == 0 and tabs.stdout.strip():
                parts.append(f"{label}: {tabs.stdout.strip()}")

        return "\n".join(parts)

    def set_clipboard(self, text: str) -> bool:
        try:
            proc = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"))
            return proc.returncode == 0
        except Exception:
            logger.debug("pbcopy failed", exc_info=True)
            return False

    def activate_app(self, app: str) -> bool:
        name = _MAC_APP_ALIASES.get(app.lower(), app)
        res = _run(["osascript", "-e", f'tell application "{name}" to activate'])
        return bool(res and res.returncode == 0)

    def switch_browser_tab(self, index: int) -> bool:
        res = _run([
            "osascript", "-e",
            f'tell application "Google Chrome" to set active tab index '
            f"of front window to {index}",
        ])
        return bool(res and res.returncode == 0)

    def input_permission_ok(self) -> bool:
        res = _run([
            "osascript", "-e",
            'tell application "System Events" to keystroke ""',
        ])
        return bool(res and res.returncode == 0)

    def permission_hint(self) -> str:
        return (
            "Grant Accessibility: System Settings → Privacy & Security → "
            "Accessibility → add your terminal app. Text will still be copied."
        )


class WindowsPlatform(Platform):
    name = "windows"
    paste_combo = "Ctrl+V"
    _paste_modifier = "ctrl"

    def active_app(self) -> str:
        try:
            import ctypes

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return categorize_app(buf.value)
        except Exception:
            logger.debug("GetForegroundWindow failed", exc_info=True)
            return "default"

    def set_clipboard(self, text: str) -> bool:
        try:
            # PowerShell's Set-Clipboard handles Unicode reliably (unlike `clip`,
            # which is bound to the console code page).
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "$input | Set-Clipboard"],
                input=text.encode("utf-8"),
                timeout=5,
            )
            return proc.returncode == 0
        except Exception:
            logger.debug("Set-Clipboard failed", exc_info=True)
            return False


class LinuxPlatform(Platform):
    name = "linux"
    paste_combo = "Ctrl+V"
    _paste_modifier = "ctrl"

    def __init__(self) -> None:
        self._clip_cmd = self._detect_clipboard_tool()

    @staticmethod
    def _detect_clipboard_tool() -> list[str] | None:
        if shutil.which("wl-copy"):
            return ["wl-copy"]  # Wayland
        if shutil.which("xclip"):
            return ["xclip", "-selection", "clipboard"]
        if shutil.which("xsel"):
            return ["xsel", "--clipboard", "--input"]
        return None

    def active_app(self) -> str:
        if shutil.which("xdotool"):
            res = _run(["xdotool", "getactivewindow", "getwindowname"])
            if res and res.returncode == 0:
                return categorize_app(res.stdout.strip())
        return "default"

    def set_clipboard(self, text: str) -> bool:
        if not self._clip_cmd:
            return False
        try:
            proc = subprocess.run(self._clip_cmd, input=text.encode("utf-8"), timeout=5)
            return proc.returncode == 0
        except Exception:
            logger.debug("Clipboard tool failed: %s", self._clip_cmd[0], exc_info=True)
            return False

    def input_permission_ok(self) -> bool:
        return self._clip_cmd is not None

    def permission_hint(self) -> str:
        if self._clip_cmd is None:
            return (
                "No clipboard tool found — install `wl-clipboard` (Wayland) or "
                "`xclip`/`xsel` (X11) for auto-paste."
            )
        return ""


_BROWSER_TABS_SCRIPT = """
tell application "System Events"
    if not (exists process "{app}") then return ""
end tell
tell application "{app}"
    set tabInfo to {{}}
    set i to 1
    repeat with t in tabs of front window
        set end of tabInfo to (i as text) & ". " & {prop} of t
        set i to i + 1
    end repeat
    return my joinList(tabInfo, " | ")
end tell
on joinList(theList, delim)
    set oldDelims to AppleScript's text item delimiters
    set AppleScript's text item delimiters to delim
    set result to theList as text
    set AppleScript's text item delimiters to oldDelims
    return result
end joinList"""


def get_platform() -> Platform:
    """Return the backend for the current OS (no-op fallback for unknowns)."""
    if sys.platform == "darwin":
        return MacPlatform()
    if sys.platform.startswith("win"):
        return WindowsPlatform()
    if sys.platform.startswith("linux"):
        return LinuxPlatform()
    return Platform()
