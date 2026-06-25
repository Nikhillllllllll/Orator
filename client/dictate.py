"""
Wisper macOS dictation client.

Press Ctrl+Shift to start recording, press again to stop.
Cleaned text is pasted at your cursor automatically.

Requires:
  - macOS Accessibility permission for this terminal app
  - Backend running at localhost:8000
"""

import json
import os
import queue
import subprocess
import sys
import threading

import httpx
import numpy as np
import sounddevice as sd
from pynput import keyboard
from websockets.sync.client import connect as ws_connect

SERVER_URL = "http://localhost:8000"
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1600  # 100 ms chunks at 16 kHz
HOTKEY = {keyboard.Key.ctrl_l, keyboard.Key.shift_l}

TOAST_SCRIPT = os.path.join(os.path.dirname(__file__), "toast.py")
TOAST_RECORDING = ("●  Recording…", "#c0392b")
TOAST_TRANSCRIBING = ("●  Transcribing…", "#b9770e")
TOAST_TYPING = ("●  Typing…", "#1e8449")
_toast_proc: subprocess.Popen | None = None


def show_toast(spec: tuple[str, str]) -> None:
    """Display a floating status HUD, replacing any current one."""
    global _toast_proc
    hide_toast()
    message, color = spec
    try:
        _toast_proc = subprocess.Popen(
            [sys.executable, TOAST_SCRIPT, message, color],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        _toast_proc = None


def hide_toast() -> None:
    global _toast_proc
    if _toast_proc and _toast_proc.poll() is None:
        try:
            _toast_proc.terminate()
        except Exception:
            pass
    _toast_proc = None

recording = False
stream: sd.InputStream | None = None
current_keys: set = set()
hotkey_active = False  # edge-trigger guard so key auto-repeat can't re-toggle

send_q: "queue.Queue[bytes | None]" = queue.Queue()


def get_active_app() -> str:
    try:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to get name of first application process whose frontmost is true'],
            capture_output=True, text=True, timeout=2,
        )
        name = result.stdout.strip().lower()
        app_map = {
            "slack": "slack",
            "mail": "email",
            "gmail": "email",
            "outlook": "email",
            "messages": "imessage",
            "discord": "discord",
            "code": "vscode",
            "cursor": "vscode",
            "notes": "notes",
            "notion": "notes",
            "google docs": "docs",
        }
        for key, val in app_map.items():
            if key in name:
                return val
    except Exception:
        pass
    return "default"


APP_ALIASES = {
    "chrome": "Google Chrome",
    "imessage": "Messages",
    "messages": "Messages",
    "vscode": "Visual Studio Code",
    "code": "Visual Studio Code",
    "iterm": "iTerm2",
    "terminal": "Terminal",
    "finder": "Finder",
}


def get_desktop_context() -> str:
    parts = []
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of every application process whose visible is true'],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append(f"Open apps: {result.stdout.strip()}")
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["osascript", "-e", '''
tell application "System Events"
    if not (exists process "Google Chrome") then return ""
end tell
tell application "Google Chrome"
    set tabInfo to {}
    set i to 1
    repeat with t in tabs of front window
        set end of tabInfo to (i as text) & ". " & title of t
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
end joinList'''],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append(f"Chrome tabs: {result.stdout.strip()}")
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["osascript", "-e", '''
tell application "System Events"
    if not (exists process "Safari") then return ""
end tell
tell application "Safari"
    set tabInfo to {}
    set i to 1
    repeat with t in tabs of front window
        set end of tabInfo to (i as text) & ". " & name of t
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
end joinList'''],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts.append(f"Safari tabs: {result.stdout.strip()}")
    except Exception:
        pass

    return "\n".join(parts)


def execute_command(data: dict):
    cmd = data.get("command", "")
    target = data.get("target")

    if cmd == "switch_app":
        app_name = APP_ALIASES.get(str(target).lower(), str(target))
        try:
            subprocess.run(
                ["osascript", "-e", f'tell application "{app_name}" to activate'],
                capture_output=True, timeout=3,
            )
            print(f"\r🔀 Switched to {app_name}")
        except Exception as e:
            print(f"\r❌ Couldn't switch to {app_name}: {e}")

    elif cmd == "switch_tab":
        tab_index = int(target) if target is not None else 1
        try:
            subprocess.run(
                ["osascript", "-e",
                 f'tell application "Google Chrome" to set active tab index of front window to {tab_index}'],
                capture_output=True, timeout=3,
            )
            print(f"\r🔀 Switched to tab {tab_index}")
        except Exception as e:
            print(f"\r❌ Couldn't switch tab: {e}")

    elif cmd == "open_url":
        url = str(target)
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        try:
            subprocess.run(["open", url], timeout=3)
            print(f"\r🌐 Opened {url}")
        except Exception as e:
            print(f"\r❌ Couldn't open URL: {e}")

    else:
        print(f"\r🔧 Command: {cmd}")


def audio_callback(indata: np.ndarray, frames: int, time_info, status):
    if recording:
        # Stream raw int16 PCM chunks to the sender thread as they're captured.
        send_q.put(bytes(indata))


def sender_loop(conn):
    """Drain captured PCM to the websocket, then signal end-of-utterance."""
    while True:
        chunk = send_q.get()
        if chunk is None:  # sentinel: recording stopped, flush done
            break
        try:
            conn.send(chunk)
        except Exception:
            return
    try:
        conn.send("END")
    except Exception:
        pass


def receiver_loop(conn, app_context: str):
    """Render live partials and act on the final result."""
    try:
        for message in conn:
            data = json.loads(message)

            if "error" in data:
                print(f"\r❌ {data['error']}                    ")
                return

            if data.get("stage") == "partial":
                text = data.get("text", "")
                if text:
                    clipped = text[:70] + ("…" if len(text) > 70 else "")
                    print(f"\r✏️  {clipped}", end="", flush=True)

            elif data.get("stage") == "final":
                if "command" in data:
                    execute_command(data)
                    return
                text = data.get("text", "")
                if not text:
                    print("\r⚠️  No speech detected.                       ")
                    return
                show_toast(TOAST_TYPING)
                pasted = paste_text(text)
                timing = data.get("timing", {})
                total = timing.get("total", "?")
                label = "Pasted" if pasted else "Copied"
                clipped = text[:60] + ("…" if len(text) > 60 else "")
                print(f"\r✅ {label} ({total}s) | {app_context} | \"{clipped}\"")
                return
    except Exception as e:
        print(f"\r❌ Stream error: {e}                          ")
    finally:
        hide_toast()
        try:
            conn.close()
        except Exception:
            pass


def start_recording():
    global recording, stream
    if recording:
        return
    app_context = get_active_app()
    screen_text = get_desktop_context()
    ws_url = SERVER_URL.replace("http://", "ws://").replace("https://", "wss://")

    try:
        conn = ws_connect(f"{ws_url}/ws/transcribe")
    except Exception:
        print("\r❌ Backend not running — start it with: uv run uvicorn backend.main:app --port 8000")
        return

    conn.send(json.dumps({
        "app_context": app_context,
        "screen_text": screen_text,
        "mode": "stream",
        "sample_rate": SAMPLE_RATE,
    }))

    # Drain any stale audio from a previous session.
    while not send_q.empty():
        send_q.get_nowait()

    recording = True
    threading.Thread(target=sender_loop, args=(conn,), daemon=True).start()
    threading.Thread(target=receiver_loop, args=(conn, app_context), daemon=True).start()

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=BLOCKSIZE,
        callback=audio_callback,
    )
    stream.start()
    show_toast(TOAST_RECORDING)
    print("\r🎙  Recording... (press hotkey again to stop)", end="", flush=True)


def stop_recording():
    global recording, stream
    if not recording:
        return
    recording = False
    if stream:
        stream.stop()
        stream.close()
        stream = None
    send_q.put(None)  # tell sender to flush remaining audio and send END
    show_toast(TOAST_TRANSCRIBING)
    print("\r⏳ Processing...                              ", end="", flush=True)


def paste_text(text: str) -> bool:
    process = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
    process.communicate(text.encode("utf-8"))
    result = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to keystroke "v" using command down'],
        capture_output=True, text=True, timeout=2,
    )
    if result.returncode != 0:
        print("\r⚠️  Auto-paste failed (grant Accessibility to your terminal app). Text copied to clipboard — Cmd+V to paste.")
        return False
    return True


def on_press(key):
    global hotkey_active
    current_keys.add(key)
    # Edge-trigger: fire once when the combo is first satisfied. Modifier keys
    # auto-repeat while held, so without this guard a single hold toggles
    # start/stop many times per second.
    if HOTKEY.issubset(current_keys) and not hotkey_active:
        hotkey_active = True
        if recording:
            stop_recording()
        else:
            start_recording()


def on_release(key):
    global hotkey_active
    current_keys.discard(key)
    if not HOTKEY.issubset(current_keys):
        hotkey_active = False


def main():
    print("=" * 50)
    print("  Wisper — Voice Dictation")
    print("=" * 50)
    print("  Hotkey:  Left Ctrl + Left Shift")
    print(f"  Server:  {SERVER_URL}")
    print()

    try:
        httpx.get(f"{SERVER_URL}/health", timeout=3)
        print("  ✅ Backend connected")
    except Exception:
        print("  ⚠️  Backend not reachable — start it first")

    check = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to keystroke ""'],
        capture_output=True, timeout=3,
    )
    if check.returncode != 0:
        print("  ⚠️  Accessibility not granted — auto-paste won't work")
        print("     Go to: System Settings → Privacy & Security → Accessibility")
        print("     and add your terminal app. Text will still be copied to clipboard.")
    else:
        print("  ✅ Accessibility granted")

    print()
    print("  Listening for hotkey... (Ctrl+C to quit)")
    print("-" * 50)

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    finally:
        hide_toast()


if __name__ == "__main__":
    main()
