"""
Wisper dictation client (macOS, Windows, Linux).

Press Ctrl+Shift to start recording, press again to stop.
Cleaned text is pasted at your cursor automatically.

Requires:
  - Backend running at localhost:8000
  - macOS: Accessibility permission for this terminal app
  - Linux: a clipboard helper (`wl-clipboard`, `xclip`, or `xsel`)

OS-specific integration (clipboard, paste, foreground-app and tab detection,
app switching) lives in ``platforms.py``; this module is platform-neutral.
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

try:  # works both as `python -m client.dictate` and `python client/dictate.py`
    from client.platforms import get_platform
except ImportError:
    from platforms import get_platform

PLATFORM = get_platform()

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


def execute_command(data: dict):
    cmd = data.get("command", "")
    target = data.get("target")

    if cmd == "switch_app":
        if PLATFORM.activate_app(str(target)):
            print(f"\r🔀 Switched to {target}")
        else:
            print(f"\r❌ Couldn't switch to {target}")

    elif cmd == "switch_tab":
        tab_index = int(target) if target is not None else 1
        if PLATFORM.switch_browser_tab(tab_index):
            print(f"\r🔀 Switched to tab {tab_index}")
        else:
            print("\r❌ Couldn't switch tab")

    elif cmd == "open_url":
        url = str(target)
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        if PLATFORM.open_url(url):
            print(f"\r🌐 Opened {url}")
        else:
            print(f"\r❌ Couldn't open URL: {url}")

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
    app_context = PLATFORM.active_app()
    screen_text = PLATFORM.desktop_context()
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
    if PLATFORM.copy_and_paste(text):
        return True
    print(
        f"\r⚠️  Auto-paste failed. Text copied to clipboard — {PLATFORM.paste_combo} to paste."
    )
    return False


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

    if PLATFORM.input_permission_ok():
        print("  ✅ Auto-paste ready")
    else:
        print("  ⚠️  Auto-paste unavailable")
        hint = PLATFORM.permission_hint()
        if hint:
            print(f"     {hint}")

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
