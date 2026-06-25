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
import logging
import os
import queue
import subprocess
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

import httpx
import numpy as np
import sounddevice as sd
from pynput import keyboard
from websockets.sync.client import connect as ws_connect

try:  # works both as `python -m client.dictate` and `python client/dictate.py`
    from client.diagnostics import describe_no_speech
    from client.platforms import get_platform
except ImportError:
    from diagnostics import describe_no_speech
    from platforms import get_platform

PLATFORM = get_platform()

# Quiet by default so the CLI stays clean; set WISPER_DEBUG=1 to surface the
# otherwise-swallowed errors (dropped sockets, failed paste, toast spawn, …).
logger = logging.getLogger("wisper.client")

SERVER_URL = os.environ.get("WISPER_SERVER_URL", "http://localhost:8000")
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1600  # 100 ms chunks at 16 kHz
HOTKEY = {keyboard.Key.ctrl_l, keyboard.Key.shift_l}

TOAST_SCRIPT = os.path.join(os.path.dirname(__file__), "toast.py")
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
CLIENT_PID_FILE = LOG_DIR / "client.pid"     # single-instance guard
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
        logger.debug("Could not spawn toast HUD", exc_info=True)
        _toast_proc = None


def hide_toast() -> None:
    global _toast_proc
    if _toast_proc and _toast_proc.poll() is None:
        try:
            _toast_proc.terminate()
        except Exception:
            logger.debug("Could not terminate toast HUD", exc_info=True)
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
            logger.debug("Send failed mid-stream; stopping sender", exc_info=True)
            return
    try:
        conn.send("END")
    except Exception:
        logger.debug("Could not send END marker", exc_info=True)


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
                    logger.warning(
                        "no speech detected (app=%s rms=%s threshold=%s)",
                        app_context, data.get("audio_rms"), data.get("silence_threshold"),
                    )
                    msg = describe_no_speech(
                        data.get("audio_rms"), data.get("silence_threshold")
                    )
                    print(f"\r⚠️  {msg}")
                    return
                show_toast(TOAST_TYPING)
                pasted = paste_text(text)
                timing = data.get("timing", {})
                total = timing.get("total", "?")
                label = "Pasted" if pasted else "Copied"
                logger.info(
                    "final (app=%s, %ss, %s): %r", app_context, total, label.lower(), text
                )
                clipped = text[:60] + ("…" if len(text) > 60 else "")
                print(f"\r✅ {label} ({total}s) | {app_context} | \"{clipped}\"")
                return
    except Exception as e:
        logger.debug("Receiver loop error", exc_info=True)
        print(f"\r❌ Stream error: {e}                          ")
    finally:
        hide_toast()
        try:
            conn.close()
        except Exception:
            logger.debug("Could not close websocket", exc_info=True)


def _abort_recording() -> None:
    """Tear down a half-started recording (e.g. the socket connect failed)."""
    global recording, stream
    recording = False
    if stream:
        stream.stop()
        stream.close()
        stream = None
    while not send_q.empty():
        send_q.get_nowait()
    hide_toast()


def start_recording():
    global recording, stream
    if recording:
        return

    # Start capturing IMMEDIATELY — before the (sometimes slow) window-context
    # osascript calls and the socket connect — so multi-second AppleScript
    # timeouts can't clip the start of what you say. Audio buffers in send_q
    # until the sender thread starts and flushes it.
    while not send_q.empty():
        send_q.get_nowait()
    recording = True
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

    app_context = PLATFORM.active_app()
    screen_text = PLATFORM.desktop_context()
    ws_url = SERVER_URL.replace("http://", "ws://").replace("https://", "wss://")
    try:
        conn = ws_connect(f"{ws_url}/ws/transcribe")
    except Exception:
        logger.debug("WebSocket connect failed", exc_info=True)
        print("\r❌ Backend not running — start it with: uv run dictate")
        _abort_recording()
        return

    conn.send(json.dumps({
        "app_context": app_context,
        "screen_text": screen_text,
        "mode": "stream",
        "sample_rate": SAMPLE_RATE,
    }))
    logger.info("recording started (app=%s)", app_context)
    threading.Thread(target=sender_loop, args=(conn,), daemon=True).start()
    threading.Thread(target=receiver_loop, args=(conn, app_context), daemon=True).start()


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


def toggle_recording() -> None:
    """Start or stop recording (the hotkey entry point)."""
    if recording:
        stop_recording()
    else:
        start_recording()


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
        toggle_recording()


def on_release(key):
    global hotkey_active
    current_keys.discard(key)
    if not HOTKEY.issubset(current_keys):
        hotkey_active = False


def _running_client_pid() -> int | None:
    """Pid of an already-running client, or None (treats a dead pid as stale)."""
    try:
        pid = int(CLIENT_PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None
    if pid == os.getpid():
        return None
    try:
        os.kill(pid, 0)  # signal 0 = existence check, doesn't actually signal
    except ProcessLookupError:
        return None  # ESRCH: process is gone; pidfile is stale
    except PermissionError:
        return pid  # EPERM: exists but owned by another user — still running
    except OSError:
        return None
    return pid


def _configure_logging() -> None:
    """Always log to a rotating file so dropped phrases stay reviewable; mirror
    to the console only when WISPER_DEBUG is set."""
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    try:
        LOG_DIR.mkdir(exist_ok=True)
        fileh = RotatingFileHandler(
            LOG_DIR / "client.log", maxBytes=1_000_000, backupCount=3
        )
        fileh.setFormatter(fmt)
        logger.addHandler(fileh)
    except OSError:
        pass  # logging to file is best-effort; never block dictation on it

    if os.environ.get("WISPER_DEBUG"):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        logger.addHandler(console)


def main():
    _configure_logging()

    # Single-instance guard: a second client would add another hotkey listener
    # and double-type everything you say.
    existing = _running_client_pid()
    if existing:
        print(f"  ⚠️  A dictation client is already running (pid {existing}).")
        print("     Quit it first (Ctrl+C in its terminal) — two clients double-type.")
        return
    try:
        CLIENT_PID_FILE.write_text(str(os.getpid()))
    except OSError:
        pass

    print("=" * 50)
    print("  Wisper — Voice Dictation")
    print("=" * 50)
    print("  Hotkey:  Left Ctrl + Left Shift")
    print(f"  Server:  {SERVER_URL}")
    print(f"  Logs:    {LOG_DIR / 'client.log'}")
    print()

    try:
        httpx.get(f"{SERVER_URL}/health", timeout=3)
        print("  ✅ Backend connected")
    except Exception:
        logger.debug("Health check failed", exc_info=True)
        print("  ⚠️  Backend not reachable — start it first")

    if PLATFORM.input_permission_ok():
        print("  ✅ Auto-paste ready")
    else:
        print("  ⚠️  Auto-paste unavailable")
        hint = PLATFORM.permission_hint()
        if hint:
            print(f"     {hint}")

    print()
    print("  Listening for hotkey… (Ctrl+C to quit)")
    print("-" * 50)

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        pass
    finally:
        hide_toast()
        try:
            CLIENT_PID_FILE.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    main()
