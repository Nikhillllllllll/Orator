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
OVERLAY_SCRIPT = os.path.join(os.path.dirname(__file__), "overlay.py")
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
# Tiny file channel between the client and the (subprocess) floating button.
OVERLAY_CMD = LOG_DIR / "overlay.cmd"        # button -> client ("toggle")
OVERLAY_STATUS = LOG_DIR / "overlay.status"  # client -> button (status name)
TOAST_RECORDING = ("●  Recording…", "#c0392b")
TOAST_TRANSCRIBING = ("●  Transcribing…", "#b9770e")
TOAST_TYPING = ("●  Typing…", "#1e8449")
_toast_proc: subprocess.Popen | None = None

# Recording state, shared with the floating overlay (which polls current_status).
# When the overlay is active it shows status itself, so the toast HUD is muted.
_overlay_active = False
_status = "idle"  # one of: idle, recording, processing


def _set_status(name: str) -> None:
    global _status
    _status = name


def current_status() -> str:
    return _status


def show_toast(spec: tuple[str, str]) -> None:
    """Display a floating status HUD, replacing any current one."""
    global _toast_proc
    if _overlay_active:
        return  # the overlay renders status itself
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
                _set_status("processing")
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
        _set_status("idle")
        hide_toast()
        try:
            conn.close()
        except Exception:
            logger.debug("Could not close websocket", exc_info=True)


def _abort_recording() -> None:
    """Tear down a half-started recording (e.g. the socket connect failed)."""
    global recording, stream
    recording = False
    _set_status("idle")
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
    _set_status("recording")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=BLOCKSIZE,
        callback=audio_callback,
    )
    stream.start()
    show_toast(TOAST_RECORDING)
    print("\r🎙  Recording... (click/hotkey again to stop)", end="", flush=True)

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
    _set_status("processing")
    if stream:
        stream.stop()
        stream.close()
        stream = None
    send_q.put(None)  # tell sender to flush remaining audio and send END
    show_toast(TOAST_TRANSCRIBING)
    print("\r⏳ Processing...                              ", end="", flush=True)


def toggle_recording() -> None:
    """Start or stop recording — the entry point for both the hotkey and the
    floating button."""
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


def _spawn_button() -> subprocess.Popen | None:
    """Launch the floating button in its own process (isolated from the client
    so a tkinter failure can't kill dictation)."""
    try:
        OVERLAY_CMD.unlink(missing_ok=True)  # clear any stale command
        OVERLAY_STATUS.write_text(current_status())
    except OSError:
        pass
    try:
        return subprocess.Popen(
            [sys.executable, OVERLAY_SCRIPT, str(OVERLAY_CMD), str(OVERLAY_STATUS)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        logger.debug("Could not spawn floating button", exc_info=True)
        return None


def _overlay_ipc_loop(stop: threading.Event) -> None:
    """Bridge the button process: publish status, act on its toggle clicks."""
    while not stop.is_set():
        try:
            OVERLAY_STATUS.write_text(current_status())
            if OVERLAY_CMD.exists():
                cmd = OVERLAY_CMD.read_text().strip()
                OVERLAY_CMD.unlink(missing_ok=True)
                if cmd == "toggle":
                    toggle_recording()
        except OSError:
            logger.debug("Overlay IPC error", exc_info=True)
        stop.wait(0.1)


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
    overlay_enabled = not os.environ.get("WISPER_NO_OVERLAY")

    print("=" * 50)
    print("  Wisper — Voice Dictation")
    print("=" * 50)
    if overlay_enabled:
        print("  Record:  click the floating ● button, or Left Ctrl + Left Shift")
    else:
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
    print("  Ready. Ctrl+C to quit.")
    print("-" * 50)

    global _overlay_active
    # The hotkey listener and the floating button (a separate process) both run
    # in the background; the main thread blocks on the listener. tkinter never
    # runs in this process, so a GUI hiccup can't take dictation down.
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    button_proc = None
    ipc_stop = threading.Event()
    if overlay_enabled:
        _overlay_active = True  # mute the toast HUD; the button shows status
        button_proc = _spawn_button()
        threading.Thread(
            target=_overlay_ipc_loop, args=(ipc_stop,), daemon=True
        ).start()

    try:
        listener.join()
    except KeyboardInterrupt:
        pass
    finally:
        ipc_stop.set()
        if button_proc and button_proc.poll() is None:
            button_proc.terminate()
        listener.stop()
        hide_toast()


if __name__ == "__main__":
    main()
