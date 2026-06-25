"""One-command launcher: start the backend, then run the dictation client.

``uv run dictate`` boots the FastAPI backend in the background, waits for it to
become healthy, hands off to the hotkey client, and shuts the backend down on
exit. A backend already listening on the port is reused (and left running).

Honors ``HOST`` / ``PORT`` from the environment (or .env, via the backend) and
sets ``WISPER_SERVER_URL`` so the client targets the same address. Set
``WISPER_DEBUG=1`` to stream the backend's logs instead of discarding them.
"""

import os
import subprocess
import sys
import time
import urllib.request

HOST = os.environ.get("DICTATE_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
HEALTH_URL = f"http://{HOST}:{PORT}/health"
STARTUP_TIMEOUT_S = 30


def _backend_healthy(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _start_backend() -> subprocess.Popen:
    debug = bool(os.environ.get("WISPER_DEBUG"))
    sink = None if debug else subprocess.DEVNULL
    return subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "backend.main:app",
            "--host", HOST, "--port", str(PORT),
            "--log-level", "debug" if debug else "warning",
        ],
        stdout=sink,
        stderr=sink,
    )


def _wait_until_healthy(proc: subprocess.Popen) -> bool:
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        if _backend_healthy():
            return True
        if proc.poll() is not None:  # backend died during startup
            return False
        time.sleep(0.5)
    return False


def _stop_backend(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> int:
    # Keep the client pointed at whatever host/port we actually use.
    os.environ.setdefault("WISPER_SERVER_URL", f"http://{HOST}:{PORT}")

    proc: subprocess.Popen | None = None
    if _backend_healthy():
        print(f"  ✅ Using backend already running on :{PORT}")
    else:
        print(f"  ⏳ Starting backend on :{PORT}…")
        proc = _start_backend()
        if not _wait_until_healthy(proc):
            print("  ❌ Backend failed to start. Run it directly to see the error:")
            print(f"     uv run uvicorn backend.main:app --port {PORT}")
            if proc.poll() is None:
                _stop_backend(proc)
            return 1
        print(f"  ✅ Backend ready (pid {proc.pid})")

    try:
        from client.dictate import main as run_client

        run_client()
    except KeyboardInterrupt:
        pass
    finally:
        if proc is not None:
            _stop_backend(proc)
            print("\n  🛑 Backend stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
