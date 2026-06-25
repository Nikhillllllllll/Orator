"""Launcher for `uv run dictate`: ensure a backend is running, then dictate.

The backend runs as a **persistent, detached** service: the launcher starts it
in its own session (so it survives the client quitting *and* the terminal
closing), reuses one that's already running, and never stops it on exit. Manage
it explicitly:

    uv run dictate            # ensure backend is up, then run the client
    uv run dictate --status   # is the backend running?
    uv run dictate --stop     # stop the backend

Honors HOST / PORT from the environment and sets WISPER_SERVER_URL so the client
targets the same address. WISPER_DEBUG=1 makes the backend log at debug level.
"""

import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOST = os.environ.get("DICTATE_HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
HEALTH_URL = f"http://{HOST}:{PORT}/health"
STARTUP_TIMEOUT_S = 30

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
PID_FILE = LOG_DIR / "backend.pid"


def _backend_healthy(timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def _start_backend_detached() -> subprocess.Popen:
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
        # New session: detach from the controlling terminal so the backend
        # outlives this launcher and survives the terminal window closing.
        start_new_session=True,
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


# ---- pid tracking ---------------------------------------------------------

def _write_pidfile(pid: int) -> None:
    try:
        LOG_DIR.mkdir(exist_ok=True)
        PID_FILE.write_text(str(pid))
    except OSError:
        pass


def _read_pidfile() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except (OSError, ValueError):
        return None


def _find_listener_pid() -> int | None:
    """Fallback for a server we didn't start (orphaned or launched by hand)."""
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{PORT}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [int(p) for p in out.stdout.split()]
        return pids[0] if pids else None
    except Exception:
        return None


def _backend_pid() -> int | None:
    return _read_pidfile() or _find_listener_pid()


# ---- subcommands ----------------------------------------------------------

def _cmd_status() -> int:
    if _backend_healthy():
        pid = _backend_pid()
        print(f"  ✅ Backend running on :{PORT}" + (f" (pid {pid})" if pid else ""))
    else:
        print(f"  ⚪ No backend running on :{PORT}")
    return 0


def _cmd_stop() -> int:
    pid = _backend_pid()
    if pid is None:
        print(f"  No backend found on :{PORT}.")
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"  🛑 Stopped backend (pid {pid}).")
    except ProcessLookupError:
        print(f"  Backend (pid {pid}) was not running.")
    except OSError as e:
        print(f"  Could not stop backend (pid {pid}): {e}")
        return 1
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass
    return 0


def _ensure_backend() -> bool:
    """Make sure a backend is up. Returns False if it couldn't be started."""
    if _backend_healthy():
        print(f"  ✅ Using backend already running on :{PORT}")
        if _read_pidfile() is None:  # adopt an existing/orphaned server
            found = _find_listener_pid()
            if found:
                _write_pidfile(found)
        return True

    print(f"  ⏳ Starting backend on :{PORT}…")
    proc = _start_backend_detached()
    if not _wait_until_healthy(proc):
        print("  ❌ Backend failed to start. Run it directly to see the error:")
        print(f"     uv run uvicorn backend.main:app --port {PORT}")
        if proc.poll() is None:
            proc.terminate()
        return False
    _write_pidfile(proc.pid)
    print(f"  ✅ Backend started (pid {proc.pid}) — stays running.")
    print("     Stop it with:  uv run dictate --stop")
    return True


def main() -> int:
    args = sys.argv[1:]
    if "--stop" in args:
        return _cmd_stop()
    if "--status" in args:
        return _cmd_status()

    os.environ.setdefault("WISPER_SERVER_URL", f"http://{HOST}:{PORT}")
    if not _ensure_backend():
        return 1

    # The backend is persistent — we do NOT stop it when the client exits.
    try:
        from client.dictate import main as run_client

        run_client()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
