"""Tests for the `dictate` launcher: backend lifecycle + stop/status commands.

A stub ``client.dictate`` is injected so these never import the real client
(which pulls in sounddevice/pynput and needs a desktop session).
"""

import sys
import types
from unittest.mock import MagicMock

import client.launcher as launcher


def _stub_client(monkeypatch) -> list[str]:
    """Replace client.dictate with a stub; return a list it appends to when run."""
    calls: list[str] = []
    fake = types.ModuleType("client.dictate")
    fake.main = lambda: calls.append("client_ran")
    monkeypatch.setitem(sys.modules, "client.dictate", fake)
    monkeypatch.setenv("WISPER_SERVER_URL", "http://test")  # keep setdefault a no-op
    monkeypatch.setattr(sys, "argv", ["dictate"])
    return calls


def test_reuses_existing_backend(monkeypatch):
    calls = _stub_client(monkeypatch)
    monkeypatch.setattr(launcher, "_backend_healthy", lambda *a, **k: True)
    monkeypatch.setattr(launcher, "_read_pidfile", lambda: 999)  # already tracked
    start = MagicMock()
    monkeypatch.setattr(launcher, "_start_backend_detached", start)

    assert launcher.main() == 0
    assert calls == ["client_ran"]
    start.assert_not_called()  # a running backend is reused, not duplicated


def test_adopts_orphaned_backend_by_writing_pidfile(monkeypatch):
    _stub_client(monkeypatch)
    monkeypatch.setattr(launcher, "_backend_healthy", lambda *a, **k: True)
    monkeypatch.setattr(launcher, "_read_pidfile", lambda: None)  # no pidfile yet
    monkeypatch.setattr(launcher, "_find_listener_pid", lambda: 4242)
    written: list[int] = []
    monkeypatch.setattr(launcher, "_write_pidfile", lambda pid: written.append(pid))

    assert launcher.main() == 0
    assert written == [4242]  # orphaned server gets adopted so --stop can find it


def test_starts_detached_and_leaves_running(monkeypatch):
    calls = _stub_client(monkeypatch)
    proc = MagicMock()
    proc.pid = 555
    stopped: list[object] = []
    monkeypatch.setattr(launcher, "_backend_healthy", lambda *a, **k: False)
    monkeypatch.setattr(launcher, "_start_backend_detached", lambda: proc)
    monkeypatch.setattr(launcher, "_wait_until_healthy", lambda p: True)
    monkeypatch.setattr(launcher, "_write_pidfile", lambda pid: None)
    # Persistent model: the launcher must NOT terminate the backend on exit.
    monkeypatch.setattr(proc, "terminate", lambda: stopped.append("terminated"))

    assert launcher.main() == 0
    assert calls == ["client_ran"]
    assert stopped == []  # backend is left running


def test_startup_failure_returns_error(monkeypatch):
    calls = _stub_client(monkeypatch)
    proc = MagicMock()
    proc.poll.return_value = 1  # backend already exited
    monkeypatch.setattr(launcher, "_backend_healthy", lambda *a, **k: False)
    monkeypatch.setattr(launcher, "_start_backend_detached", lambda: proc)
    monkeypatch.setattr(launcher, "_wait_until_healthy", lambda p: False)

    assert launcher.main() == 1
    assert calls == []  # client never starts if the backend won't come up


def test_stop_kills_tracked_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["dictate", "--stop"])
    pidfile = tmp_path / "backend.pid"
    pidfile.write_text("777")
    monkeypatch.setattr(launcher, "PID_FILE", pidfile)
    monkeypatch.setattr(launcher, "_find_listener_pid", lambda: None)
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(launcher.os, "kill", lambda pid, sig: killed.append((pid, sig)))

    assert launcher.main() == 0
    assert killed == [(777, launcher.signal.SIGTERM)]
    assert not pidfile.exists()  # stale pidfile is removed


def test_stop_with_no_server(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dictate", "--stop"])
    monkeypatch.setattr(launcher, "_read_pidfile", lambda: None)
    monkeypatch.setattr(launcher, "_find_listener_pid", lambda: None)

    assert launcher.main() == 0  # nothing to stop is not an error


def test_status_reports_running(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["dictate", "--status"])
    monkeypatch.setattr(launcher, "_backend_healthy", lambda *a, **k: True)
    monkeypatch.setattr(launcher, "_read_pidfile", lambda: 321)
    monkeypatch.setattr(launcher, "_find_listener_pid", lambda: None)

    assert launcher.main() == 0
    assert "running" in capsys.readouterr().out
