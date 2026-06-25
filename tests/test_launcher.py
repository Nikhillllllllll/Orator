"""Tests for the `dictate` launcher's backend lifecycle decisions.

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
    return calls


def test_reuses_existing_backend(monkeypatch):
    calls = _stub_client(monkeypatch)
    monkeypatch.setattr(launcher, "_backend_healthy", lambda *a, **k: True)
    start = MagicMock()
    monkeypatch.setattr(launcher, "_start_backend", start)

    assert launcher.main() == 0
    assert calls == ["client_ran"]
    start.assert_not_called()  # an already-running backend is reused, not duplicated


def test_starts_then_stops_backend(monkeypatch):
    calls = _stub_client(monkeypatch)
    proc = MagicMock()
    stopped: list[object] = []
    monkeypatch.setattr(launcher, "_backend_healthy", lambda *a, **k: False)
    monkeypatch.setattr(launcher, "_start_backend", lambda: proc)
    monkeypatch.setattr(launcher, "_wait_until_healthy", lambda p: True)
    monkeypatch.setattr(launcher, "_stop_backend", lambda p: stopped.append(p))

    assert launcher.main() == 0
    assert calls == ["client_ran"]
    assert stopped == [proc]  # backend we started is torn down on exit


def test_startup_failure_returns_error(monkeypatch):
    calls = _stub_client(monkeypatch)
    proc = MagicMock()
    proc.poll.return_value = 1  # backend already exited
    monkeypatch.setattr(launcher, "_backend_healthy", lambda *a, **k: False)
    monkeypatch.setattr(launcher, "_start_backend", lambda: proc)
    monkeypatch.setattr(launcher, "_wait_until_healthy", lambda p: False)

    assert launcher.main() == 1
    assert calls == []  # client never starts if the backend won't come up
