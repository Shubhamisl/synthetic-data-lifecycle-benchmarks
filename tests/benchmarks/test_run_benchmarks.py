from __future__ import annotations

from benchmarks.run_benchmarks import _configure_stdout_utf8


def test_configure_stdout_utf8_reconfigures_when_supported(monkeypatch):
    calls: list[dict[str, str]] = []

    class DummyStdout:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("sys.stdout", DummyStdout())

    _configure_stdout_utf8()

    assert calls == [{"encoding": "utf-8"}]
