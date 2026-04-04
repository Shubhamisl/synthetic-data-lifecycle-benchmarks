"""Regression tests for the Adult data loader entrypoint."""

from __future__ import annotations

import data.loader as loader


def test_loader_main_invokes_preprocessing(monkeypatch) -> None:
    """
    Ensure the module entrypoint calls the preprocessing pipeline.

    Inputs: pytest monkeypatch fixture.
    Outputs: Assertion that preprocessing was invoked exactly once.
    Lifecycle stage: Verification of Stage 1 entrypoint behavior.
    Reference: Regression test for module execution contract.
    """
    called = {"count": 0}

    def fake_load_and_preprocess():
        called["count"] += 1
        return None

    monkeypatch.setattr(loader, "load_and_preprocess", fake_load_and_preprocess)

    loader.main()

    assert called["count"] == 1
