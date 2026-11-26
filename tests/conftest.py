"""Shared test setup. No test may write to the real data\\logs folder or send a trace to LangSmith."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from monitor import store  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_monitor(tmp_path, monkeypatch):
    """Every test logs into its own empty folder and has tracing switched off."""
    store.set_client(store.FileClient(tmp_path / "monitor_logs"))
    store.reset_tally()
    monkeypatch.setattr(config, "LANGSMITH_API_KEY", None, raising=False)
    yield
    store.set_client(None)
