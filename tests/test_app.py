"""Smoke tests for the Streamlit playground via streamlit's AppTest harness.

The app is an optional extra (``pip install .[app]``); importorskip keeps
the core suite green when streamlit is not installed. These tests only
assert the app runs end to end and reacts to a control — the numerical
behaviour of the pipeline is covered by the core test suite.
"""

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "main.py"


def _run_app():
    return st_testing.AppTest.from_file(str(APP_PATH)).run(timeout=120)


def test_app_runs_with_defaults_and_shows_metrics():
    at = _run_app()
    assert not at.exception
    # RMS pressure error, RMS temperature error, fringe hop count.
    assert len(at.metric) >= 3


def test_app_reruns_cleanly_when_noise_is_changed():
    at = _run_app()
    noise = [s for s in at.sidebar.slider if "noise" in s.label.lower()]
    assert noise, "expected a noise slider in the sidebar"
    noise[0].set_value(0.0)
    at.run(timeout=120)
    assert not at.exception
    assert len(at.metric) >= 1
