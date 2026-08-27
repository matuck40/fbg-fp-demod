"""Smoke tests for the Streamlit playground via streamlit's AppTest harness.

The app is an optional extra (``pip install .[app]``); importorskip keeps
the core suite green when streamlit is not installed. Beyond smoke, the
app-only math (scenario-to-spectrum wiring, derived band) is pinned by a
direct call to the pipeline: the synthesized crest must use the fringe
order the tracker actually follows, at every cavity size.
"""

import importlib.util
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "main.py"


def _load_app_module():
    spec = importlib.util.spec_from_file_location("app_main", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_app_metric_shows_a_sane_pressure_error():
    at = _run_app()
    rms_bar = float(at.metric[0].value.split()[0])
    assert rms_bar < 0.1  # the README's declared envelope


def test_pipeline_uses_the_tracked_fringe_order():
    # 89 um is one of the cavity sizes where rounding the fringe order the
    # wrong way builds a ~2% truth-vs-recovery divergence into the
    # synthesis itself. Noise-free, the correctly wired pipeline recovers
    # pressure to well under 0.02 bar RMS here.
    app = _load_app_module()
    opd_nm = 89_000.0
    result = app.run_pipeline(
        p_max=1.5,
        t_max=2.0,
        n_frames=40,
        noise_db=0.0,
        drift_db=0.0,
        opd_nm=opd_nm,
        trim=0.1,
        band=app.derived_band(opd_nm),
    )
    import numpy as np

    rms = np.sqrt(np.mean((result["pressure"] - result["pressure_true"]) ** 2))
    assert rms < 0.02
