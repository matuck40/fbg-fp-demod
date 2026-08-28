"""Tests for the Streamlit pipeline viewer.

importorskip keeps the core suite green without streamlit. Beyond smoke,
the synthetic wiring is pinned by a direct call: the synthesized crest
must use the fringe order the tracker actually follows, at every cavity
size; and the file mode is exercised against a synthetic export written
in the instrument's own format.
"""

import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from fbgfp import synth, track  # noqa: E402
from test_io import write_responses  # noqa: E402

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "main.py"


def _run_app():
    return st_testing.AppTest.from_file(str(APP_PATH)).run(timeout=120)


def _load_app_module():
    spec = importlib.util.spec_from_file_location("app_main", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_app_runs_with_defaults_and_shows_metrics():
    at = _run_app()
    assert not at.exception
    # Spectra count, reading at frame, fringe hops.
    assert len(at.metric) >= 3
    assert at.metric[0].value == "60"


def test_app_reruns_cleanly_when_the_frame_changes():
    at = _run_app()
    frame = [s for s in at.main.slider if s.label.lower() == "frame"]
    assert frame, "expected a frame slider in the main area"
    frame[0].set_value(30)
    at.run(timeout=120)
    assert not at.exception
    assert len(at.metric) >= 3


def test_file_mode_without_a_path_shows_guidance():
    at = _run_app()
    at.sidebar.radio[0].set_value("Interrogator file")
    at.run(timeout=120)
    assert not at.exception
    assert at.info, "expected guidance to enter a file path"


def test_file_mode_reads_a_synthetic_export(tmp_path):
    n_frames, n_points = 4, 8192
    wl = synth.wavelength_axis(n_points=n_points)
    blocks = np.stack(
        [
            np.stack([synth.fp_spectrum(wl, 87_000.0 + 15.0 * i)] * 4)
            for i in range(n_frames)
        ]
    )
    stamps = [datetime(2026, 1, 5) + timedelta(seconds=20 * i) for i in range(n_frames)]
    path = tmp_path / "Responses.synth.txt"
    write_responses(path, stamps, blocks)

    at = _run_app()
    at.sidebar.radio[0].set_value("Interrogator file")
    at.run(timeout=120)
    at.sidebar.text_input[0].set_value(str(path))
    reference = [n for n in at.sidebar.number_input if "reference" in n.label.lower()]
    reference[0].set_value(1470.0)
    at.run(timeout=120)

    assert not at.exception
    assert at.metric[0].value == str(n_frames)


def test_synthetic_sequence_uses_the_tracked_fringe_order():
    # 89 um is a cavity size where rounding the fringe order the wrong way
    # builds a ~2% truth-vs-recovery divergence into the synthesis itself.
    app = _load_app_module()
    opd_nm = 89_000.0
    n_frames = 30
    wl, spectra_db, _ = app.synthetic_sequence(
        p_max=1.5, t_max=0.0, n_frames=n_frames, noise_db=0.0,
        drift_db=0.0, opd_nm=opd_nm,
    )
    band = app.derived_band(opd_nm, wl)
    result = track.track_fp(spectra_db, wl, band, app.SYNTH_REFERENCE_NM)

    frames = np.arange(n_frames)
    pressure = 1.5 * (1.0 - np.exp(-frames / 15.0))
    intended_shift = -1.5648 * pressure
    recovered = result.corrected_nm - result.corrected_nm[0]
    errors = np.abs(recovered - intended_shift)
    assert np.all(errors <= 0.06 * np.abs(intended_shift) + 0.002)
