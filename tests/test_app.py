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


def test_file_mode_opens_on_the_shipped_sample():
    """A fresh clone must show real data without anyone typing a path.

    The sample lives at a path resolved from the app file, not from the
    working directory, so it is found wherever streamlit was launched
    from — and the band and reference default to the ones that cavity
    needs, which are not the synthetic scenario's.
    """
    at = _run_app()
    at.sidebar.radio[0].set_value("Interrogator file")
    at.run(timeout=120)
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    assert at.metric[0].value == "40", "expected the 40 shipped spectra"


def test_file_mode_with_the_path_cleared_shows_guidance():
    at = _run_app()
    at.sidebar.radio[0].set_value("Interrogator file")
    at.run(timeout=120)
    at.sidebar.text_input[0].set_value("")
    at.run(timeout=120)
    assert not at.exception
    assert at.info, "expected guidance once the path is emptied"


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


def _line_labelled(fig, fragment):
    """The plotted line whose legend label contains ``fragment``."""
    for axis in fig.axes:
        for line in axis.get_lines():
            label = line.get_label()
            if fragment.lower() in label.lower():
                return line
    raise AssertionError(
        f"no line labelled {fragment!r}; found "
        f"{[l.get_label() for a in fig.axes for l in a.get_lines()]}"
    )


def test_the_reading_shown_on_the_spectrum_is_plotted_on_the_trajectory():
    """The number panel 3 prints must be findable on panel 4's y axis.

    The original MATLAB plotted ``total_wave_data(b) = pico_x`` — the same
    absolute nm it printed as "Actual reading" over the filtered spectrum,
    so the two panels read as one measurement. Plotting a picometre shift
    relative to frame 0 instead breaks that link: different quantity,
    different unit, different origin.
    """
    app = _load_app_module()
    opd_nm, trim, n_frames = 87_000.0, 0.1, 60
    wl, spectra_db, _ = app.synthetic_sequence(
        p_max=1.5, t_max=2.0, n_frames=n_frames, noise_db=0.2,
        drift_db=0.3, opd_nm=opd_nm,
    )
    band = app.derived_band(opd_nm, wl)
    result = track.track_fp(spectra_db, wl, band, app.SYNTH_REFERENCE_NM, trim=trim)

    for frame in (0, 17, 30, 59):
        _, reading_nm = app.pipeline_figure(
            wl, spectra_db[frame], band, result.valley_nm[frame], trim,
            frame=frame, n_frames=n_frames,
        )
        trajectory = app.trajectory_figure(
            np.arange(n_frames), result.crest_nm, result.corrected_nm,
            result.hop_fringes, frame, "Frame",
        )
        plotted = _line_labelled(trajectory, "reading")
        assert plotted.get_ydata()[frame] == pytest.approx(reading_nm, abs=1e-9), (
            f"frame {frame}: panel 3 prints {reading_nm:.5f} nm but panel 4 "
            f"plots {plotted.get_ydata()[frame]:.5f} at that frame"
        )


def test_the_hop_panel_reports_the_direction_the_tracked_fringe_went():
    """The hop branch of the trajectory panel, which no other test reaches.

    The counters are signed by where the tracked fringe moved along the
    wavelength axis. Passing the hop *frames* in place of the signed
    fringe counts flips the reported direction, and every earlier test
    ran on a scenario with no hops at all, so nothing caught it.
    """
    app = _load_app_module()
    n_frames, opd_nm = 40, 87_000.0
    wl = synth.wavelength_axis()
    sequence = synth.simulate_sequence(
        wl,
        opd_nm=opd_nm * (1.0 + np.linspace(0.0, 0.06, n_frames)),
        fbg_centers_nm=np.column_stack(
            [np.full(n_frames, 1525.4), np.full(n_frames, 1554.8)]
        ),
        noise_db=0.2, drift_amplitude_db=0.3, seed=0,
    )
    band = app.derived_band(opd_nm, wl)
    result = track.track_fp(
        sequence.spectra_db, wl, band, app.SYNTH_REFERENCE_NM, trim=0.1
    )
    assert result.hop_frames, "this scenario must hop for the test to mean anything"

    # Here the tracker is handed down the axis at every hop; the counters
    # must say so rather than reporting the opposite direction.
    for i in result.hop_frames:
        assert result.valley_nm[i] < result.valley_nm[i - 1]
    assert all(n < 0 for n in result.hop_fringes)

    figure = app.trajectory_figure(
        np.arange(n_frames), result.crest_nm, result.corrected_nm,
        result.hop_fringes, result.hop_frames[0], "Frame",
    )
    title = figure.axes[0].get_title()
    assert f"{len(result.hop_fringes)} down the axis" in title, title
    assert "0 up the axis" in title, title

    # Once a hop has fired the unwrapped series is drawn, and it is the
    # one without the one-fringe step.
    unwrapped = _line_labelled(figure, "unwrapped")
    assert np.abs(np.diff(unwrapped.get_ydata())).max() < 0.5 * np.abs(
        np.diff(result.crest_nm)
    ).max()
