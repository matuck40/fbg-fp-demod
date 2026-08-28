"""End-to-end test of the processing script on synthetic export files.

Mirrors the original MATLAB pipeline's optical outputs: FPI demodulated
from the spectra, FBGs taken from the Peaks stream (Savitzky-Golay
filtered), aligned on the spectra's timestamps.
"""

import csv
import importlib.util
import pathlib
from datetime import datetime, timedelta

import numpy as np

from fbgfp import synth

from test_io import write_peaks, write_responses


def _load_script():
    path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "demodulate.py"
    spec = importlib.util.spec_from_file_location("demodulate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_writes_the_optical_time_series(tmp_path):
    n_frames, n_points = 5, 8192
    wl = synth.wavelength_axis(n_points=n_points)
    opd = 87_000.0 + 15.0 * np.arange(n_frames)
    rng = np.random.default_rng(3)
    blocks = np.stack(
        [
            np.stack(
                [
                    rng.normal(-30.0, 0.1, n_points),
                    synth.fp_spectrum(wl, opd[i]),
                    synth.fbg_spectrum(wl, [1470.5, 1480.2]),
                    synth.fbg_spectrum(wl, [1466.0, 1476.0, 1486.0]),
                ]
            )
            for i in range(n_frames)
        ]
    )
    t0 = datetime(2026, 1, 5, 12, 0, 0)
    spectra_stamps = [t0 + timedelta(seconds=20 * i) for i in range(n_frames)]
    write_responses(tmp_path / "Responses.synth.txt", spectra_stamps, blocks)

    # Peaks stream at 2 s cadence covering the same window, with a known
    # drift on the first FBG so the alignment is observable.
    n_peaks = 60
    peak_stamps = [t0 + timedelta(seconds=2 * i) for i in range(n_peaks)]
    peak_rows = np.column_stack(
        [
            np.full(n_peaks, 1525.4) + 1e-4 * np.arange(n_peaks),  # CH1
            np.full(n_peaks, 1554.3),  # CH2
            np.full(n_peaks, 1540.6), np.full(n_peaks, 1554.8),  # CH3
            np.full(n_peaks, 1526.2), np.full(n_peaks, 1540.5), np.full(n_peaks, 1554.8),  # CH4
        ]
    )
    write_peaks(tmp_path / "Peaks.synth.txt", peak_stamps, (1, 1, 2, 3), peak_rows)

    out = tmp_path / "result.csv"
    script = _load_script()
    script.main(
        [
            str(tmp_path / "Responses.synth.txt"),
            "--peaks", str(tmp_path / "Peaks.synth.txt"),
            "--channel", "2",
            "--reference", "1470",
            "--sg-window", "11",
            "-o", str(out),
        ]
    )

    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_frames

    # Time axis: elapsed seconds from the first spectrum.
    assert [float(r["Time_s"]) for r in rows] == [0.0, 20.0, 40.0, 60.0, 80.0]

    # FPI: the crest moves by lambda * dOPD/OPD per step (~4 pm here).
    fpi = np.array([float(r["FPI_Wavelength_nm"]) for r in rows])
    steps = np.diff(fpi) * 1e3
    expected = fpi[0] * 15.0 / 87_000.0 * 1e3
    assert np.all(np.abs(steps - expected) < 0.6 * expected)

    # FBGs come from the Peaks stream, aligned to each spectrum's time:
    # CH1 drifts 0.1 pm per peak sample = 1 pm per spectrum step.
    ch1 = np.array([float(r["CH1_peak1_nm"]) for r in rows])
    np.testing.assert_allclose(np.diff(ch1) * 1e3, 1.0, atol=0.2)
    assert float(rows[0]["CH3_peak2_nm"]) == 1554.8

    # No potentiostat columns.
    assert not any("Volt" in k or "Current" in k for k in rows[0])


def test_script_runs_without_a_peaks_file(tmp_path):
    n_frames, n_points = 3, 8192
    wl = synth.wavelength_axis(n_points=n_points)
    blocks = np.stack(
        [
            np.stack([synth.fp_spectrum(wl, 87_000.0)] * 4)
            for _ in range(n_frames)
        ]
    )
    stamps = [datetime(2026, 1, 5) + timedelta(seconds=20 * i) for i in range(n_frames)]
    write_responses(tmp_path / "Responses.synth.txt", stamps, blocks)

    out = tmp_path / "result.csv"
    _load_script().main(
        [str(tmp_path / "Responses.synth.txt"), "--channel", "1",
         "--reference", "1470", "-o", str(out)]
    )
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == n_frames
    assert "FPI_Wavelength_nm" in rows[0]


def test_spectra_outside_the_peak_stream_get_nan(tmp_path):
    # The peak stream covers only the first spectrum; the others must come
    # out as NaN rather than borrowing a stale sample.
    n_frames, n_points = 3, 8192
    wl = synth.wavelength_axis(n_points=n_points)
    blocks = np.stack(
        [np.stack([synth.fp_spectrum(wl, 87_000.0)] * 4) for _ in range(n_frames)]
    )
    t0 = datetime(2026, 1, 5, 12, 0, 0)
    stamps = [t0 + timedelta(seconds=60 * i) for i in range(n_frames)]
    write_responses(tmp_path / "Responses.synth.txt", stamps, blocks)
    write_peaks(
        tmp_path / "Peaks.synth.txt",
        [t0 + timedelta(seconds=i) for i in range(3)],
        (1, 1, 2, 3),
        np.full((3, 7), 1525.0),
    )

    out = tmp_path / "result.csv"
    _load_script().main(
        [str(tmp_path / "Responses.synth.txt"), "--peaks",
         str(tmp_path / "Peaks.synth.txt"), "--channel", "1",
         "--reference", "1470", "--sg-window", "3", "-o", str(out)]
    )
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]["CH1_peak1_nm"]) == 1525.0
    assert np.isnan(float(rows[1]["CH1_peak1_nm"]))
    assert np.isnan(float(rows[2]["CH1_peak1_nm"]))
