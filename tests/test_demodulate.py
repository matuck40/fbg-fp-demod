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
import pytest
from test_io import write_peaks, write_responses

from fbgfp import synth


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


def test_missing_file_gives_a_clean_error(tmp_path, capsys):
    import pytest as _pytest

    with _pytest.raises(SystemExit, match="no such file"):
        _load_script().main(
            ["/nowhere/Responses*.txt", "-o", str(tmp_path / "out.csv")]
        )


def test_the_cell_record_is_written_alongside_the_optical_columns(tmp_path):
    """Given a potentiostat export, its electrical columns ride along.

    The MATLAB wrote voltage, current, charge, discharge and power on the
    same row as each spectrum. A column the export does not carry comes out
    as NaN — the short CCCV export has no separate charge and discharge
    counters, and a guessed value would be worse than an empty one.
    """
    from test_potentiostat import write_potentiostat

    n_frames, n_points = 3, 8192
    wl = synth.wavelength_axis(n_points=n_points)
    blocks = np.stack(
        [np.stack([synth.fp_spectrum(wl, 87_000.0)] * 4) for _ in range(n_frames)]
    )
    t0 = datetime(2026, 9, 7, 17, 37, 3)
    write_responses(
        tmp_path / "Responses.synth.txt",
        [t0 + timedelta(seconds=20 * i) for i in range(n_frames)],
        blocks,
    )
    # Potentiostat at 2 s cadence: the spectrum at 20 s * i meets sample 10 * i.
    cell_stamps = [t0 + timedelta(seconds=2 * k) for k in range(40)]
    write_potentiostat(tmp_path / "cell.txt", cell_stamps, 3.0 + 0.01 * np.arange(40))

    out = tmp_path / "result.csv"
    _load_script().main(
        [str(tmp_path / "Responses.synth.txt"), "--cell", str(tmp_path / "cell.txt"),
         "--channel", "1", "--reference", "1470", "-o", str(out)]
    )
    with open(out) as f:
        rows = list(csv.DictReader(f))

    np.testing.assert_allclose(
        [float(r["Voltage_V"]) for r in rows], [3.0, 3.1, 3.2], atol=1e-6
    )
    assert float(rows[0]["Current_mA"]) == 0.0
    assert float(rows[0]["Power_W"]) == 0.0
    assert np.isnan(float(rows[0]["Q_Charge_mAh"]))
    assert np.isnan(float(rows[0]["Q_Discharge_mAh"]))


def _quiet_responses(tmp_path, t0, n_frames=3, n_points=8192):
    wl = synth.wavelength_axis(n_points=n_points)
    blocks = np.stack(
        [np.stack([synth.fp_spectrum(wl, 87_000.0)] * 4) for _ in range(n_frames)]
    )
    path = tmp_path / "Responses.synth.txt"
    write_responses(path, [t0 + timedelta(seconds=20 * i) for i in range(n_frames)], blocks)
    return path


_BASIC = ["--channel", "1", "--reference", "1470"]


@pytest.mark.parametrize("value", ["5", "0", "3_0", "ch 3", "ch"])
def test_an_fbg_channel_outside_the_interrogator_is_refused(tmp_path, value):
    """The interrogator has four channels; anything else is a typo.

    Accepting it wrote a silent all-NaN group: 5 names no channel, and
    Python's int() reads 3_0 as 30.
    """
    responses = _quiet_responses(tmp_path, datetime(2026, 9, 7, 17, 37, 3))
    with pytest.raises(SystemExit):
        _load_script().main([str(responses), *_BASIC, "--fbg-in", value,
                             "-o", str(tmp_path / "out.csv")])


def test_the_other_exports_are_checked_before_the_demodulation(tmp_path, monkeypatch):
    """A mistyped --peaks or --cell path fails at once, not after the spectra.

    A long recording takes far longer to demodulate than its peak stream or
    cell record take to read, and nothing is written until the end, so a bad
    path discovered last costs the whole run.
    """
    script = _load_script()

    def demodulation(*args, **kwargs):
        raise AssertionError("demodulated before the other inputs were read")

    monkeypatch.setattr(script.track, "track_fp", demodulation)
    responses = _quiet_responses(tmp_path, datetime(2026, 9, 7, 17, 37, 3))
    for option in ("--peaks", "--cell"):
        with pytest.raises(SystemExit, match="no such file"):
            script.main([str(responses), *_BASIC, option, str(tmp_path / "missing.txt"),
                         "-o", str(tmp_path / "out.csv")])


def test_a_cell_export_that_is_not_one_is_refused_cleanly(tmp_path):
    responses = _quiet_responses(tmp_path, datetime(2026, 9, 7, 17, 37, 3))
    wrong = tmp_path / "Peaks.txt"
    write_peaks(wrong, [datetime(2026, 9, 7, 17, 37, 3)], (1, 1, 2, 3), np.full((1, 7), 1525.0))
    with pytest.raises(SystemExit, match="time/s"):
        _load_script().main([str(responses), *_BASIC, "--cell", str(wrong),
                             "-o", str(tmp_path / "out.csv")])


def _ambiguous_cell(tmp_path, t0):
    """A long-form cell export dated month-first with no day above twelve, so
    the dates alone cannot tell which order they are in."""
    from test_potentiostat import write_full_export

    stamps = [t0 + timedelta(seconds=2 * k) for k in range(40)]
    path = tmp_path / "cell.txt"
    write_full_export(path, stamps, 3.0 + 0.01 * np.arange(40), np.zeros(40))
    return path


def test_month_first_cell_dates_can_be_declared(tmp_path):
    t0 = datetime(2026, 3, 5, 10, 0, 0)
    responses = _quiet_responses(tmp_path, t0)
    out = tmp_path / "out.csv"
    _load_script().main([str(responses), *_BASIC, "--cell", str(_ambiguous_cell(tmp_path, t0)),
                         "--cell-monthfirst", "-o", str(out)])
    with open(out) as f:
        rows = list(csv.DictReader(f))
    np.testing.assert_allclose([float(r["Voltage_V"]) for r in rows], [3.0, 3.1, 3.2], atol=1e-6)


def test_a_cell_record_that_misses_every_spectrum_is_reported(tmp_path, capsys):
    """Read in the wrong date order, a cell record lands months from the
    spectra and every electrical column comes out NaN. Say so and name the
    flag, rather than write empty columns and exit as if nothing happened."""
    t0 = datetime(2026, 3, 5, 10, 0, 0)
    responses = _quiet_responses(tmp_path, t0)
    _load_script().main([str(responses), *_BASIC, "--cell", str(_ambiguous_cell(tmp_path, t0)),
                         "-o", str(tmp_path / "out.csv")])
    assert "--cell-monthfirst" in capsys.readouterr().err
