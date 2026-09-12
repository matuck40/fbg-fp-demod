"""The processing script's MATLAB-compatible text output.

The original pipeline wrote a tab-separated .txt with a short header block
and a fixed column order, and whatever reads those files downstream reads
that layout. These tests pin the layout itself: the header lines, the
column names and their order, the FBG channels grouped as the MATLAB
grouped them, and NaN written the way MATLAB's fprintf writes it.
"""

import re
from datetime import datetime, timedelta

import numpy as np

from fbgfp import synth
from test_demodulate import _load_script
from test_io import write_peaks, write_responses
from test_potentiostat import write_potentiostat

T0 = datetime(2026, 9, 7, 17, 37, 3)


def _responses(tmp_path, n_frames=3, n_points=8192):
    wl = synth.wavelength_axis(n_points=n_points)
    blocks = np.stack(
        [np.stack([synth.fp_spectrum(wl, 87_000.0)] * 4) for _ in range(n_frames)]
    )
    path = tmp_path / "Responses.synth.txt"
    write_responses(path, [T0 + timedelta(seconds=20 * i) for i in range(n_frames)],
                    blocks)
    return path


def _run(tmp_path, *extra):
    out = tmp_path / "result.txt"
    _load_script().main(
        [str(_responses(tmp_path)), "--channel", "1", "--reference", "1470",
         "--sg-window", "3", "--format", "matlab", "-o", str(out), *extra]
    )
    return out.read_text().split("\n")


def test_the_matlab_format_reproduces_the_original_layout(tmp_path):
    stamps = [T0 + timedelta(seconds=2 * k) for k in range(30)]
    columns = np.column_stack([
        np.full(30, 1525.0),                              # CH1
        np.full(30, 1554.3),                              # CH2: the FP, no FBG group
        np.full(30, 1540.6), np.full(30, 1554.8),         # CH3
        np.full(30, 1526.2), np.full(30, 1540.5), np.full(30, 1554.9),  # CH4
    ])
    write_peaks(tmp_path / "Peaks.txt", stamps, (1, 1, 2, 3), columns)
    write_potentiostat(tmp_path / "cell.txt", stamps, 3.0 + 0.01 * np.arange(30))

    lines = _run(tmp_path, "--peaks", str(tmp_path / "Peaks.txt"),
                 "--cell", str(tmp_path / "cell.txt"))

    assert lines[0] == "FPI valley detection "
    assert re.fullmatch(r"Date:\d{2}-[A-Z][a-z]{2}-\d{4} \d{2}:\d{2}:\d{2}", lines[1])
    assert lines[2] == "Initial reference: 1470.00"
    assert lines[3] == "Band-pass filter frequencies: 0.030000 0.042000"
    assert lines[4] == " " and lines[5] == " "
    header = lines[6].split("\t")
    assert header == [
        "Time(s)", "FPI_Wavelength(nm)",
        "FBG_in(nm)", "FBG_out(nm)", "FBG_out(nm)",
        "FBG_env(nm)", "FBG_env(nm)", "FBG_env(nm)",
        "Freq_FFT", "Amp_FFT",
        "Voltage(V)", "Current(mA)", "Q_Charge(mAh)", "Q_Discharge(mAh)", "Power(W)",
    ]

    rows = [line.split("\t") for line in lines[7:] if line]
    assert len(rows) == 3
    assert all(len(row) == len(header) for row in rows)
    assert [row[0] for row in rows] == ["0.000000", "20.000000", "40.000000"]
    # Default groups are the MATLAB's: in = ch1, out = ch3, env = ch4.
    assert rows[0][2:8] == ["1525.000000", "1540.600000", "1554.800000",
                            "1526.200000", "1540.500000", "1554.900000"]
    # MATLAB normalizes the whole FFT magnitude to [0, 1] before picking
    # the in-band peak, so its amplitude is a fraction, never above one.
    assert 0.0 < float(rows[0][9]) <= 1.0
    assert [row[10] for row in rows] == ["3.000000", "3.100000", "3.200000"]
    # The short CCCV export has no charge counter: NaN, spelled as MATLAB does.
    assert rows[0][12] == "NaN"


def test_an_empty_fbg_group_still_gets_its_column(tmp_path):
    """A channel with no peaks keeps one NaN column under its header.

    The MATLAB wrote one label per group even when the channel was empty,
    and then no values for it — shifting every later column one place left
    of its header. Here the header and the values stay in step.
    """
    stamps = [T0 + timedelta(seconds=2 * k) for k in range(30)]
    write_peaks(tmp_path / "Peaks.txt", stamps, (0, 1, 2, 0),
                np.column_stack([np.full(30, 1553.9), np.full(30, 1530.8),
                                 np.full(30, 1539.2)]))

    lines = _run(tmp_path, "--peaks", str(tmp_path / "Peaks.txt"),
                 "--fbg-in", "ch1", "--fbg-out", "ch3", "--fbg-env", "ch4")

    header = lines[6].split("\t")
    row = lines[7].split("\t")
    assert len(row) == len(header)
    assert header[2:6] == ["FBG_in(nm)", "FBG_out(nm)", "FBG_out(nm)", "FBG_env(nm)"]
    assert row[2:6] == ["NaN", "1530.800000", "1539.200000", "NaN"]
