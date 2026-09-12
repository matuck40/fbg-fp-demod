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

from fbgfp import io, synth
from test_demodulate import _load_script
from test_io import write_peaks, write_responses
from test_potentiostat import write_potentiostat

T0 = datetime(2026, 9, 7, 17, 37, 3)


def _responses(tmp_path, n_frames=3, n_points=8192, t0=T0):
    wl = synth.wavelength_axis(n_points=n_points)
    blocks = np.stack(
        [np.stack([synth.fp_spectrum(wl, 87_000.0)] * 4) for _ in range(n_frames)]
    )
    path = tmp_path / "Responses.synth.txt"
    write_responses(path, [t0 + timedelta(seconds=20 * i) for i in range(n_frames)],
                    blocks)
    return path


def _run(tmp_path, *extra, t0=T0):
    out = tmp_path / "result.txt"
    _load_script().main(
        [str(_responses(tmp_path, t0=t0)), "--channel", "1", "--reference", "1470",
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

    The MATLAB did the same for an empty envelope channel — one label, one
    NaN — and would have stopped on an indexing error for an empty internal
    or external one. Here every group behaves as the envelope did.
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


def test_the_fft_columns_follow_the_matlab_definition(tmp_path):
    """Freq_FFT and Amp_FFT, computed the MATLAB's way and compared exactly.

    The expected figures follow the MATLAB line by line — fftshift, a
    range-normalized magnitude with the DC bin in it, the shifted frequency
    axis — rather than the port's fftfreq formulation. A port that drifted
    to its CSV amplitude, dropped the min subtraction or lost the frequency
    no longer matches.
    """
    lines = _run(tmp_path)
    header, row = lines[6].split("\t"), lines[7].split("\t")
    recorded = io.read_responses(tmp_path / "Responses.synth.txt", channel=1)
    step = recorded.wavelength_nm[1] - recorded.wavelength_nm[0]
    linear = 10.0 ** (recorded.spectra_db[0] / 10.0)
    normalized = (linear - linear.min()) / (linear.max() - linear.min())
    n = normalized.size
    amplitude = np.abs(np.fft.fftshift(np.fft.fft(normalized)))
    amplitude = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
    shifted = (np.arange(n) - n // 2) * ((1.0 / step) / n)
    band = np.flatnonzero((shifted >= 0.030) & (shifted <= 0.042))
    peak = band[np.argmax(amplitude[band])]
    assert row[header.index("Freq_FFT")] == f"{shifted[peak]:.6f}"
    assert row[header.index("Amp_FFT")] == f"{amplitude[peak]:.6f}"


def test_each_electrical_column_comes_from_its_own_source(tmp_path):
    """Five distinct values in, each in its own column out.

    With voltage the only non-zero figure in a test export, current and
    power could trade sources, or charge and discharge, and still pass.
    """
    t0 = datetime(2026, 9, 17, 17, 37, 3)  # day 17: the order is unambiguous
    lines = ["Ewe/V\t<I>/mA\tQ discharge/mA.h\tQ charge/mA.h\tPwe/W\t\ttime/s\t"]
    for k in range(40):
        stamp = t0 + timedelta(seconds=2 * k)
        lines.append("\t".join([
            "3,3000000E+000", "-6,400000000000000E+002", "7,250000000000000E+000",
            "1,250000000000000E+001", "-2,1000000E+000", "0",
            stamp.strftime("%m/%d/%Y %H:%M:%S.%f")[:-2],
        ]))
    (tmp_path / "cell.txt").write_text("\r\n".join(lines), encoding="latin-1")

    out = _run(tmp_path, "--cell", str(tmp_path / "cell.txt"), t0=t0)
    header, row = out[6].split("\t"), out[7].split("\t")
    got = {name: row[header.index(name)] for name in
           ("Voltage(V)", "Current(mA)", "Q_Charge(mAh)", "Q_Discharge(mAh)", "Power(W)")}
    assert got == {"Voltage(V)": "3.300000", "Current(mA)": "-640.000000",
                   "Q_Charge(mAh)": "12.500000", "Q_Discharge(mAh)": "7.250000",
                   "Power(W)": "-2.100000"}


def test_the_matlab_format_says_which_columns_it_could_not_fill(tmp_path, capsys):
    """The MATLAB warned when it found no peak or potentiostat file and wrote
    NaN columns; a silent file of NaN columns reads like a measurement."""
    _run(tmp_path)
    err = capsys.readouterr().err
    assert "--peaks" in err and "--cell" in err
