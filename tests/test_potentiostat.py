"""Tests for the potentiostat export reader.

The cell's electrical data comes from a separate instrument with its own
clock, its own locale and its own sampling rate, so reading it is one
problem and marrying it to the spectra is another. Both are tested here
against files written in the exporter's format.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from fbgfp import io

COLUMNS = "time/s\t(Q-Qo)/mA.h\tEwe/V\tPwe/W\t<I>/mA\tCapacity/mA.h\t"


def write_potentiostat(path, timestamps, voltages, dayfirst=True):
    """Write a BioLogic-style CCCV export: tab separated, decimal commas."""
    order = "%d/%m/%Y" if dayfirst else "%m/%d/%Y"
    lines = [COLUMNS]
    for stamp, voltage in zip(timestamps, voltages):
        cells = [
            stamp.strftime(order + " %H:%M:%S.%f")[:-2],
            "0,000000000000000E+000",
            ("%.7E" % voltage).replace(".", ","),
            "0,0000000E+000",
            "0,000000000000000E+000",
            "0,000000000000000E+000",
        ]
        lines.append("\t".join(cells))
    path.write_text("\r\n".join(lines), encoding="latin-1")


def test_the_export_reads_back_with_its_columns(tmp_path):
    stamps = [datetime(2026, 9, 7, 11, 27, 3) + timedelta(seconds=2 * i)
              for i in range(6)]
    voltages = np.linspace(3.174, 3.421, 6)
    path = tmp_path / "cell_CCCV_C01.txt"
    write_potentiostat(path, stamps, voltages)

    result = io.read_potentiostat(path)

    assert result.timestamps == stamps
    assert "Ewe/V" in result.columns
    np.testing.assert_allclose(result.columns["Ewe/V"], voltages, rtol=1e-6)


def test_a_gzipped_export_reads_the_same(tmp_path):
    import gzip

    stamps = [datetime(2026, 9, 7, 11, 27, 3) + timedelta(seconds=2 * i)
              for i in range(4)]
    plain = tmp_path / "cell.txt"
    write_potentiostat(plain, stamps, np.linspace(3.2, 3.4, 4))
    packed = tmp_path / "cell.txt.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes(), 9))

    assert io.read_potentiostat(packed).timestamps == io.read_potentiostat(plain).timestamps


def test_a_day_past_the_twelfth_settles_the_date_order(tmp_path):
    """Where the data can say which order it is in, it must be believed.

    The export carries no locale header, and the potentiostat need not be
    set to the same one as the interrogator recording the same cell. A day
    above twelve cannot be a month, so it decides — and here it decides
    against the caller's default.
    """
    stamps = [datetime(2026, 9, 20, 10, 0, 0) + timedelta(seconds=2 * i)
              for i in range(3)]
    path = tmp_path / "cell.txt"
    write_potentiostat(path, stamps, [3.2, 3.3, 3.4], dayfirst=True)

    result = io.read_potentiostat(path, dayfirst=False)  # wrong default

    assert result.timestamps == stamps


def test_values_are_carried_onto_the_spectra_clock(tmp_path):
    """One voltage per spectrum, NaN where the record does not reach."""
    base = datetime(2026, 9, 7, 11, 27, 3)
    stamps = [base + timedelta(seconds=2 * i) for i in range(30)]
    path = tmp_path / "cell.txt"
    write_potentiostat(path, stamps, np.linspace(3.0, 3.6, 30))
    data = io.read_potentiostat(path)

    spectra = [base + timedelta(seconds=20 * i) for i in range(3)]
    spectra.append(base + timedelta(hours=5))  # far outside the record

    aligned = io.align_series(data.timestamps, data.columns["Ewe/V"], spectra,
                              max_gap_s=5.0)

    assert aligned.shape == (4,)
    assert np.isnan(aligned[-1]), "a spectrum outside the record must be NaN"
    assert np.all(np.isfinite(aligned[:3]))
    assert aligned[0] == pytest.approx(3.0, abs=1e-6)


def write_full_export(path, timestamps, voltages, charges):
    """The long-form export the MATLAB read: time/s near the end, after an
    unnamed column, a header one field longer than its rows, month-first."""
    header = "mode\tEwe/V\tI Range\tQ discharge/mA.h\tQ charge/mA.h\t\ttime/s\t"
    lines = [header]
    for stamp, voltage, charge in zip(timestamps, voltages, charges):
        cells = [
            "3",
            ("%.7E" % voltage).replace(".", ","),
            "14",
            "0,000000000000000E+000",
            ("%.15E" % charge).replace(".", ","),
            "0",
            stamp.strftime("%m/%d/%Y %H:%M:%S.%f")[:-2],
        ]
        lines.append("\t".join(cells))
    path.write_text("\r\n".join(lines), encoding="latin-1")


def test_the_long_form_export_is_read_by_column_name(tmp_path):
    """The time column is found by its name, wherever the export puts it.

    The short CCCV export leads with ``time/s``; the long-form one the
    MATLAB read puts it twenty-third, after an unnamed column, under a
    header one field longer than its rows. Reading by position sent every
    row of that file to the reject pile.
    """
    stamps = [datetime(2025, 3, 22, 14, 24, 24, 419100) + timedelta(seconds=2 * i)
              for i in range(4)]
    voltages = [1.9640788, 1.9640788, 1.9640597, 2.2020700]
    charges = [0.0, 0.0, 0.0, 2.5006520]
    path = tmp_path / "Na Ion completo_CCCV_C01.txt"
    write_full_export(path, stamps, voltages, charges)

    result = io.read_potentiostat(path)

    assert result.timestamps == stamps
    np.testing.assert_allclose(result.columns["Ewe/V"], voltages, rtol=1e-7)
    np.testing.assert_allclose(result.columns["Q charge/mA.h"], charges, atol=1e-12)
    assert "" not in result.columns
    assert "time/s" not in result.columns
