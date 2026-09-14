"""Tests against the recorded sample shipped in ``sample/``.

Everywhere else the suite writes the files it reads, which proves the
readers handle the format as documented rather than as the instruments
actually write it. These tests close that gap on a real excerpt: thirteen
minutes of a lithium cell under a 2C discharge, with the spectra, the
instrument's own peak stream and the potentiostat's record of the same
minutes — three instruments, three clocks, three locales.

Unlike the synthetic benchmark there is no ground truth here, so these
assert the plumbing and the shape of what was measured, not accuracy.
"""

from pathlib import Path

import numpy as np
import pytest

from fbgfp import io, track

SAMPLE = Path(__file__).resolve().parent.parent / "sample"
RESPONSES = SAMPLE / "Responses.sample.txt.gz"
PEAKS = SAMPLE / "Peaks.sample.txt.gz"
CELL = SAMPLE / "Cell.sample.txt.gz"

# The settings the README documents for this sample.
CHANNEL, BAND, REFERENCE_NM = 2, (0.155, 0.185), 1540.0


@pytest.fixture(scope="module")
def loaded():
    return io.read_responses(RESPONSES, channel=CHANNEL)


def test_the_shipped_sample_reads_as_the_instrument_wrote_it(loaded):
    assert loaded.spectra_db.shape == (40, 20000)
    assert loaded.wavelength_nm[0] == pytest.approx(1460.0)
    assert loaded.wavelength_nm[-1] == pytest.approx(1619.992)
    gaps = np.diff([t.timestamp() for t in loaded.timestamps])
    assert np.all(gaps > 0)
    assert gaps.mean() == pytest.approx(20.0, abs=0.5)
    # Written with Culture: en-US, so 9/7/2026 is September, not July.
    assert (loaded.timestamps[0].month, loaded.timestamps[0].day) == (9, 7)


def test_the_method_measures_the_cell_swelling(loaded):
    result = track.track_fp(
        loaded.spectra_db, loaded.wavelength_nm, BAND, REFERENCE_NM, trim=0.1
    )
    assert result.hop_frames == (), "the window was chosen to need no unwrap"
    shift_pm = (result.corrected_nm - result.corrected_nm[0]) * 1e3
    # The cavity shortens by nearly four nanometres across the discharge.
    assert shift_pm[-1] < -3000.0
    assert np.ptp(shift_pm) > 3000.0
    # Comfortably inside one fringe per frame, which is what keeps the
    # tracker from mistaking the motion for a hop.
    free_spectral_range_nm = 1.0 / np.mean(BAND)
    assert np.abs(np.diff(result.crest_nm)).max() < 0.5 * free_spectral_range_nm


def test_the_peak_stream_aligns_onto_the_recorded_spectra(loaded):
    names, rows = io.align_peaks(io.read_peaks(PEAKS), loaded.timestamps, 3, 75, 5.0)
    rows = np.asarray(rows, dtype=float)
    assert rows.shape == (40, len(names))
    assert not np.isnan(rows).any(), "the shipped window brackets every spectrum"
    assert names == ["CH2_peak1_nm", "CH3_peak1_nm", "CH3_peak2_nm"]
    assert np.all((rows > 1460.0) & (rows < 1620.0))


def test_the_cell_record_aligns_onto_the_recorded_spectra(loaded):
    """The potentiostat writes day-first while the interrogator wrote
    month-first for the same minutes; both must land on the same clock."""
    cell = io.read_potentiostat(CELL)
    voltage = io.align_series(
        cell.timestamps, cell.columns["Ewe/V"], loaded.timestamps, max_gap_s=5.0
    )
    assert voltage.shape == (40,)
    assert not np.isnan(voltage).any()
    assert 2.4 < voltage.min() < voltage.max() < 4.3
    current = io.align_series(
        cell.timestamps, cell.columns["<I>/mA"], loaded.timestamps, max_gap_s=5.0
    )
    assert current.min() < -1000.0, "this window is a discharge"
