"""Tests for single-spectrum preprocessing and valley location.

Ground truth comes from the fringe physics: for a cavity of OPD D the
fringe maxima sit at D/m and the valleys at D/(m + 1/2), m integer.
"""

import numpy as np
import pytest

from fbgfp import peaks, synth

OPD_NM = 87_000.0
BAND = (0.030, 0.042)  # cycles/nm, the original first-peak pass band
STEP_NM = 0.008


def _predicted_valleys(wl):
    orders = np.arange(1, int(OPD_NM / wl[0]) + 2) + 0.5
    valleys = OPD_NM / orders
    return valleys[(valleys > wl[0]) & (valleys < wl[-1])]


def test_linearize_maps_spectrum_to_unit_range():
    wl = synth.wavelength_axis()
    spectrum_db = synth.fp_spectrum(wl, OPD_NM)
    linear = peaks.linearize(spectrum_db)
    assert linear.min() == pytest.approx(0.0)
    assert linear.max() == pytest.approx(1.0)
    # Monotonic mapping: the strongest sample stays the strongest.
    assert np.argmax(linear) == np.argmax(spectrum_db)


def test_bandpass_removes_baseline_and_keeps_the_fringe_shape():
    wl = synth.wavelength_axis()
    fp_only = peaks.linearize(synth.fp_spectrum(wl, OPD_NM))
    multiplexed = peaks.linearize(
        synth.multiplexed_spectrum(wl, OPD_NM, [1525.4, 1554.8])
    )

    filtered_fp = peaks.fft_bandpass(fp_only, STEP_NM, BAND)
    filtered_mux = peaks.fft_bandpass(multiplexed, STEP_NM, BAND)

    # DC (the baseline) is outside the band.
    assert abs(filtered_mux.mean()) < 1e-9
    # The fringe survives with meaningful amplitude...
    assert filtered_fp.max() - filtered_fp.min() > 0.1
    # ...and adding FBGs does not change the shape of the filtered fringe
    # (amplitude may rescale: linearize normalizes by the spectrum's range).
    assert np.corrcoef(filtered_fp, filtered_mux)[0, 1] > 0.999


def test_valleys_found_where_the_cavity_predicts():
    wl = synth.wavelength_axis()
    filtered = peaks.fft_bandpass(
        peaks.linearize(synth.fp_spectrum(wl, OPD_NM)), STEP_NM, BAND
    )
    found = peaks.find_valleys(wl, filtered)

    # The pass band admits only two FFT bins (resolution 0.00625 cycles/nm),
    # so the filtered extrema sit up to ~1 nm from the physical valleys —
    # a property of the original method, not a defect of this port. What
    # tracking needs is one filtered valley per physical valley, well within
    # half a fringe period (~14 nm) so the pairing is unambiguous.
    interior = _predicted_valleys(wl)
    interior = interior[(interior > wl[0] + 20.0) & (interior < wl[-1] - 20.0)]
    assert interior.size >= 4
    for valley in interior:
        assert np.abs(found - valley).min() < 2.0


def test_nearest_valley_pair_picks_reference_neighbours():
    valleys = [1500.0, 1528.0, 1556.0, 1584.0]
    left, right = peaks.nearest_valley_pair(valleys, reference_nm=1554.0)
    assert left == 1556.0
    assert right == 1584.0


def test_nearest_valley_pair_refuses_the_last_valley():
    # The original MATLAB indexed one past the end here; we fail loudly.
    with pytest.raises(ValueError, match="last"):
        peaks.nearest_valley_pair([1500.0, 1528.0, 1556.0], reference_nm=1570.0)


def test_nearest_valley_pair_needs_two_valleys():
    with pytest.raises(ValueError, match="two"):
        peaks.nearest_valley_pair([1500.0], reference_nm=1500.0)
