"""Tests for Gaussian fitting: sub-sample centres from exact and noisy data.

The declared tolerances are the measured contract of the method on the
interrogator grid: an FBG centre to within 5 pm under 0.2 dB of noise; the
fringe-crest reading within 50 pm of the physical fringe maximum; OPD
shifts recovered within 6% (the two-bin band-limited reconstruction
compresses shifts by ~5%, a property of the original pipeline as well);
crest jitter under noise below 10 pm; and multiplexed FBGs perturbing the
crest by less than 30 pm.
"""

import numpy as np
import pytest

from fbgfp import fit, peaks, synth

OPD_NM = 87_000.0
BAND = (0.030, 0.042)
STEP_NM = 0.008
REFERENCE_NM = 1560.0  # clearly nearest to the valley at ~1568 nm
TRUE_CREST_NM = OPD_NM / 55.0  # fringe maximum of order 55: 1581.818... nm


def test_gaussian_fit_recovers_exact_parameters():
    x = np.linspace(1525.0, 1526.0, 200)
    y = fit.gaussian(x, amplitude=2.0, center=1525.4, sigma=0.08, offset=0.3)
    result = fit.fit_gaussian(x, y)
    assert result.amplitude == pytest.approx(2.0, rel=1e-6)
    assert result.center == pytest.approx(1525.4, abs=1e-6)
    assert result.sigma == pytest.approx(0.08, rel=1e-6)
    assert result.offset == pytest.approx(0.3, abs=1e-6)


def test_fit_recovers_fbg_center_within_5pm_under_noise():
    center_true = 1525.4
    wl = synth.wavelength_axis()
    seq = synth.simulate_sequence(
        wl,
        opd_nm=OPD_NM,
        fbg_centers_nm=[[center_true]],
        noise_db=0.2,
        seed=11,
    )
    linear = peaks.linearize(seq.spectra_db[0])

    window = np.abs(wl - center_true) < 0.5
    result = fit.fit_gaussian(wl[window], linear[window])
    assert result.center == pytest.approx(center_true, abs=0.005)


def _crest_from_single_spectrum(spectrum_db, wl, reference_nm):
    filtered = peaks.fft_bandpass(peaks.linearize(spectrum_db), STEP_NM, BAND)
    valleys = peaks.find_valleys(wl, filtered)
    left, right = peaks.nearest_valley_pair(valleys, reference_nm)
    return fit.fit_fringe_crest(wl, filtered, left, right).center


def test_fringe_crest_matches_the_physical_maximum():
    wl = synth.wavelength_axis()
    spectrum_db = synth.fp_spectrum(wl, OPD_NM)
    center = _crest_from_single_spectrum(spectrum_db, wl, REFERENCE_NM)
    # Absolute bias of the band-limited crest, measured at -14 pm here; it
    # varies with position but stays well under one grid period.
    assert center == pytest.approx(TRUE_CREST_NM, abs=0.050)


def test_fringe_crest_shift_tracks_an_opd_change():
    # The property the application actually uses: a cavity change moves the
    # reading by dlambda = lambda * dOPD / OPD. The two-bin reconstruction
    # compresses the shift by ~5%, so 6% is the declared tolerance.
    wl = synth.wavelength_axis()
    d_opd = 20.0
    before = _crest_from_single_spectrum(synth.fp_spectrum(wl, OPD_NM), wl, REFERENCE_NM)
    after = _crest_from_single_spectrum(
        synth.fp_spectrum(wl, OPD_NM + d_opd), wl, REFERENCE_NM
    )
    predicted = before * d_opd / OPD_NM
    assert after - before == pytest.approx(predicted, rel=0.06)


def test_fringe_crest_is_stable_under_noise():
    wl = synth.wavelength_axis()
    clean = _crest_from_single_spectrum(
        synth.fp_spectrum(wl, OPD_NM), wl, REFERENCE_NM
    )
    noisy_seq = synth.simulate_sequence(wl, opd_nm=OPD_NM, noise_db=0.2, seed=3)
    noisy = _crest_from_single_spectrum(noisy_seq.spectra_db[0], wl, REFERENCE_NM)
    assert noisy == pytest.approx(clean, abs=0.010)


def test_fringe_crest_is_unmoved_by_multiplexed_fbgs():
    wl = synth.wavelength_axis()
    alone = _crest_from_single_spectrum(synth.fp_spectrum(wl, OPD_NM), wl, REFERENCE_NM)
    multiplexed = _crest_from_single_spectrum(
        synth.multiplexed_spectrum(wl, OPD_NM, [1525.4, 1554.8]), wl, REFERENCE_NM
    )
    assert multiplexed == pytest.approx(alone, abs=0.030)


def test_fringe_crest_requires_ordered_valleys():
    wl = synth.wavelength_axis(n_points=2048)
    signal = np.zeros_like(wl)
    with pytest.raises(ValueError, match="left"):
        fit.fit_fringe_crest(wl, signal, 1500.0, 1490.0)


def test_fringe_crest_rejects_a_center_outside_the_bracket():
    # On a pathological window curve_fit can converge silently to a centre
    # far outside the valleys (seed 2 lands at ~-8.9 on this window). The
    # tracker must see an error, not a poisoned reference.
    x = np.linspace(0.0, 1.0, 200)
    noise = np.random.default_rng(2).normal(0.0, 1.0, x.size)
    with pytest.raises(ValueError, match="outside"):
        fit.fit_fringe_crest(x, noise, 0.0, 1.0)


def test_fringe_crest_validates_trim():
    x = np.linspace(0.0, 1.0, 100)
    with pytest.raises(ValueError, match="trim"):
        fit.fit_fringe_crest(x, x, 0.0, 1.0, trim=0.5)


def test_fringe_crest_regression_pins_the_ported_method():
    # Characterization pin, exact to the picometre: the band-passed crest
    # of OPD 87 um fitted with the 10% trim reads 1581.8038 nm. A change
    # in trim, band handling or start values moves this value.
    wl = synth.wavelength_axis()
    center = _crest_from_single_spectrum(synth.fp_spectrum(wl, OPD_NM), wl, REFERENCE_NM)
    assert center == pytest.approx(1581.8038, abs=0.001)
