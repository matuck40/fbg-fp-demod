"""Tests for the synthetic spectrum generator against known ground truth.

The generator stands in for measured interrogator data, so every property
asserted here is one the demodulation tests will later rely on: the grid
matches the instrument, peaks appear exactly where requested, the fringe
frequency follows the cavity OPD, and noise is reproducible by seed.
"""

import numpy as np
import pytest

from fbgfp import synth

HYPERION_STEP_NM = 0.008


def test_wavelength_axis_reproduces_interrogator_grid():
    wl = synth.wavelength_axis()
    assert wl.shape == (20000,)
    assert wl[0] == pytest.approx(1460.0)
    assert wl[-1] == pytest.approx(1619.992)
    assert np.allclose(np.diff(wl), HYPERION_STEP_NM)


def test_fbg_peak_lands_on_requested_center():
    wl = synth.wavelength_axis()
    spectrum_db = synth.fbg_spectrum(wl, centers_nm=[1525.4])
    assert wl[np.argmax(spectrum_db)] == pytest.approx(1525.4, abs=HYPERION_STEP_NM)


def test_fp_fringe_frequency_matches_cavity_opd():
    wl = synth.wavelength_axis()
    opd_nm = 87_000.0  # the ~87 um cavity of the original first-peak band
    spectrum_db = synth.fp_spectrum(wl, opd_nm=opd_nm)

    linear = 10.0 ** (spectrum_db / 10.0)
    linear -= linear.mean()
    freqs = np.fft.rfftfreq(wl.size, d=HYPERION_STEP_NM)
    dominant = freqs[np.argmax(np.abs(np.fft.rfft(linear)))]

    center = 0.5 * (wl[0] + wl[-1])
    expected = opd_nm / center**2  # local fringe frequency in cycles/nm
    assert dominant == pytest.approx(expected, rel=0.15)


def test_multiplexed_spectrum_carries_both_signatures():
    wl = synth.wavelength_axis()
    spectrum_db = synth.multiplexed_spectrum(
        wl, opd_nm=87_000.0, fbg_centers_nm=[1525.4]
    )
    # The FBG must dominate its own neighbourhood...
    near_fbg = np.abs(wl - 1525.4) < 0.5
    assert wl[near_fbg][np.argmax(spectrum_db[near_fbg])] == pytest.approx(
        1525.4, abs=HYPERION_STEP_NM
    )
    # ...while far from it the FP fringe still oscillates.
    far = wl > 1600.0
    assert spectrum_db[far].max() - spectrum_db[far].min() > 1.0


def test_multiplexing_adds_linear_power_not_db():
    # Align a fringe maximum exactly on the FBG center (a grid point), with
    # both signatures at the same level: the linear sum must rise by
    # 10*log10((2+V)/(1+V)) over the fringe alone. Summing in dB instead
    # would give a wildly different number.
    wl = synth.wavelength_axis()
    center = 1525.4
    opd_nm = 87_000.0
    visibility = 0.8
    phase = -2.0 * np.pi * opd_nm / center

    fp_only = synth.fp_spectrum(wl, opd_nm, visibility=visibility, phase_rad=phase)
    both = synth.multiplexed_spectrum(
        wl, opd_nm, [center], visibility=visibility, phase_rad=phase
    )

    at_center = np.argmin(np.abs(wl - center))
    expected_rise = 10.0 * np.log10((2.0 + visibility) / (1.0 + visibility))
    assert both[at_center] - fp_only[at_center] == pytest.approx(expected_rise, abs=1e-3)


def test_fbg_width_at_half_maximum_matches_request():
    fwhm_nm = 0.2
    floor_db = -60.0
    wl = synth.wavelength_axis(start_nm=1524.0, step_nm=0.001, n_points=2801)
    spectrum_db = synth.fbg_spectrum(wl, [1525.4], fwhm_nm=fwhm_nm, floor_db=floor_db)

    # Width is defined on the linear-power bump above the floor.
    bump = 10.0 ** (spectrum_db / 10.0) - 10.0 ** (floor_db / 10.0)
    above_half = wl[bump >= bump.max() / 2.0]
    measured = above_half[-1] - above_half[0]
    assert measured == pytest.approx(fwhm_nm, abs=0.005)


def test_sequence_accepts_a_scalar_opd_as_one_frame():
    wl = synth.wavelength_axis(n_points=2048)
    seq = synth.simulate_sequence(wl, opd_nm=87_000.0)
    assert seq.spectra_db.shape == (1, wl.size)
    assert seq.opd_nm.shape == (1,)


def test_sequence_rejects_one_dimensional_fbg_centers():
    # A flat list of static centers is ambiguous with n_frames rows; requiring
    # an explicit (n_frames, n_fbg) array keeps the stored truth unambiguous.
    wl = synth.wavelength_axis(n_points=2048)
    with pytest.raises(ValueError, match="tile"):
        synth.simulate_sequence(
            wl, opd_nm=np.full(3, 87_000.0), fbg_centers_nm=[1470.5, 1480.2, 1490.0]
        )


def test_sequence_returns_spectra_and_the_truth_that_made_them():
    wl = synth.wavelength_axis(n_points=4096)
    opd_nm = np.linspace(87_000.0, 87_050.0, 10)
    fbg_centers_nm = np.tile([1470.5, 1480.2], (10, 1))

    seq = synth.simulate_sequence(
        wl, opd_nm=opd_nm, fbg_centers_nm=fbg_centers_nm, noise_db=0.05, seed=7
    )

    assert seq.spectra_db.shape == (10, wl.size)
    np.testing.assert_array_equal(seq.opd_nm, opd_nm)
    np.testing.assert_array_equal(seq.fbg_centers_nm, fbg_centers_nm)
    np.testing.assert_array_equal(seq.wavelength_nm, wl)


def test_same_seed_reproduces_the_same_sequence():
    wl = synth.wavelength_axis(n_points=2048)
    opd_nm = np.full(5, 87_000.0)

    kwargs = dict(noise_db=0.1, drift_amplitude_db=0.5)
    a = synth.simulate_sequence(wl, opd_nm=opd_nm, seed=42, **kwargs)
    b = synth.simulate_sequence(wl, opd_nm=opd_nm, seed=42, **kwargs)
    c = synth.simulate_sequence(wl, opd_nm=opd_nm, seed=43, **kwargs)

    np.testing.assert_array_equal(a.spectra_db, b.spectra_db)
    assert not np.array_equal(a.spectra_db, c.spectra_db)


def test_noise_and_drift_default_to_off():
    wl = synth.wavelength_axis(n_points=2048)
    opd_nm = np.full(3, 87_000.0)

    seq = synth.simulate_sequence(wl, opd_nm=opd_nm)

    # Without noise or drift every frame of a constant trajectory is identical.
    np.testing.assert_array_equal(seq.spectra_db[0], seq.spectra_db[1])
    np.testing.assert_array_equal(seq.spectra_db[0], seq.spectra_db[2])
