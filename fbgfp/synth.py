"""Synthetic spectra for a Fabry-Perot cavity and FBG sensors on one fibre.

Stands in for measured interrogator data: every spectrum is generated from
known parameters, so demodulation results can be compared against the truth
that produced them. Defaults mirror the Luna Hyperion si155 grid and the
battery-cell sensors of the original MATLAB pipeline (1460-1620 nm, 8 pm
step, an ~87 um cavity fringe near 1554 nm, FBG peaks rising to about the
fringe's mean level).

All functions are pure and operate on NumPy arrays; amplitudes are optical
power in dB, as recorded by the instrument. Signatures mix in the linear
power domain, where reflections actually add.
"""

from dataclasses import dataclass

import numpy as np

# A Gaussian's full width at half maximum is sigma * 2*sqrt(2*ln 2).
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _to_linear(level_db):
    return 10.0 ** (np.asarray(level_db, dtype=float) / 10.0)


def _to_db(linear):
    return 10.0 * np.log10(linear)


def wavelength_axis(start_nm=1460.0, step_nm=0.008, n_points=20000):
    """Uniform wavelength grid in nm; defaults reproduce the interrogator's."""
    return start_nm + step_nm * np.arange(n_points)


def _fp_linear(wavelength_nm, opd_nm, visibility, mean_level_db, phase_rad):
    # Two-beam interference: sinusoidal in wavenumber 1/lambda, so the fringe
    # period in nm grows with wavelength, as in the measured spectra.
    fringe = 1.0 + visibility * np.cos(2.0 * np.pi * opd_nm / wavelength_nm + phase_rad)
    return _to_linear(mean_level_db) * fringe


def _fbg_linear(wavelength_nm, centers_nm, fwhm_nm, peak_level_db):
    sigma = fwhm_nm * _FWHM_TO_SIGMA
    bumps = np.zeros_like(wavelength_nm)
    for center in np.atleast_1d(centers_nm):
        bumps += np.exp(-((wavelength_nm - center) ** 2) / (2.0 * sigma**2))
    return _to_linear(peak_level_db) * bumps


def fp_spectrum(
    wavelength_nm,
    opd_nm,
    *,
    visibility=0.8,
    mean_level_db=-35.0,
    phase_rad=0.0,
):
    """Fabry-Perot fringe spectrum in dB for a cavity of the given OPD (nm)."""
    return _to_db(_fp_linear(wavelength_nm, opd_nm, visibility, mean_level_db, phase_rad))


def fbg_spectrum(
    wavelength_nm,
    centers_nm,
    *,
    fwhm_nm=0.2,
    peak_level_db=-35.0,
    floor_db=-60.0,
):
    """FBG reflection peaks in dB: Gaussian bumps rising above a flat floor."""
    linear = _to_linear(floor_db) + _fbg_linear(
        wavelength_nm, centers_nm, fwhm_nm, peak_level_db
    )
    return _to_db(linear)


def multiplexed_spectrum(
    wavelength_nm,
    opd_nm,
    fbg_centers_nm,
    *,
    visibility=0.8,
    mean_level_db=-35.0,
    phase_rad=0.0,
    fbg_fwhm_nm=0.2,
    fbg_peak_level_db=-35.0,
):
    """FP fringe and FBG peaks on the same fibre, summed as linear power."""
    linear = _fp_linear(
        wavelength_nm, opd_nm, visibility, mean_level_db, phase_rad
    ) + _fbg_linear(wavelength_nm, fbg_centers_nm, fbg_fwhm_nm, fbg_peak_level_db)
    return _to_db(linear)


@dataclass(frozen=True)
class SyntheticSequence:
    """A stack of generated spectra together with the truth that made them."""

    wavelength_nm: np.ndarray
    spectra_db: np.ndarray  # shape (n_frames, n_points)
    opd_nm: np.ndarray  # shape (n_frames,)
    fbg_centers_nm: np.ndarray | None  # shape (n_frames, n_fbg) or None


def simulate_sequence(
    wavelength_nm,
    opd_nm,
    fbg_centers_nm=None,
    *,
    visibility=0.8,
    mean_level_db=-35.0,
    phase_rad=0.0,
    fbg_fwhm_nm=0.2,
    fbg_peak_level_db=-35.0,
    noise_db=0.0,
    drift_amplitude_db=0.0,
    drift_period_nm=200.0,
    seed=None,
):
    """Generate one spectrum per trajectory step, with optional noise and drift.

    ``opd_nm`` is the cavity trajectory, one value per frame. ``fbg_centers_nm``
    is an optional (n_frames, n_fbg) array of Bragg peak positions. Noise is
    additive Gaussian in dB; drift is a slow sinusoid across wavelength (well
    below the fringe frequency) whose phase advances over the sequence.
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    opd_nm = np.asarray(opd_nm, dtype=float)
    n_frames = opd_nm.size
    if fbg_centers_nm is not None:
        fbg_centers_nm = np.asarray(fbg_centers_nm, dtype=float)
        if fbg_centers_nm.shape[0] != n_frames:
            raise ValueError("fbg_centers_nm must have one row per frame")

    rng = np.random.default_rng(seed)
    spectra_db = np.empty((n_frames, wavelength_nm.size))
    for i in range(n_frames):
        if fbg_centers_nm is None:
            frame = fp_spectrum(
                wavelength_nm,
                opd_nm[i],
                visibility=visibility,
                mean_level_db=mean_level_db,
                phase_rad=phase_rad,
            )
        else:
            frame = multiplexed_spectrum(
                wavelength_nm,
                opd_nm[i],
                fbg_centers_nm[i],
                visibility=visibility,
                mean_level_db=mean_level_db,
                phase_rad=phase_rad,
                fbg_fwhm_nm=fbg_fwhm_nm,
                fbg_peak_level_db=fbg_peak_level_db,
            )
        if drift_amplitude_db:
            drift_phase = 2.0 * np.pi * i / max(n_frames - 1, 1)
            frame = frame + drift_amplitude_db * np.sin(
                2.0 * np.pi * wavelength_nm / drift_period_nm + drift_phase
            )
        if noise_db:
            frame = frame + rng.normal(0.0, noise_db, wavelength_nm.size)
        spectra_db[i] = frame

    return SyntheticSequence(
        wavelength_nm=wavelength_nm,
        spectra_db=spectra_db,
        opd_nm=opd_nm,
        fbg_centers_nm=fbg_centers_nm,
    )
