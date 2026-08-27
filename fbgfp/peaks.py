"""Single-spectrum preprocessing and valley location for FP demodulation.

The chain mirrors the original MATLAB pipeline, split into testable steps:
linearize the recorded dB spectrum, isolate the Fabry-Perot fringe with an
ideal band-pass in spatial frequency (cycles per nm), then locate fringe
valleys and pick the pair that brackets the crest to be fitted.
"""

import numpy as np
from scipy.signal import find_peaks


def linearize(spectrum_db):
    """dB optical power to linear, min-max normalized to [0, 1]."""
    linear = 10.0 ** (np.asarray(spectrum_db, dtype=float) / 10.0)
    low, high = linear.min(), linear.max()
    if high == low:
        raise ValueError("flat spectrum: nothing to normalize (dead frame?)")
    return (linear - low) / (high - low)


def fft_bandpass(signal, step_nm, band_cycles_per_nm):
    """Zero-phase ideal band-pass in spatial frequency.

    Keeps only components whose absolute frequency (cycles/nm) lies inside
    ``band_cycles_per_nm``; masking positive and negative frequencies alike
    preserves conjugate symmetry, so the inverse transform is real.
    """
    low, high = band_cycles_per_nm
    signal = np.asarray(signal, dtype=float)
    freqs = np.fft.fftfreq(signal.size, d=step_nm)
    mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    return np.real(np.fft.ifft(np.fft.fft(signal) * mask))


def find_valleys(wavelength_nm, signal):
    """Wavelengths of the local minima of ``signal``, in ascending order."""
    indices, _ = find_peaks(-np.asarray(signal, dtype=float))
    return np.asarray(wavelength_nm, dtype=float)[indices]


def nearest_valley_pair(valley_wavelengths_nm, reference_nm):
    """The valley nearest to ``reference_nm`` and its right neighbour.

    These two valleys bracket the fringe crest whose Gaussian centre is the
    demodulated reading. ``reference_nm`` tracks the LEFT VALLEY: when
    following a sequence, update it to the returned left valley (as the
    original pipeline updated ``ref_wave``), never to the fitted crest —
    the crest sits mid-fringe, where the ~1 nm displacement of filtered
    valleys can flip the nearest-valley choice to the next fringe.

    Raises ``ValueError`` when no right neighbour exists (the original
    MATLAB indexed past the end of the array here).
    """
    valleys = np.asarray(valley_wavelengths_nm, dtype=float)
    if valleys.size < 2:
        raise ValueError("need at least two valleys to bracket a crest")
    nearest = int(np.argmin(np.abs(valleys - reference_nm)))
    if nearest == valleys.size - 1:
        raise ValueError(
            "tracked valley is the last one detected; no right neighbour "
            "brackets the crest"
        )
    return valleys[nearest], valleys[nearest + 1]
