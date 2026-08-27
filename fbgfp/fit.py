"""Gaussian fitting: sub-sample peak centres from sampled spectra.

The model and start values mirror the original MATLAB ``fit`` call — a
four-parameter Gaussian with offset, started from the window's maximum,
mean, standard deviation and minimum.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def gaussian(x, amplitude, center, sigma, offset):
    """Four-parameter Gaussian: a*exp(-(x-b)^2 / (2 c^2)) + d."""
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2)) + offset


@dataclass(frozen=True)
class GaussianFit:
    """Fitted parameters; ``center`` carries the units of the fitted axis."""

    amplitude: float
    center: float
    sigma: float
    offset: float


def fit_gaussian(x, y):
    """Least-squares Gaussian fit over the given window."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    start = [y.max(), x.mean(), x.std(), y.min()]
    params, _ = curve_fit(gaussian, x, y, p0=start)
    return GaussianFit(*params)


def fit_fringe_crest(wavelength_nm, signal, valley_left_nm, valley_right_nm, *, trim=0.1):
    """Fit the fringe crest bracketed by two valleys.

    The window is trimmed by ``trim`` of the valley separation on each side
    (10% in the original pipeline), so the fit sees the crest rather than
    the valley walls. A fit whose centre converges outside the bracketing
    valleys is rejected: silently accepting it would poison a tracker's
    self-updating reference.
    """
    if not valley_left_nm < valley_right_nm:
        raise ValueError("valley_left_nm must be smaller than valley_right_nm")
    if not 0.0 <= trim < 0.5:
        raise ValueError("trim must lie in [0, 0.5) to leave a window to fit")
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    signal = np.asarray(signal, dtype=float)
    span = valley_right_nm - valley_left_nm
    low = valley_left_nm + trim * span
    high = valley_right_nm - trim * span
    window = (wavelength_nm >= low) & (wavelength_nm <= high)
    result = fit_gaussian(wavelength_nm[window], signal[window])
    if not valley_left_nm <= result.center <= valley_right_nm:
        raise ValueError(
            f"fitted centre {result.center:.4f} lies outside the bracketing "
            f"valleys [{valley_left_nm:.4f}, {valley_right_nm:.4f}]"
        )
    return result
