"""Trajectory tracking across a sequence of spectra.

Composes the single-spectrum primitives: per frame the FP fringe is
band-passed, its valleys located, the crest bracketing the reference is
fitted, and the reference updates to the tracked valley (never the crest —
see ``peaks.nearest_valley_pair``). Fringe hops are unwrapped against the
locally measured valley spacing, reimplementing the recovery the original
MATLAB pipeline left commented out: the raw reading jumps by one fringe,
the corrected trajectory does not.
"""

from dataclasses import dataclass

import numpy as np

from fbgfp import fit, peaks


@dataclass(frozen=True)
class FpTrack:
    """Per-frame FP readings; ``corrected_nm`` is continuous across hops."""

    crest_nm: np.ndarray  # raw fitted crest per frame
    corrected_nm: np.ndarray  # crest minus the accumulated fringe offset
    valley_nm: np.ndarray  # tracked left valley per frame
    hop_frames: tuple  # frames where the tracked fringe changed


def track_fp(spectra_db, wavelength_nm, band_cycles_per_nm, reference_nm, *, trim=0.1):
    """Follow one FP fringe crest through a sequence of spectra."""
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    spectra_db = np.atleast_2d(np.asarray(spectra_db, dtype=float))
    step_nm = wavelength_nm[1] - wavelength_nm[0]
    n_frames = spectra_db.shape[0]

    crest = np.empty(n_frames)
    corrected = np.empty(n_frames)
    valley = np.empty(n_frames)
    hops = []
    offset = 0.0
    reference = float(reference_nm)

    for i in range(n_frames):
        filtered = peaks.fft_bandpass(
            peaks.linearize(spectra_db[i]), step_nm, band_cycles_per_nm
        )
        valleys = peaks.find_valleys(wavelength_nm, filtered)
        try:
            left, right = peaks.nearest_valley_pair(valleys, reference)
        except ValueError:
            if valleys.size < 2:
                raise
            # The tracked fringe lost its right neighbour at the spectrum
            # edge: hand over to the last crest still fully in view. The
            # unwrap below absorbs the one-fringe step this causes.
            left, right = valleys[-2], valleys[-1]

        spacing = right - left
        if i > 0:
            n_fringes = int(np.rint((left - valley[i - 1]) / spacing))
            if n_fringes:
                offset += n_fringes * spacing
                hops.append(i)

        crest[i] = fit.fit_fringe_crest(
            wavelength_nm, filtered, left, right, trim=trim
        ).center
        corrected[i] = crest[i] - offset
        valley[i] = left
        reference = left

    return FpTrack(crest, corrected, valley, tuple(hops))


def track_fbg(spectra_db, wavelength_nm, initial_centers_nm, *, window_nm=1.0):
    """Follow FBG peaks through a sequence by windowed Gaussian fits.

    Each centre is refitted on a ``window_nm``-wide window around its
    previous position, so the window follows the peak. Returns an array of
    shape (n_frames, n_fbg).
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    spectra_db = np.atleast_2d(np.asarray(spectra_db, dtype=float))
    centers = np.atleast_1d(np.asarray(initial_centers_nm, dtype=float)).copy()

    recovered = np.empty((spectra_db.shape[0], centers.size))
    for i in range(spectra_db.shape[0]):
        linear = peaks.linearize(spectra_db[i])
        for k, center in enumerate(centers):
            window = np.abs(wavelength_nm - center) < window_nm / 2.0
            recovered[i, k] = fit.fit_gaussian(
                wavelength_nm[window], linear[window]
            ).center
        centers = recovered[i].copy()
    return recovered
