"""Generate a synthetic multiplexed spectrum and demodulate one frame.

Run ``python demo.py``: it builds a Fabry-Perot + FBG spectrum on the
interrogator's wavelength grid, isolates the fringe with the FFT band-pass,
locates its valleys, fits a Gaussian to the crest bracketing the reference,
and saves the figure to ``demo.png``. Trajectory tracking across a sequence
arrives in a later phase.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fbgfp import fit, peaks, synth

OPD_NM = 87_000.0
BAND = (0.030, 0.042)  # cycles/nm
REFERENCE_NM = 1560.0


def main():
    wl = synth.wavelength_axis()
    step_nm = wl[1] - wl[0]  # derive from the axis so they cannot disagree
    fbg_centers = [1525.4, 1540.6, 1554.8]
    spectrum_db = synth.multiplexed_spectrum(wl, OPD_NM, fbg_centers)

    # Single-frame demodulation: linearize, band-pass, valleys, crest fit.
    filtered = peaks.fft_bandpass(peaks.linearize(spectrum_db), step_nm, BAND)
    valleys = peaks.find_valleys(wl, filtered)
    left, right = peaks.nearest_valley_pair(valleys, REFERENCE_NM)
    crest = fit.fit_fringe_crest(wl, filtered, left, right)

    n_frames = 40
    opd_trajectory = OPD_NM + 20.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_frames))
    seq = synth.simulate_sequence(
        wl,
        opd_nm=opd_trajectory,
        fbg_centers_nm=np.tile(fbg_centers, (n_frames, 1)),
        noise_db=0.2,
        drift_amplitude_db=0.5,
        seed=0,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)

    axes[0, 0].plot(wl, spectrum_db, lw=0.4)
    axes[0, 0].set_title("Synthetic multiplexed spectrum: FP fringe + 3 FBG peaks")
    axes[0, 0].set_xlabel("Wavelength (nm)")
    axes[0, 0].set_ylabel("Power (dB)")

    axes[0, 1].plot(wl, filtered, lw=0.5, label="band-passed fringe")
    valley_idx = np.searchsorted(wl, valleys)
    axes[0, 1].plot(valleys, filtered[valley_idx], "v", ms=5, label="valleys")
    window = (wl >= left) & (wl <= right)
    axes[0, 1].plot(
        wl[window],
        fit.gaussian(wl[window], crest.amplitude, crest.center, crest.sigma, crest.offset),
        "--",
        lw=1.5,
        label=f"Gaussian crest fit: {crest.center:.3f} nm",
    )
    axes[0, 1].axvline(crest.center, color="k", lw=0.8, ls=":")
    axes[0, 1].set_title("FFT band-pass + valley detection + crest fit")
    axes[0, 1].set_xlabel("Wavelength (nm)")
    axes[0, 1].set_ylabel("Filtered amplitude")
    axes[0, 1].legend(loc="lower left", fontsize=8)

    zoom = (wl > 1546.0) & (wl < 1564.0)
    axes[1, 0].plot(wl[zoom], seq.spectra_db[0, zoom], lw=0.6, label="frame 0")
    axes[1, 0].plot(
        wl[zoom], seq.spectra_db[n_frames // 4, zoom], lw=0.6, label=f"frame {n_frames // 4}"
    )
    axes[1, 0].set_title("Zoom near 1554 nm: fringe moves as the cavity OPD changes")
    axes[1, 0].set_xlabel("Wavelength (nm)")
    axes[1, 0].set_ylabel("Power (dB)")
    axes[1, 0].legend()

    axes[1, 1].plot(seq.opd_nm, ".-")
    axes[1, 1].set_title("Ground-truth cavity OPD trajectory (the tracker's target)")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("OPD (nm)")

    fig.savefig("demo.png", dpi=150)
    print("Wrote demo.png")
    print(f"Crest fitted at {crest.center:.4f} nm "
          f"(physical fringe maximum at {OPD_NM / 55.0:.4f} nm)")


if __name__ == "__main__":
    main()
