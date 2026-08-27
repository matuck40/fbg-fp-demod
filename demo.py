"""Generate a synthetic multiplexed spectrum and preview it.

Run ``python demo.py``: it builds one Fabry-Perot + FBG spectrum on the
interrogator's wavelength grid, plus a short noisy sequence with a moving
cavity, and saves the figure to ``demo.png``. As the project grows this
script grows with it: demodulation and trajectory recovery will be added
in later phases.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fbgfp import synth


def main():
    wl = synth.wavelength_axis()
    fbg_centers = [1525.4, 1540.6, 1554.8]

    spectrum_db = synth.multiplexed_spectrum(wl, opd_nm=87_000.0, fbg_centers_nm=fbg_centers)

    n_frames = 40
    opd_trajectory = 87_000.0 + 20.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_frames))
    seq = synth.simulate_sequence(
        wl,
        opd_nm=opd_trajectory,
        fbg_centers_nm=np.tile(fbg_centers, (n_frames, 1)),
        noise_db=0.2,
        drift_amplitude_db=0.5,
        seed=0,
    )

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), constrained_layout=True)

    axes[0].plot(wl, spectrum_db, lw=0.4)
    axes[0].set_title("Synthetic multiplexed spectrum: FP fringe + 3 FBG peaks")
    axes[0].set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Power (dB)")

    zoom = (wl > 1546.0) & (wl < 1564.0)
    axes[1].plot(wl[zoom], seq.spectra_db[0][zoom], lw=0.6, label="frame 0")
    axes[1].plot(wl[zoom], seq.spectra_db[n_frames // 4][zoom], lw=0.6, label=f"frame {n_frames // 4}")
    axes[1].set_title("Zoom near 1554 nm: fringe moves as the cavity OPD changes")
    axes[1].set_xlabel("Wavelength (nm)")
    axes[1].set_ylabel("Power (dB)")
    axes[1].legend()

    axes[2].plot(seq.opd_nm, ".-")
    axes[2].set_title("Ground-truth cavity OPD trajectory (the demodulator's target)")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("OPD (nm)")

    fig.savefig("demo.png", dpi=150)
    print("Wrote demo.png")


if __name__ == "__main__":
    main()
