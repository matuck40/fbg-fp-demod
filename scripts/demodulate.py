"""Demodulate interrogator exports into an optical time series.

The Python counterpart of the original MATLAB processing script, optical
signals only: the Fabry-Perot fringe is demodulated from the saved spectra
(``Responses*.txt``), the FBG positions are taken from the instrument's
own peak stream (``Peaks*.txt``, Savitzky-Golay filtered) — the stream
runs at full acquisition rate, so it is the measurement of record for the
FBGs — and everything is aligned on the spectra's timestamps.

Usage:
    python scripts/demodulate.py "Responses*.txt" --peaks Peaks.txt -o out.csv

Output: one CSV row per spectrum with the elapsed time, the demodulated
FPI wavelength, the dominant in-band fringe frequency and amplitude, and
one column per FBG peak. No electrical data is read or written.
"""

import argparse
import csv
import glob
import sys

import numpy as np
from scipy.signal import savgol_filter

from fbgfp import io, peaks as fpeaks, track


def dominant_fringe_component(spectrum_db, step_nm, band):
    """Frequency (cycles/nm) and amplitude of the strongest in-band FFT bin,
    as the original pipeline recorded per spectrum."""
    linear = fpeaks.linearize(spectrum_db)
    freqs = np.fft.rfftfreq(linear.size, d=step_nm)
    magnitude = np.abs(np.fft.rfft(linear - linear.mean()))
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        return float("nan"), float("nan")
    index = np.argmax(np.where(in_band, magnitude, 0.0))
    return freqs[index], 2.0 * magnitude[index] / linear.size


def align_peaks(peak_data, spectra_timestamps, sg_order, sg_window, max_gap_s):
    """One row of (filtered) FBG values per spectrum, NaN when no peak
    sample lies within ``max_gap_s`` of the spectrum's timestamp."""
    base = peak_data.timestamps[0]
    peak_seconds = np.array(
        [(t - base).total_seconds() for t in peak_data.timestamps]
    )
    columns, names = [], []
    for ch_index, channel in enumerate(peak_data.channels):
        window = min(sg_window, channel.shape[0] - (channel.shape[0] + 1) % 2)
        window -= 1 - window % 2  # Savitzky-Golay windows must be odd
        for peak_index in range(channel.shape[1]):
            series = channel[:, peak_index]
            if window > sg_order:
                series = savgol_filter(series, window, sg_order)
            columns.append(series)
            names.append(f"CH{ch_index + 1}_peak{peak_index + 1}_nm")

    rows = []
    for stamp in spectra_timestamps:
        seconds = (stamp - base).total_seconds()
        nearest = int(np.clip(np.searchsorted(peak_seconds, seconds), 1, peak_seconds.size) - 1)
        if nearest + 1 < peak_seconds.size and abs(peak_seconds[nearest + 1] - seconds) < abs(
            peak_seconds[nearest] - seconds
        ):
            nearest += 1
        if abs(peak_seconds[nearest] - seconds) > max_gap_s:
            rows.append([float("nan")] * len(columns))
        else:
            rows.append([column[nearest] for column in columns])
    return names, rows


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Demodulate interrogator exports into an optical time series."
    )
    parser.add_argument("responses", nargs="+", help="Responses file(s) or glob pattern(s)")
    parser.add_argument("--peaks", help="Peaks export with the instrument's FBG tracking")
    parser.add_argument("--channel", type=int, default=2,
                        help="1-based physical spectral channel carrying the FP fringe (default 2)")
    parser.add_argument("--band", type=float, nargs=2, default=(0.030, 0.042),
                        metavar=("LOW", "HIGH"),
                        help="FFT pass band in cycles/nm (default 0.030 0.042)")
    parser.add_argument("--reference", type=float, default=1554.0,
                        help="initial tracking reference in nm (default 1554)")
    parser.add_argument("--trim", type=float, default=0.1,
                        help="Gaussian fit window trim (default 0.1)")
    parser.add_argument("--sg-order", type=int, default=3,
                        help="Savitzky-Golay order for the peak stream (default 3)")
    parser.add_argument("--sg-window", type=int, default=75,
                        help="Savitzky-Golay window for the peak stream (default 75)")
    parser.add_argument("--max-gap", type=float, default=5.0,
                        help="max seconds between a spectrum and its peak sample (default 5)")
    parser.add_argument("-o", "--output", required=True, help="output CSV path")
    args = parser.parse_args(argv)

    paths = sorted(p for pattern in args.responses for p in glob.glob(pattern) or [pattern])

    timestamps, spectra = [], []
    wavelength_nm = None
    for path in paths:
        loaded = io.read_responses(path, channel=args.channel)
        if wavelength_nm is None:
            wavelength_nm = loaded.wavelength_nm
        elif not np.array_equal(loaded.wavelength_nm, wavelength_nm):
            raise SystemExit(f"{path}: wavelength grid differs from the first file")
        timestamps.extend(loaded.timestamps)
        spectra.append(loaded.spectra_db)
    if not timestamps:
        raise SystemExit("no spectra found")
    spectra = np.vstack(spectra)
    step_nm = wavelength_nm[1] - wavelength_nm[0]

    if any(b < a for a, b in zip(timestamps, timestamps[1:])):
        print("warning: spectra are not in chronological order; "
              "check the input file ordering", file=sys.stderr)

    result = track.track_fp(spectra, wavelength_nm, tuple(args.band), args.reference,
                            trim=args.trim)
    fringe = [dominant_fringe_component(s, step_nm, args.band) for s in spectra]

    peak_names, peak_rows = [], None
    if args.peaks:
        peak_data = io.read_peaks(args.peaks)
        peak_names, peak_rows = align_peaks(
            peak_data, timestamps, args.sg_order, args.sg_window, args.max_gap
        )

    base = timestamps[0]
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Time_s", "Timestamp", "FPI_Wavelength_nm", "Freq_FFT_cycles_per_nm",
             "Amp_FFT"] + peak_names
        )
        for i, stamp in enumerate(timestamps):
            row = [
                f"{(stamp - base).total_seconds():.3f}",
                stamp.isoformat(),
                f"{result.corrected_nm[i]:.6f}",
                f"{fringe[i][0]:.6f}",
                f"{fringe[i][1]:.6e}",
            ]
            if peak_rows is not None:
                row += [f"{value:.6f}" for value in peak_rows[i]]
            writer.writerow(row)

    print(f"{len(timestamps)} spectra from {len(paths)} file(s) -> {args.output}")
    if result.hop_frames:
        print(f"fringe hops at frames: {list(result.hop_frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
