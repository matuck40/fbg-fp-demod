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
FPI wavelength, the dominant in-band fringe frequency and amplitude, one
column per FBG peak and, given a potentiostat export with ``--cell``, the
cell's voltage, current, charge, discharge and power on the same row.
"""

import argparse
import csv
import glob
import sys

import numpy as np

from fbgfp import io, peaks as fpeaks, track

# The electrical columns the MATLAB wrote beside each spectrum: the name in
# this script's CSV, the name in the MATLAB's text output, and the column
# of the BioLogic export it is read from. A column the export lacks is
# written as NaN — the short CCCV export has no separate charge and
# discharge counters, and a guessed value would be worse than an empty one.
ELECTRICAL = (
    ("Voltage_V", "Voltage(V)", "Ewe/V"),
    ("Current_mA", "Current(mA)", "<I>/mA"),
    ("Q_Charge_mAh", "Q_Charge(mAh)", "Q charge/mA.h"),
    ("Q_Discharge_mAh", "Q_Discharge(mAh)", "Q discharge/mA.h"),
    ("Power_W", "Power(W)", "Pwe/W"),
)


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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Demodulate interrogator exports into an optical time series."
    )
    parser.add_argument("responses", nargs="+", help="Responses file(s) or glob pattern(s)")
    parser.add_argument("--peaks", help="Peaks export with the instrument's FBG tracking")
    parser.add_argument("--cell", help="potentiostat export (BioLogic text) for the "
                                       "electrical columns")
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
                        help="max seconds between a spectrum and its peak or cell "
                             "sample (default 5)")
    parser.add_argument("-o", "--output", required=True, help="output CSV path")
    args = parser.parse_args(argv)

    paths = sorted(p for pattern in args.responses for p in glob.glob(pattern) or [pattern])

    timestamps, spectra = [], []
    wavelength_nm = None
    for path in paths:
        try:
            loaded = io.read_responses(path, channel=args.channel)
        except FileNotFoundError:
            raise SystemExit(
                f"no such file: {path} — check the path (glob patterns must "
                "be quoted so the shell does not expand them)"
            )
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
        peak_names, peak_rows = io.align_peaks(
            peak_data, timestamps, args.sg_order, args.sg_window, args.max_gap
        )

    electrical = None
    if args.cell:
        try:
            cell = io.read_potentiostat(args.cell)
        except FileNotFoundError:
            raise SystemExit(f"no such file: {args.cell}")
        electrical = [
            io.align_series(cell.timestamps, cell.columns[source], timestamps,
                            max_gap_s=args.max_gap)
            if source in cell.columns
            else np.full(len(timestamps), np.nan)
            for _, _, source in ELECTRICAL
        ]

    base = timestamps[0]
    with open(args.output, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["Time_s", "Timestamp", "FPI_Wavelength_nm", "Freq_FFT_cycles_per_nm",
             "Amp_FFT"] + peak_names
            + ([name for name, _, _ in ELECTRICAL] if electrical is not None else [])
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
            if electrical is not None:
                row += [f"{column[i]:.6f}" for column in electrical]
            writer.writerow(row)

    print(f"{len(timestamps)} spectra from {len(paths)} file(s) -> {args.output}")
    if result.hop_frames:
        print(f"fringe hops at frames: {list(result.hop_frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
