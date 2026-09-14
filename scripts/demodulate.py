"""Demodulate interrogator exports into a time series, one row per spectrum.

The Python counterpart of the original MATLAB processing script: the
Fabry-Perot fringe is demodulated from the saved spectra
(``Responses*.txt``), the FBG positions are taken from the instrument's
own peak stream (``Peaks*.txt``, Savitzky-Golay filtered) — the stream
runs at full acquisition rate, so it is the measurement of record for the
FBGs — the cell's electrical record comes from the potentiostat's export,
and everything is aligned on the spectra's timestamps.

Usage:
    python scripts/demodulate.py "Responses*.txt" --peaks Peaks.txt -o out.csv
    python scripts/demodulate.py "Responses*.txt" --peaks Peaks.txt \\
        --cell cell.txt --format matlab -o out.txt

Output: CSV by default, or with ``--format matlab`` the original pipeline's
tab-separated text layout. Each row carries the elapsed time, the
demodulated FPI wavelength, the dominant in-band fringe frequency and
amplitude, the FBG positions and, given ``--cell``, the cell's voltage,
current, charge, discharge and power.
"""

import argparse
import csv
import glob
import re
import sys
from datetime import datetime

import numpy as np

from fbgfp import io, track
from fbgfp import peaks as fpeaks

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
    """Frequency (cycles/nm) and amplitude of the strongest in-band FFT bin.

    The CSV's figure: a single-sided amplitude of the mean-removed spectrum.
    The MATLAB recorded a different quantity under the same name — see
    matlab_fringe_component — so the two layouts' Amp_FFT do not compare.
    """
    linear = fpeaks.linearize(spectrum_db)
    freqs = np.fft.rfftfreq(linear.size, d=step_nm)
    magnitude = np.abs(np.fft.rfft(linear - linear.mean()))
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        return float("nan"), float("nan")
    index = np.argmax(np.where(in_band, magnitude, 0.0))
    return freqs[index], 2.0 * magnitude[index] / linear.size


def matlab_fringe_component(spectrum_db, step_nm, band):
    """Frequency and amplitude of the in-band FFT peak, as the MATLAB wrote them.

    Not the amplitude dominant_fringe_component reports. The MATLAB takes
    the magnitude of the whole FFT of the normalized spectrum, DC bin
    included, rescales it to [0, 1], and only then picks the largest
    in-band bin — so its figure is a fraction of the DC term, and the two
    cannot be compared across formats.
    """
    linear = fpeaks.linearize(spectrum_db)
    magnitude = np.abs(np.fft.fft(linear))
    span = magnitude.max() - magnitude.min()
    amplitude = (magnitude - magnitude.min()) / span if span else np.zeros_like(magnitude)
    freqs = np.fft.fftfreq(linear.size, d=step_nm)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    if not in_band.any():
        return float("nan"), float("nan")
    index = np.argmax(np.where(in_band, amplitude, -np.inf))
    return freqs[index], amplitude[index]


def _peak_channel(value):
    """'ch3' or '3' -> 3: one of the interrogator's four Peaks channels, named
    as the MATLAB named it. Anything else is refused rather than turned into
    a silent all-NaN group."""
    match = re.fullmatch(r"(?:ch)?([1-4])", value.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(
            f"expected a Peaks channel ch1-ch4 (or 1-4), got {value!r}"
        )
    return int(match.group(1))


def _read_export(kind, reader, path):
    """Read a side export, turning a bad path or a wrong file into a message."""
    try:
        return reader(path)
    except FileNotFoundError:
        raise SystemExit(f"no such file: {path}") from None
    except ValueError as error:
        raise SystemExit(
            f"could not read {path} as a {kind} export: {error}"
        ) from None


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _matlab_number(value):
    """%f as MATLAB's fprintf writes it, NaN included."""
    return "NaN" if np.isnan(value) else f"{value:.6f}"


def _write_csv(path, timestamps, fpi_nm, fringe, peak_names, peak_rows, electrical):
    base = timestamps[0]
    with open(path, "w", newline="") as handle:
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
                f"{fpi_nm[i]:.6f}",
                f"{fringe[i][0]:.6f}",
                f"{fringe[i][1]:.6e}",
            ]
            if peak_rows is not None:
                row += [f"{value:.6f}" for value in peak_rows[i]]
            if electrical is not None:
                row += [f"{column[i]:.6f}" for column in electrical]
            writer.writerow(row)


def _write_matlab(path, args, timestamps, fpi_nm, fringe, peak_names, peak_rows,
                  electrical):
    """The original pipeline's text output: a header block, then tab-separated rows.

    Names and column order follow the MATLAB. Its FBG groups are Peaks
    channels chosen per experiment (--fbg-in, --fbg-out, --fbg-env); a group
    whose channel carries no peaks keeps one NaN column, as the MATLAB did
    for the envelope group — an empty internal or external channel would
    have stopped it on an indexing error. The FPI column holds the unwrapped
    reading, which equals the MATLAB's raw pico_x until a fringe hop fires —
    the MATLAB had no unwrap, so past a hop its own column would have jumped
    by a fringe.
    """
    n = len(timestamps)
    rows = None if peak_rows is None else np.asarray(peak_rows, dtype=float)
    groups = []
    for label, channel in (("FBG_in(nm)", args.fbg_in), ("FBG_out(nm)", args.fbg_out),
                           ("FBG_env(nm)", args.fbg_env)):
        wanted = [] if rows is None else [
            j for j, name in enumerate(peak_names) if name.startswith(f"CH{channel}_")
        ]
        columns = [rows[:, j] for j in wanted] or [np.full(n, np.nan)]
        groups += [(label, column) for column in columns]
    if electrical is None:
        electrical = [np.full(n, np.nan) for _ in ELECTRICAL]

    header = (["Time(s)", "FPI_Wavelength(nm)"] + [label for label, _ in groups]
              + ["Freq_FFT", "Amp_FFT"] + [name for _, name, _ in ELECTRICAL])
    now = datetime.now()
    lines = [
        "FPI valley detection ",
        f"Date:{now:%d}-{_MONTHS[now.month - 1]}-{now:%Y %H:%M:%S}",
        f"Initial reference: {args.reference:.2f}",
        f"Band-pass filter frequencies: {args.band[0]:f} {args.band[1]:f}",
        " ",
        " ",
        "\t".join(header),
    ]
    base = timestamps[0]
    for i, stamp in enumerate(timestamps):
        values = ([(stamp - base).total_seconds(), fpi_nm[i]]
                  + [column[i] for _, column in groups]
                  + [fringe[i][0], fringe[i][1]]
                  + [column[i] for column in electrical])
        lines.append("\t".join(_matlab_number(v) for v in values))
    with open(path, "w", newline="") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Demodulate interrogator exports into a time series, "
                    "one row per spectrum."
    )
    parser.add_argument("responses", nargs="+", help="Responses file(s) or glob pattern(s)")
    parser.add_argument("--peaks", help="Peaks export with the instrument's FBG tracking")
    parser.add_argument("--cell", help="potentiostat export (BioLogic text) for the "
                                       "electrical columns")
    parser.add_argument("--cell-monthfirst", action="store_true",
                        help="the cell export writes month-first dates, as the "
                             "long-form BioLogic export does; otherwise day-first "
                             "is assumed unless a day above 12 settles it")
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
                             "sample, NaN beyond it (default 5; inf takes the "
                             "nearest sample however far, as the MATLAB did)")
    parser.add_argument("--format", choices=("csv", "matlab"), default="csv",
                        help="csv (default), or matlab: the original pipeline's "
                             "tab-separated .txt layout")
    parser.add_argument("--fbg-in", type=_peak_channel, default=1,
                        help="Peaks channel of the MATLAB's FBG_in group (default ch1)")
    parser.add_argument("--fbg-out", type=_peak_channel, default=3,
                        help="Peaks channel of the MATLAB's FBG_out group (default ch3)")
    parser.add_argument("--fbg-env", type=_peak_channel, default=4,
                        help="Peaks channel of the MATLAB's FBG_env group (default ch4)")
    parser.add_argument("-o", "--output", required=True, help="output path")
    args = parser.parse_args(argv)

    paths = sorted(p for pattern in args.responses for p in glob.glob(pattern) or [pattern])

    # The side exports are read before the spectra: a long recording takes
    # far longer to demodulate than these take to read, and nothing is
    # written until the end, so a bad path found last would cost the run.
    peak_data = _read_export("Peaks", io.read_peaks, args.peaks) if args.peaks else None
    cell = None
    if args.cell:
        cell = _read_export(
            "cell",
            lambda path: io.read_potentiostat(path, dayfirst=not args.cell_monthfirst),
            args.cell,
        )

    timestamps, spectra = [], []
    wavelength_nm = None
    for path in paths:
        try:
            loaded = io.read_responses(path, channel=args.channel)
        except FileNotFoundError:
            raise SystemExit(
                f"no such file: {path} — check the path (glob patterns must "
                "be quoted so the shell does not expand them)"
            ) from None
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

    if any(b < a for a, b in zip(timestamps, timestamps[1:], strict=False)):
        print("warning: spectra are not in chronological order; "
              "check the input file ordering", file=sys.stderr)

    result = track.track_fp(spectra, wavelength_nm, tuple(args.band), args.reference,
                            trim=args.trim)
    component = (matlab_fringe_component if args.format == "matlab"
                 else dominant_fringe_component)
    fringe = [component(s, step_nm, args.band) for s in spectra]

    peak_names, peak_rows = [], None
    if peak_data is not None:
        peak_names, peak_rows = io.align_peaks(
            peak_data, timestamps, args.sg_order, args.sg_window, args.max_gap
        )

    electrical = None
    if cell is not None:
        electrical = [
            io.align_series(cell.timestamps, cell.columns[source], timestamps,
                            max_gap_s=args.max_gap)
            if source in cell.columns
            else np.full(len(timestamps), np.nan)
            for _, _, source in ELECTRICAL
        ]
        if all(np.isnan(column).all() for column in electrical):
            print(f"warning: the cell export overlaps none of these spectra within "
                  f"--max-gap {args.max_gap:g} s, so every electrical column is NaN; "
                  f"if its dates are month-first, pass --cell-monthfirst",
                  file=sys.stderr)

    if args.format == "matlab":
        if peak_data is None:
            print("warning: no --peaks export given; the FBG columns are written "
                  "as NaN", file=sys.stderr)
        if cell is None:
            print("warning: no --cell export given; the electrical columns are "
                  "written as NaN", file=sys.stderr)

    if args.format == "matlab":
        _write_matlab(args.output, args, timestamps, result.corrected_nm, fringe,
                      peak_names, peak_rows, electrical)
    else:
        _write_csv(args.output, timestamps, result.corrected_nm, fringe,
                   peak_names, peak_rows, electrical)

    print(f"{len(timestamps)} spectra from {len(paths)} file(s) -> {args.output}")
    if result.hop_frames:
        print(f"fringe hops at frames: {list(result.hop_frames)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
