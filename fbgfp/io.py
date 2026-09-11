"""Readers for the interrogator's export files.

Two formats, both text with pt-PT decimal commas:

* ``Responses``: full spectra. A header (whose first line is its own row
  count) carrying the wavelength grid, then blocks of one timestamp line
  followed by one line of dB values per channel.
* ``Peaks``: the instrument's own peak tracking at full acquisition rate —
  one row per sample: timestamp, per-channel peak counts, then the peak
  wavelengths. For the FBG sensors this stream is the measurement of
  record: the saved spectra are decimated and quantized to the export
  grid, so only the Fabry-Perot fringe (which needs the full spectrum) is
  demodulated from ``Responses``.
"""

import gzip
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.signal import savgol_filter

_TIME_OF_DAY = " %H:%M:%S.%f"

# The instrument stamps its exports in whichever locale it was configured
# with, and announces it in a ``Culture:`` header line. Day-first and
# month-first are indistinguishable for the first twelve days of a month,
# so reading one as the other does not fail — it silently returns the
# wrong date. Cultures absent here fall back to day-first, which is what
# this reader did for every file before the header was consulted.
_DATE_ORDERS = {
    "en-us": "%m/%d/%Y",
    "pt-pt": "%d/%m/%Y",
    "pt-br": "%d/%m/%Y",
    "en-gb": "%d/%m/%Y",
}
_DEFAULT_DATE_ORDER = "%d/%m/%Y"
_TIMESTAMP_FORMAT = _DEFAULT_DATE_ORDER + _TIME_OF_DAY


def _timestamp_format(culture):
    """The timestamp format a ``Culture:`` header line implies."""
    if culture is None:
        return _TIMESTAMP_FORMAT
    order = _DATE_ORDERS.get(culture.strip().lower(), _DEFAULT_DATE_ORDER)
    return order + _TIME_OF_DAY


@dataclass(frozen=True)
class Responses:
    """Parsed spectra: ``spectra_db`` is (n_frames, n_channels, n_points),
    or (n_frames, n_points) when a single channel was selected."""

    timestamps: list
    spectra_db: np.ndarray
    wavelength_nm: np.ndarray


@dataclass(frozen=True)
class Peaks:
    """Parsed peak stream: ``channels[i]`` is (n_samples, counts[i])."""

    timestamps: list
    counts: tuple
    channels: list


def _open_text(path):
    """Open an export for reading, transparently decompressing gzip.

    These exports are text columns of numbers and compress by roughly
    seven to one, which is what makes a sample small enough to ship with
    the source — and what makes keeping a multi-gigabyte recording on
    disk practical. Detected by magic bytes rather than by suffix, so a
    renamed file still reads.
    """
    with open(path, "rb") as probe:
        compressed = probe.read(2) == b"\x1f\x8b"
    if compressed:
        return gzip.open(path, "rt", encoding="latin-1")
    return open(path, "r", encoding="latin-1")


def _parse_header_value(line):
    return float(line.split(":")[1].strip().replace(",", "."))


def _iter_responses(path):
    """Stream a Responses export: the first ``next()`` yields the wavelength
    axis read from the header; every later item is ``(datetime, block)``
    with ``block`` shaped (n_channels, n_points). Internal — use
    ``read_responses`` for the materialized form."""
    with _open_text(path) as handle:
        n_header = int(handle.readline())
        start_nm = step_nm = n_points = culture = None
        for _ in range(n_header - 1):
            line = handle.readline()
            if line.startswith("Culture"):
                culture = line.split(":", 1)[1]
            elif line.startswith("Wavelength Start"):
                start_nm = _parse_header_value(line)
            elif line.startswith("Wavelength Delta"):
                step_nm = _parse_header_value(line)
            elif line.startswith("Number of Points"):
                n_points = int(line.split(":")[1])
        if None in (start_nm, step_nm, n_points):
            raise ValueError(f"{path}: header carries no wavelength grid")
        yield start_nm + step_nm * np.arange(n_points)  # first item: the axis

        stamp_format = _timestamp_format(culture)
        timestamp = None
        channels = []
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                timestamp = datetime.strptime(line.strip(), stamp_format)
                channels = []
                continue
            row = np.asarray(line.replace(",", ".").split("\t"), dtype=float)
            if row.size != n_points:
                raise ValueError(
                    f"{path}: channel line with {row.size} values (expected "
                    f"{n_points}) near {timestamp}"
                )
            channels.append(row)
            if len(channels) == 4:
                yield timestamp, np.vstack(channels)
                timestamp, channels = None, []


def read_responses(path, channel=None, max_spectra=None):
    """Materialize a Responses export.

    ``channel`` is 1-based and PHYSICAL: channel 1 is the first data line
    after each timestamp. (The original MATLAB discarded that line as a
    separator, so its channel numbering was shifted by one.)
    ``max_spectra`` stops reading after that many records — useful to
    preview a multi-gigabyte export.
    """
    if channel is not None and channel < 1:
        raise ValueError(f"channel is 1-based and physical; got {channel}")
    stream = _iter_responses(path)
    wavelength_nm = next(stream)
    timestamps, blocks = [], []
    for timestamp, block in stream:
        if max_spectra is not None and len(blocks) >= max_spectra:
            break
        if channel is not None and channel > block.shape[0]:
            raise ValueError(
                f"channel {channel} requested but the file has "
                f"{block.shape[0]} channels"
            )
        timestamps.append(timestamp)
        # .copy() releases the full 4-channel block; without it every view
        # keeps its parent alive and a selected-channel read of a 4 GB
        # export still retains all four channels in memory.
        blocks.append(block if channel is None else block[channel - 1].copy())
    return Responses(timestamps, np.array(blocks), wavelength_nm)


def read_peaks(path):
    """Read a Peaks export; malformed rows are skipped, not fatal."""
    timestamps, rows = [], []
    counts = None
    with _open_text(path) as handle:
        n_header = int(handle.readline())
        culture = None
        for _ in range(n_header - 1):
            line = handle.readline()
            if line.startswith("Culture"):
                culture = line.split(":", 1)[1]
        stamp_format = _timestamp_format(culture)
        for line in handle:
            parts = line.rstrip("\n").replace(",", ".").split("\t")
            if len(parts) < 2:
                continue
            try:
                stamp = datetime.strptime(parts[0].strip(), stamp_format)
                row_counts = tuple(int(c) for c in parts[1:5])
                values = np.asarray(parts[5 : 5 + sum(row_counts)], dtype=float)
            except (ValueError, IndexError):
                continue
            if values.size != sum(row_counts):
                continue
            if counts is None:
                counts = row_counts
            elif row_counts != counts:
                continue  # layout changed mid-file; keep the established one
            timestamps.append(stamp)
            rows.append(values)
    if counts is None:
        raise ValueError(f"{path}: no valid peak rows found")
    table = np.vstack(rows)
    edges = np.cumsum((0,) + counts)
    channels = [table[:, edges[i] : edges[i + 1]] for i in range(len(counts))]
    return Peaks(timestamps, counts, channels)


def align_peaks(peak_data, spectra_timestamps, sg_order, sg_window, max_gap_s):
    """One row of (filtered) FBG values per spectrum, NaN when no peak
    sample lies within ``max_gap_s`` of the spectrum's timestamp.

    The peak stream runs at the instrument's full acquisition rate while
    spectra are saved every few seconds, so the two have to be married on
    the spectra's timestamps before they can be plotted or written side
    by side. Column names are ``CH{channel}_peak{n}_nm``, in the order the
    export lists them.
    """
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
