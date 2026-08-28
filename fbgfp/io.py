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

from dataclasses import dataclass
from datetime import datetime

import numpy as np

_TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S.%f"


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


def _parse_header_value(line):
    return float(line.split(":")[1].strip().replace(",", "."))


def _iter_responses(path):
    """Stream a Responses export: the first ``next()`` yields the wavelength
    axis read from the header; every later item is ``(datetime, block)``
    with ``block`` shaped (n_channels, n_points). Internal — use
    ``read_responses`` for the materialized form."""
    with open(path, "r", encoding="latin-1") as handle:
        n_header = int(handle.readline())
        start_nm = step_nm = n_points = None
        for _ in range(n_header - 1):
            line = handle.readline()
            if line.startswith("Wavelength Start"):
                start_nm = _parse_header_value(line)
            elif line.startswith("Wavelength Delta"):
                step_nm = _parse_header_value(line)
            elif line.startswith("Number of Points"):
                n_points = int(line.split(":")[1])
        if None in (start_nm, step_nm, n_points):
            raise ValueError(f"{path}: header carries no wavelength grid")
        yield start_nm + step_nm * np.arange(n_points)  # first item: the axis

        timestamp = None
        channels = []
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                timestamp = datetime.strptime(line.strip(), _TIMESTAMP_FORMAT)
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


def read_responses(path, channel=None):
    """Materialize a Responses export.

    ``channel`` is 1-based and PHYSICAL: channel 1 is the first data line
    after each timestamp. (The original MATLAB discarded that line as a
    separator, so its channel numbering was shifted by one.)
    """
    if channel is not None and channel < 1:
        raise ValueError(f"channel is 1-based and physical; got {channel}")
    stream = _iter_responses(path)
    wavelength_nm = next(stream)
    timestamps, blocks = [], []
    for timestamp, block in stream:
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
    with open(path, "r", encoding="latin-1") as handle:
        n_header = int(handle.readline())
        for _ in range(n_header - 1):
            handle.readline()
        for line in handle:
            parts = line.rstrip("\n").replace(",", ".").split("\t")
            if len(parts) < 2:
                continue
            try:
                stamp = datetime.strptime(parts[0].strip(), _TIMESTAMP_FORMAT)
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
