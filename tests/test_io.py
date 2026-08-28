"""Tests for reading interrogator export files.

No measured data: the tests WRITE synthetic files in the instrument's own
export format (pt-PT decimal commas, header, timestamp blocks) and read
them back, so the reader is exercised against the format, not against any
recorded content.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from fbgfp import io, synth, track


def _comma(values, fmt="%.2f"):
    return "\t".join((fmt % v).replace(".", ",") for v in values)


def write_responses(path, timestamps, blocks, start_nm=1460.0, step_nm=0.008):
    """Write an ENLIGHT-style Responses export: header, then blocks of
    [timestamp, one line per channel], blank line between blocks."""
    n_points = blocks.shape[2]
    header = [
        "Culture: pt-PT ",
        f"Date: {timestamps[0].strftime('%d/%m/%Y %H:%M:%S.%f')[:-1]}",
        "Module Type: Hyperion",
        f"Wavelength Start (nm): {('%.5f' % start_nm).replace('.', ',')}",
        f"Wavelength Delta (nm): {('%.4f' % step_nm).replace('.', ',')}",
        f"Number of Points: {n_points}",
        "",
    ]
    lines = [str(len(header) + 1)] + header
    for stamp, block in zip(timestamps, blocks):
        lines.append(stamp.strftime("%d/%m/%Y %H:%M:%S.%f")[:-1])
        for channel in block:
            lines.append(_comma(channel))
        lines.append("")
    path.write_text("\n".join(lines), encoding="latin-1")


def write_peaks(path, timestamps, counts, peaks):
    """Write a Peaks export: header, column row, then one row per sample."""
    header = ["Culture: pt-PT ", "Module Type: Hyperion",
              "Timestamp\t" + "\t".join(f"# CH {i+1}" for i in range(len(counts)))]
    lines = [str(len(header) + 1)] + header
    for stamp, row in zip(timestamps, peaks):
        lines.append(
            stamp.strftime("%d/%m/%Y %H:%M:%S.%f")[:-1]
            + "\t" + "\t".join(str(c) for c in counts)
            + "\t" + _comma(row, "%.5f")
        )
    path.write_text("\n".join(lines), encoding="latin-1")


def _synthetic_blocks(n_frames=3, n_points=4096):
    wl = synth.wavelength_axis(n_points=n_points)
    rng = np.random.default_rng(1)
    blocks = np.stack(
        [
            np.stack(
                [
                    synth.fp_spectrum(wl, 87_000.0 + 10.0 * i),
                    synth.fbg_spectrum(wl, [1470.5, 1480.2]),
                    rng.normal(-60.0, 0.2, n_points),
                    rng.normal(-60.0, 0.2, n_points),
                ]
            )
            for i in range(n_frames)
        ]
    )
    return wl, blocks


def test_responses_round_trip(tmp_path):
    wl, blocks = _synthetic_blocks()
    stamps = [datetime(2026, 1, 5, 12, 0, 0) + timedelta(seconds=20 * i) for i in range(3)]
    path = tmp_path / "Responses.synthetic.txt"
    write_responses(path, stamps, blocks)

    result = io.read_responses(path)

    np.testing.assert_allclose(result.wavelength_nm, wl, atol=1e-9)
    assert result.timestamps == stamps
    assert result.spectra_db.shape == blocks.shape
    # Values survive the comma-decimal 2-decimal format.
    np.testing.assert_allclose(result.spectra_db, np.round(blocks, 2), atol=1e-9)


def test_responses_channel_selection(tmp_path):
    _, blocks = _synthetic_blocks()
    stamps = [datetime(2026, 1, 5) + timedelta(seconds=20 * i) for i in range(3)]
    path = tmp_path / "Responses.synthetic.txt"
    write_responses(path, stamps, blocks)

    result = io.read_responses(path, channel=1)
    assert result.spectra_db.shape == (3, blocks.shape[2])
    np.testing.assert_allclose(result.spectra_db, np.round(blocks[:, 0], 2), atol=1e-9)


def test_demodulation_survives_the_export_format(tmp_path):
    # Writing to the instrument format and reading back must not change the
    # demodulated trajectory beyond the 0.01 dB quantization of the export.
    wl, blocks = _synthetic_blocks(n_frames=4, n_points=8192)
    stamps = [datetime(2026, 1, 5) + timedelta(seconds=20 * i) for i in range(4)]
    path = tmp_path / "Responses.synthetic.txt"
    write_responses(path, stamps, blocks)

    band = (0.030, 0.042)
    direct = track.track_fp(blocks[:, 0], wl, band, 1470.0)
    loaded = io.read_responses(path, channel=1)
    from_file = track.track_fp(loaded.spectra_db, loaded.wavelength_nm, band, 1470.0)

    np.testing.assert_allclose(from_file.crest_nm, direct.crest_nm, atol=0.001)


def test_peaks_round_trip(tmp_path):
    stamps = [datetime(2026, 1, 5) + timedelta(seconds=2 * i) for i in range(5)]
    counts = (1, 1, 2, 3)
    rng = np.random.default_rng(2)
    peaks = 1525.0 + rng.random((5, 7))
    path = tmp_path / "Peaks.synthetic.txt"
    write_peaks(path, stamps, counts, peaks)

    result = io.read_peaks(path)

    assert result.timestamps == stamps
    assert result.counts == counts
    assert [c.shape for c in result.channels] == [(5, 1), (5, 1), (5, 2), (5, 3)]
    np.testing.assert_allclose(np.hstack(result.channels), np.round(peaks, 5), atol=1e-12)


def test_peaks_skips_malformed_rows(tmp_path):
    stamps = [datetime(2026, 1, 5) + timedelta(seconds=2 * i) for i in range(3)]
    counts = (1, 1, 2, 3)
    peaks = np.full((3, 7), 1525.0)
    path = tmp_path / "Peaks.synthetic.txt"
    write_peaks(path, stamps, counts, peaks)
    with open(path, "a", encoding="latin-1") as f:
        f.write("\ntruncated line without fields")

    result = io.read_peaks(path)
    assert len(result.timestamps) == 3


def test_channel_selection_rejects_non_positive_indices(tmp_path):
    # channel is 1-based and physical; 0 or negative would silently wrap
    # to another channel's data via Python indexing.
    _, blocks = _synthetic_blocks()
    stamps = [datetime(2026, 1, 5) + timedelta(seconds=20 * i) for i in range(3)]
    path = tmp_path / "Responses.synthetic.txt"
    write_responses(path, stamps, blocks)
    with pytest.raises(ValueError, match="channel"):
        io.read_responses(path, channel=0)
    with pytest.raises(ValueError, match="channel"):
        io.read_responses(path, channel=5)
