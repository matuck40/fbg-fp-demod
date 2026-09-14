"""Tests for reading interrogator export files.

No measured data: the tests WRITE synthetic files in the instrument's own
export format (pt-PT decimal commas, header, timestamp blocks) and read
them back, so the reader is exercised against the format, not against any
recorded content.
"""

import gzip
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from fbgfp import io, synth, track


def _comma(values, fmt="%.2f"):
    return "\t".join((fmt % v).replace(".", ",") for v in values)


def write_responses(path, timestamps, blocks, start_nm=1460.0, step_nm=0.008,
                    culture="pt-PT"):
    """Write an ENLIGHT-style Responses export: header, then blocks of
    [timestamp, one line per channel], blank line between blocks."""
    n_points = blocks.shape[2]
    order = "%m/%d/%Y" if culture == "en-US" else "%d/%m/%Y"
    header = [
        f"Culture: {culture} ",
        f"Date: {timestamps[0].strftime(order + ' %H:%M:%S.%f')[:-1]}",
        "Module Type: Hyperion",
        f"Wavelength Start (nm): {f'{start_nm:.5f}'.replace('.', ',')}",
        f"Wavelength Delta (nm): {f'{step_nm:.4f}'.replace('.', ',')}",
        f"Number of Points: {n_points}",
        "",
    ]
    lines = [str(len(header) + 1)] + header
    for stamp, block in zip(timestamps, blocks, strict=False):
        lines.append(stamp.strftime(order + " %H:%M:%S.%f")[:-1])
        for channel in block:
            lines.append(_comma(channel))
        lines.append("")
    path.write_text("\n".join(lines), encoding="latin-1")


def write_peaks(path, timestamps, counts, peaks):
    """Write a Peaks export: header, column row, then one row per sample."""
    header = ["Culture: pt-PT ", "Module Type: Hyperion",
              "Timestamp\t" + "\t".join(f"# CH {i+1}" for i in range(len(counts)))]
    lines = [str(len(header) + 1)] + header
    for stamp, row in zip(timestamps, peaks, strict=False):
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


def test_read_responses_can_cap_the_number_of_spectra(tmp_path):
    _, blocks = _synthetic_blocks(n_frames=3)
    stamps = [datetime(2026, 1, 5) + timedelta(seconds=20 * i) for i in range(3)]
    path = tmp_path / "Responses.synthetic.txt"
    write_responses(path, stamps, blocks)

    result = io.read_responses(path, channel=1, max_spectra=2)
    assert result.spectra_db.shape[0] == 2
    assert result.timestamps == stamps[:2]


def test_a_gzipped_export_reads_back_identically(tmp_path):
    """Exports compress by roughly 7x; the reader should not care.

    These files are text columns of numbers, so gzip takes a multi-gigabyte
    export down to a fraction of its size — and takes a sample small enough
    to ship with the source. Reading must give the same arrays either way.
    """
    stamps = [datetime(2026, 3, 24, 9, 39) + timedelta(seconds=20 * i) for i in range(3)]
    wl = synth.wavelength_axis(n_points=4096)
    blocks = np.stack(
        [np.stack([synth.fp_spectrum(wl, 87_000.0 + 12.0 * i)] * 4) for i in range(3)]
    )
    plain = tmp_path / "Responses.plain.txt"
    write_responses(plain, stamps, blocks)

    packed = tmp_path / "Responses.packed.txt.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes(), 9))

    from_plain = io.read_responses(plain, channel=2)
    from_packed = io.read_responses(packed, channel=2)

    np.testing.assert_allclose(from_packed.wavelength_nm, from_plain.wavelength_nm)
    np.testing.assert_allclose(from_packed.spectra_db, from_plain.spectra_db)
    assert from_packed.timestamps == from_plain.timestamps
    assert packed.stat().st_size < 0.5 * plain.stat().st_size


def test_a_gzipped_peaks_export_reads_back_identically(tmp_path):
    stamps = [datetime(2026, 3, 24, 9, 39) + timedelta(seconds=2 * i) for i in range(5)]
    counts = (1, 1, 2, 3)
    values = 1525.0 + np.random.default_rng(2).random((5, 7))
    plain = tmp_path / "Peaks.plain.txt"
    write_peaks(plain, stamps, counts, values)
    packed = tmp_path / "Peaks.packed.txt.gz"
    packed.write_bytes(gzip.compress(plain.read_bytes(), 9))

    from_plain, from_packed = io.read_peaks(plain), io.read_peaks(packed)
    assert from_packed.timestamps == from_plain.timestamps
    assert from_packed.counts == from_plain.counts
    for packed_channel, plain_channel in zip(
        from_packed.channels, from_plain.channels, strict=True
    ):
        np.testing.assert_allclose(packed_channel, plain_channel)


def test_an_en_us_export_is_not_read_with_the_day_and_month_swapped():
    """The instrument writes dates in whatever locale it was set to.

    ``9/7/2026`` is 7 September in the en-US export the instrument writes
    with ``Culture: en-US``, and 9 July if read day-first. The two orders
    are indistinguishable for the first twelve days of any month, so
    guessing wrong does not fail — it silently returns the wrong date,
    and any alignment against another recording quietly slides by months.
    """
    from tempfile import TemporaryDirectory

    stamps = [datetime(2026, 9, 7, 11, 27, 3) + timedelta(seconds=20 * i)
              for i in range(2)]
    _, blocks = _synthetic_blocks(n_frames=2, n_points=512)
    with TemporaryDirectory() as folder:
        path = Path(folder) / "Responses.en-US.txt"
        write_responses(path, stamps, blocks, culture="en-US")
        recovered = io.read_responses(path, channel=1)
    assert recovered.timestamps == stamps
    assert recovered.timestamps[0].month == 9, "read as July: day/month swapped"


def test_a_pt_pt_export_still_reads_day_first():
    from tempfile import TemporaryDirectory

    stamps = [datetime(2026, 9, 7, 11, 27, 3) + timedelta(seconds=20 * i)
              for i in range(2)]
    _, blocks = _synthetic_blocks(n_frames=2, n_points=512)
    with TemporaryDirectory() as folder:
        path = Path(folder) / "Responses.pt-PT.txt"
        write_responses(path, stamps, blocks, culture="pt-PT")
        recovered = io.read_responses(path, channel=1)
    assert recovered.timestamps == stamps
