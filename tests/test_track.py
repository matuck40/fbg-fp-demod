"""Tests for trajectory tracking across a sequence of spectra.

Ground truth comes from the generator's OPD and FBG trajectories. The FP
tolerances inherit the single-spectrum contract (shifts compressed ~5% by
the two-bin reconstruction, <10 pm jitter at 0.2 dB noise); the fringe-hop
cases exercise the recovery that the original MATLAB left commented out.
"""

import numpy as np
import pytest

from fbgfp import synth, track

OPD_NM = 87_000.0
BAND = (0.030, 0.042)
REFERENCE_NM = 1560.0


def test_constant_cavity_gives_a_constant_reading():
    wl = synth.wavelength_axis()
    seq = synth.simulate_sequence(wl, opd_nm=np.full(4, OPD_NM))
    result = track.track_fp(seq.spectra_db, wl, BAND, REFERENCE_NM)

    assert result.crest_nm.shape == (4,)
    assert np.ptp(result.crest_nm) < 1e-9  # identical frames, identical reading
    assert np.ptp(result.valley_nm) < 1e-9
    assert result.hop_frames == ()
    np.testing.assert_array_equal(result.corrected_nm, result.crest_nm)


def _sinusoidal_opd(n_frames):
    return OPD_NM + 20.0 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_frames))


def test_recovers_a_sinusoidal_trajectory_under_noise():
    wl = synth.wavelength_axis()
    opd = _sinusoidal_opd(30)
    seq = synth.simulate_sequence(wl, opd_nm=opd, noise_db=0.2, seed=5)
    result = track.track_fp(seq.spectra_db, wl, BAND, REFERENCE_NM)

    assert result.hop_frames == ()
    recovered = result.corrected_nm - result.corrected_nm[0]
    predicted = result.corrected_nm[0] * (opd - opd[0]) / OPD_NM
    errors = np.abs(recovered - predicted)
    # Declared contract: 6% method compression plus 10 pm of noise jitter.
    assert np.all(errors <= 0.06 * np.abs(predicted) + 0.010)


def test_baseline_drift_bias_stays_within_the_declared_bound():
    # A 0.5 dB baseline drift whose phase sweeps the sequence is NOT free:
    # in the linear domain it is a moving envelope whose sidebands
    # (0.031 +/- 0.005 cycles/nm) fall inside the pass band and phase-
    # modulate the filtered fringe. Measured systematic error: ~200 pm.
    # The original MATLAB pipeline shares this mechanism; the declared
    # bound makes it visible instead of hiding it.
    wl = synth.wavelength_axis()
    opd = _sinusoidal_opd(30)
    seq = synth.simulate_sequence(
        wl, opd_nm=opd, noise_db=0.2, drift_amplitude_db=0.5, seed=5
    )
    result = track.track_fp(seq.spectra_db, wl, BAND, REFERENCE_NM)

    assert result.hop_frames == ()
    recovered = result.corrected_nm - result.corrected_nm[0]
    predicted = result.corrected_nm[0] * (opd - opd[0]) / OPD_NM
    errors = np.abs(recovered - predicted)
    assert np.all(errors <= 0.06 * np.abs(predicted) + 0.250)


def test_hands_over_when_the_fringe_exits_the_right_edge():
    wl = synth.wavelength_axis()
    n_frames = 80
    d_opd = 2_500.0  # pushes the tracked crest itself past 1620 nm
    opd = np.linspace(OPD_NM, OPD_NM + d_opd, n_frames)
    seq = synth.simulate_sequence(wl, opd_nm=opd)
    result = track.track_fp(seq.spectra_db, wl, BAND, REFERENCE_NM)

    assert len(result.hop_frames) >= 1
    # The raw reading jumps by about one fringe at the handover...
    assert np.abs(np.diff(result.crest_nm)).max() > 10.0
    # ...and the corrected trajectory absorbs it.
    assert np.abs(np.diff(result.corrected_nm)).max() < 1.0
    total_predicted = result.corrected_nm[0] * d_opd / OPD_NM
    total_recovered = result.corrected_nm[-1] - result.corrected_nm[0]
    assert total_recovered == pytest.approx(total_predicted, rel=0.08)


def test_recovers_a_silent_hop_at_the_left_edge():
    # Track the leftmost fringe and shrink the cavity: its valley slides
    # below 1460 nm and vanishes. No exception marks this case — the
    # nearest-valley match silently lands one fringe to the right, and only
    # the unwrap keeps the trajectory continuous.
    wl = synth.wavelength_axis()
    n_frames = 30
    opd = np.linspace(OPD_NM, OPD_NM - 250.0, n_frames)
    seq = synth.simulate_sequence(wl, opd_nm=opd)
    result = track.track_fp(seq.spectra_db, wl, BAND, reference_nm=1463.0)

    assert len(result.hop_frames) >= 1
    assert np.abs(np.diff(result.corrected_nm)).max() < 1.0


def test_tracks_fbg_centers_through_a_noisy_sequence():
    wl = synth.wavelength_axis()
    n_frames = 20
    centers = np.column_stack(
        [
            np.linspace(1525.4, 1525.45, n_frames),
            np.linspace(1554.8, 1554.77, n_frames),
        ]
    )
    seq = synth.simulate_sequence(
        wl, opd_nm=np.full(n_frames, OPD_NM), fbg_centers_nm=centers,
        noise_db=0.2, seed=9,
    )
    recovered = track.track_fbg(seq.spectra_db, wl, [1525.4, 1554.8])

    assert recovered.shape == (n_frames, 2)
    assert np.abs(recovered - centers).max() < 0.010  # 10 pm measured jitter
