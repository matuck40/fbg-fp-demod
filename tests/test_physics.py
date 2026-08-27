"""Tests for the conversion from spectral shifts to physical quantities.

The sensitivities come from the original calibration: the FBG sees only
temperature (10.215 pm/degC), the FPI sees pressure (-1.5648 nm/bar) with a
temperature cross-sensitivity (-45.412 pm/degC). The matrix method inverts
the 2x2 system; both inputs are shifts from the SAME reference frame, so
pressure and temperature both start at zero by construction (the original
script mixed references and started at -0.7 to -0.9 bar).
"""

import numpy as np
import pytest

from fbgfp import physics, synth, track


def test_fbg_temperature_converts_a_known_shift():
    # 0.30645 nm above the reference at 10.215 pm/degC is exactly 30 degC.
    temperature = physics.fbg_temperature(1531.30645, 1531.0)
    assert temperature == pytest.approx(30.0, abs=1e-9)


def test_fbg_temperature_broadcasts_over_sensors():
    references = np.array([1531.0, 1526.0])
    sensitivities = np.array([0.010215, 0.009])
    wavelengths = references + np.array([[0.30645, 0.225]])  # 30 and 25 degC
    temperature = physics.fbg_temperature(wavelengths, references, sensitivities)
    assert temperature.shape == (1, 2)
    assert temperature[0, 0] == pytest.approx(30.0, abs=1e-9)
    assert temperature[0, 1] == pytest.approx(25.0, abs=1e-9)


def test_separation_round_trips_exactly():
    pressure_true = np.array([0.0, 0.5, 1.5, 2.0])
    temperature_true = np.array([0.0, 0.8, 1.6, 2.4])

    delta_fbg = 0.010215 * temperature_true
    delta_fpi = -1.5648 * pressure_true - 0.045412 * temperature_true

    pressure, temperature = physics.separate_pressure_temperature(delta_fbg, delta_fpi)
    np.testing.assert_allclose(pressure, pressure_true, atol=1e-12)
    np.testing.assert_allclose(temperature, temperature_true, atol=1e-12)


def test_zero_shift_means_zero_pressure_and_temperature():
    pressure, temperature = physics.separate_pressure_temperature(0.0, 0.0)
    assert pressure == 0.0
    assert temperature == 0.0


def test_custom_sensitivities_round_trip():
    pressure, temperature = physics.separate_pressure_temperature(
        delta_fbg_nm=0.012 * 3.0 + 0.001 * 1.5,  # an FBG that also sees pressure
        delta_fpi_nm=-2.0 * 1.5 - 0.05 * 3.0,
        fbg_pressure_nm_per_bar=0.001,
        fbg_temperature_nm_per_c=0.012,
        fpi_pressure_nm_per_bar=-2.0,
        fpi_temperature_nm_per_c=-0.05,
    )
    assert pressure == pytest.approx(1.5)
    assert temperature == pytest.approx(3.0)


def test_singular_sensitivity_matrix_is_rejected():
    with pytest.raises(ValueError, match="singular"):
        physics.separate_pressure_temperature(
            0.1,
            0.1,
            fbg_pressure_nm_per_bar=1.0,
            fbg_temperature_nm_per_c=1.0,
            fpi_pressure_nm_per_bar=2.0,
            fpi_temperature_nm_per_c=2.0,
        )


def test_end_to_end_recovers_pressure_and_temperature():
    # Full chain: known P(t), T(t) -> spectral shifts via the sensitivities
    # -> synthetic spectra -> tracking -> conversion. The declared tolerance
    # inherits the tracker contract: the ~5% shift compression maps onto the
    # pressure scale, and FBG jitter (~5 pm) maps onto ~0.5 degC.
    opd0 = 87_000.0
    crest0 = opd0 / 55.0
    fbg0 = 1525.4
    n_frames = 25

    pressure_true = np.linspace(0.0, 2.0, n_frames)  # bar
    temperature_true = np.linspace(0.0, 2.0, n_frames)  # degC above start

    delta_fpi = -1.5648 * pressure_true - 0.045412 * temperature_true
    delta_fbg = 0.010215 * temperature_true

    wl = synth.wavelength_axis()
    seq = synth.simulate_sequence(
        wl,
        opd_nm=opd0 * (1.0 + delta_fpi / crest0),
        fbg_centers_nm=(fbg0 + delta_fbg)[:, np.newaxis],
        noise_db=0.2,
        seed=21,
    )

    fp = track.track_fp(seq.spectra_db, wl, (0.030, 0.042), 1560.0)
    fbg = track.track_fbg(seq.spectra_db, wl, [fbg0])

    pressure, temperature = physics.separate_pressure_temperature(
        fbg[:, 0] - fbg[0, 0], fp.corrected_nm - fp.corrected_nm[0]
    )

    assert np.abs(pressure - pressure_true).max() < 0.06 * 2.0 + 0.05  # bar
    assert np.abs(temperature - temperature_true).max() < 1.0  # degC
