"""Spectral shifts to physical quantities: temperature and pressure.

Ports the calibration of the original battery-cell instrumentation. An FBG
converts to absolute temperature through its reference wavelength and
sensitivity. Pressure and temperature are separated with the matrix method:
the FBG and FPI shifts respond to (P, T) through four sensitivities,

    [ delta_fbg ]   [ k_fbg_P   k_fbg_T ] [ P ]
    [ delta_fpi ] = [ k_fpi_P   k_fpi_T ] [ T ]

and the 2x2 system is solved for (P, T). Defaults are the original
calibration: the FBG sees only temperature; the FPI sees pressure with a
temperature cross-sensitivity. Both shift inputs must share the same
reference frame (e.g. the first frame), so P and T start at zero by
construction — the original script mixed an absolute FBG reference with a
first-sample FPI reference and inherited a spurious pressure offset.
"""

import numpy as np

# Original calibration sensitivities.
FBG_TEMPERATURE_NM_PER_C = 0.010215
FPI_PRESSURE_NM_PER_BAR = -1.5648
FPI_TEMPERATURE_NM_PER_C = -0.045412


def fbg_temperature(wavelength_nm, reference_nm, sensitivity_nm_per_c=FBG_TEMPERATURE_NM_PER_C):
    """Absolute FBG temperature in degC: (wavelength - reference)/sensitivity.

    Broadcasts: pass an (n_frames, n_fbg) wavelength array with per-sensor
    references and sensitivities to convert a whole tracked sequence.
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)
    return (wavelength_nm - reference_nm) / sensitivity_nm_per_c


def separate_pressure_temperature(
    delta_fbg_nm,
    delta_fpi_nm,
    *,
    fbg_pressure_nm_per_bar=0.0,
    fbg_temperature_nm_per_c=FBG_TEMPERATURE_NM_PER_C,
    fpi_pressure_nm_per_bar=FPI_PRESSURE_NM_PER_BAR,
    fpi_temperature_nm_per_c=FPI_TEMPERATURE_NM_PER_C,
):
    """Solve the 2x2 sensitivity system for pressure (bar) and temperature (degC).

    ``delta_fbg_nm`` and ``delta_fpi_nm`` are wavelength shifts from the same
    reference frame and must share one shape. The four keyword sensitivities
    are the entries of the matrix above; override any of them to match
    another sensor pair. Returns ``(pressure, temperature)`` with the shape
    of the inputs (0-d arrays for scalar inputs).
    """
    delta_fbg_nm = np.asarray(delta_fbg_nm, dtype=float)
    delta_fpi_nm = np.asarray(delta_fpi_nm, dtype=float)
    if delta_fbg_nm.shape != delta_fpi_nm.shape:
        raise ValueError(
            f"delta_fbg_nm shape {delta_fbg_nm.shape} != "
            f"delta_fpi_nm shape {delta_fpi_nm.shape}"
        )

    matrix = np.array(
        [
            [fbg_pressure_nm_per_bar, fbg_temperature_nm_per_c],
            [fpi_pressure_nm_per_bar, fpi_temperature_nm_per_c],
        ]
    )
    if abs(np.linalg.det(matrix)) < 1e-9 * np.abs(matrix).max() ** 2:
        raise ValueError(
            "sensitivity matrix is singular: the FBG and FPI responses are "
            "proportional, so pressure and temperature cannot be separated"
        )

    shifts = np.stack([np.ravel(delta_fbg_nm), np.ravel(delta_fpi_nm)])
    pressure, temperature = np.linalg.solve(matrix, shifts)
    return (
        pressure.reshape(delta_fbg_nm.shape),
        temperature.reshape(delta_fbg_nm.shape),
    )
