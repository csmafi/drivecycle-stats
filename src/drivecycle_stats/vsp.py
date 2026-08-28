"""Vehicle-specific power (VSP).

Reference: Jimenez-Palacios, J.L. (1999). "Understanding and Quantifying
Motor Vehicle Emissions with Vehicle Specific Power and TILDAS Remote
Sensing." PhD thesis, Massachusetts Institute of Technology.

The coefficients used here are the commonly cited light-duty-vehicle form
of the VSP equation, expressed in W/kg. They assume a generic light-duty
vehicle mass-to-drag ratio; they are not refit to any particular vehicle
in this package. Road grade is assumed to be zero. This is a limitation
of the calculation, not a result: on a graded road, VSP computed this way
will be biased low on downhill sections and biased high on uphill
sections.
"""

from __future__ import annotations

import numpy as np


def vsp(speed_kmh, accel_ms2):
    """Compute vehicle-specific power in W/kg.

    Parameters
    ----------
    speed_kmh : array-like
        Vehicle speed in kilometres per hour.
    accel_ms2 : array-like
        Longitudinal acceleration in metres per second squared, aligned
        one-to-one with ``speed_kmh``.

    Returns
    -------
    numpy.ndarray
        VSP in watts per kilogram, same shape as the inputs.

    Notes
    -----
    Formula (light-duty vehicle form, Jimenez-Palacios 1999)::

        u = speed_kmh / 3.6                     # km/h -> m/s
        vsp = u * (1.1 * a + 0.132) + 0.000302 * u**3

    Zero speed gives zero VSP regardless of acceleration, since every
    term in the formula is a function of speed.

    Road grade is assumed zero. This is a limitation: results on graded
    roads will be biased.
    """
    speed_kmh = np.asarray(speed_kmh, dtype=float)
    accel_ms2 = np.asarray(accel_ms2, dtype=float)
    if speed_kmh.shape != accel_ms2.shape:
        raise ValueError(
            f"speed_kmh and accel_ms2 must have the same shape, "
            f"got {speed_kmh.shape} and {accel_ms2.shape}"
        )

    u = speed_kmh / 3.6
    return u * (1.1 * accel_ms2 + 0.132) + 0.000302 * u**3
