"""Low-density outlier flag.

A general-purpose utility: fit a Gaussian kernel density estimate on a
two-column array and flag the points that fall in its lowest-density
region. This is a standard idea (points in sparse regions of a fitted
density are treated as candidate outliers); only the implementation
here is specific to this package.

The percentile used to decide "how sparse counts as low density" is a
tuning constant. It has no theoretical justification -- it is a choice,
not a derived quantity. Any analysis that uses this flag should be
repeated at a different percentile to check whether the conclusion
depends on the choice.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def flag_low_density(xy, pctl: float = 2.0) -> np.ndarray:
    """Flag points in the lowest-density region of a 2-D point cloud.

    Parameters
    ----------
    xy : array-like, shape (n, 2)
        Two-column array of points, e.g. a descriptor pair such as
        (average speed, idle share) for a set of trips.
    pctl : float, default 2.0
        Percentage of points to flag as low-density, e.g. 2.0 flags
        the bottom 2% by estimated density. This is a tuning constant
        with no theoretical basis; see the module docstring.

    Returns
    -------
    numpy.ndarray of bool, shape (n,)
        True where the point is flagged as low-density.

    Raises
    ------
    ValueError
        If ``xy`` is not shape (n, 2) with n >= 3, or if ``pctl`` is
        not in (0, 100).
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape (n, 2), got {xy.shape}")
    if xy.shape[0] < 3:
        raise ValueError("xy must have at least 3 points to fit a KDE")
    if not (0 < pctl < 100):
        raise ValueError("pctl must be in (0, 100)")

    kde = gaussian_kde(xy.T)
    density = kde(xy.T)

    z = (density - density.mean()) / density.std(ddof=0)

    threshold = np.percentile(z, pctl)
    return z <= threshold
