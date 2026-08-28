import numpy as np
import pytest

from drivecycle_stats.outliers import flag_low_density


def test_isolated_point_flagged():
    rng = np.random.default_rng(0)
    cloud = rng.normal(0, 1, size=(200, 2))
    isolated = np.array([[50.0, 50.0]])
    xy = np.vstack([cloud, isolated])

    flags = flag_low_density(xy, pctl=2.0)

    assert flags[-1] == True
    # Not everything should be flagged.
    assert flags.sum() < len(xy)


def test_pctl_controls_flagged_fraction():
    rng = np.random.default_rng(1)
    xy = rng.normal(0, 1, size=(500, 2))

    flags_small = flag_low_density(xy, pctl=1.0)
    flags_large = flag_low_density(xy, pctl=10.0)

    assert flags_large.sum() >= flags_small.sum()


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        flag_low_density(np.array([1.0, 2.0, 3.0]))  # 1-D
    with pytest.raises(ValueError):
        flag_low_density(np.zeros((5, 3)))  # wrong number of columns
    with pytest.raises(ValueError):
        flag_low_density(np.zeros((2, 2)))  # too few points


def test_bad_pctl_raises():
    xy = np.random.default_rng(2).normal(0, 1, size=(20, 2))
    with pytest.raises(ValueError):
        flag_low_density(xy, pctl=0)
    with pytest.raises(ValueError):
        flag_low_density(xy, pctl=100)
    with pytest.raises(ValueError):
        flag_low_density(xy, pctl=-5)
