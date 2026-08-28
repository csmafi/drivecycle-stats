import numpy as np
import pytest

from drivecycle_stats.compare import (
    energy_distance,
    energy_test,
    tost,
    tost_variance_ratio,
)


def test_energy_distance_zero_for_identical_samples():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert energy_distance(x, x.copy()) == pytest.approx(0.0, abs=1e-10)


def test_energy_distance_positive_for_different_samples():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    assert energy_distance(x, y) > 0


def test_energy_distance_symmetric():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 20)
    y = rng.normal(1, 2, 25)
    assert energy_distance(x, y) == pytest.approx(energy_distance(y, x), rel=1e-10)


def test_energy_distance_empty_raises():
    with pytest.raises(ValueError):
        energy_distance([], [1.0, 2.0])


def test_energy_test_p_value_near_uniform_same_distribution():
    rng = np.random.default_rng(42)
    # Both samples from the same distribution: p-value should not be
    # small. We can't assert exact uniformity from one draw, but we can
    # assert it's not falsely significant.
    x = rng.normal(0, 1, 30)
    y = rng.normal(0, 1, 30)
    result = energy_test(x, y, n_perm=500, seed=1)
    assert result["p_value"] > 0.05
    assert len(result["null_distribution"]) == 500
    assert result["statistic"] >= 0


def test_energy_test_detects_different_distributions():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 30)
    y = rng.normal(5, 1, 30)
    result = energy_test(x, y, n_perm=500, seed=2)
    assert result["p_value"] < 0.05


def test_energy_test_resolution_limit():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([100.0, 101.0, 102.0])
    result = energy_test(x, y, n_perm=100, seed=0)
    # smallest possible p-value with n_perm=100 is 1/101
    assert result["p_value"] >= 1 / 101


def test_tost_rejects_equivalence_for_different_means():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 50)
    y = rng.normal(10, 1, 50)
    result = tost(x, y, bound=1.0)
    assert result["equivalent"] is False


def test_tost_accepts_equivalence_for_near_identical_means():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.1, 200)
    y = rng.normal(0, 0.1, 200)
    result = tost(x, y, bound=1.0)
    assert result["equivalent"] is True


def test_tost_bad_input_raises():
    with pytest.raises(ValueError):
        tost([1.0], [1.0, 2.0], bound=1.0)
    with pytest.raises(ValueError):
        tost([1.0, 2.0], [1.0, 2.0], bound=-1.0)
    with pytest.raises(ValueError):
        tost([1.0, 2.0], [1.0, 2.0], bound=1.0, alpha=1.5)


def test_tost_variance_ratio_detects_different_variance():
    rng = np.random.default_rng(5)
    x = rng.normal(0, 1, 100)
    y = rng.normal(0, 5, 100)
    result = tost_variance_ratio(x, y, bound_low=0.8, bound_high=1.25)
    assert result["equivalent"] is False


def test_tost_variance_ratio_accepts_similar_variance():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 1, 500)
    y = rng.normal(0, 1, 500)
    result = tost_variance_ratio(x, y, bound_low=0.5, bound_high=2.0)
    assert result["equivalent"] is True


def test_tost_variance_ratio_bad_bounds_raise():
    with pytest.raises(ValueError):
        tost_variance_ratio([1.0, 2.0], [1.0, 2.0], bound_low=2.0, bound_high=1.0)
