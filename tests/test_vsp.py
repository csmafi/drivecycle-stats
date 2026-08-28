import numpy as np
import pytest

from drivecycle_stats.vsp import vsp


def test_zero_speed_gives_zero_vsp():
    speed = np.array([0.0, 0.0, 0.0])
    accel = np.array([0.0, 2.0, -3.0])
    result = vsp(speed, accel)
    assert np.allclose(result, 0.0)


def test_hand_computed_value():
    # speed = 36 km/h -> u = 10 m/s ; accel = 1.0 m/s^2
    # vsp = 10 * (1.1*1.0 + 0.132) + 0.000302 * 10**3
    #     = 10 * 1.232 + 0.302 = 12.32 + 0.302 = 12.622
    speed = np.array([36.0])
    accel = np.array([1.0])
    result = vsp(speed, accel)
    assert result[0] == pytest.approx(12.622, abs=1e-6)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        vsp(np.array([1.0, 2.0]), np.array([1.0]))
