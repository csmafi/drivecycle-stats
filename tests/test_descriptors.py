import numpy as np
import pandas as pd
import pytest

from drivecycle_stats.descriptors import descriptors_table, trip_descriptors


def test_hand_computed_five_second_trace():
    # 5 seconds: speeds 0, 10, 20, 20, 0 km/h (idle at start and end)
    speed = [0.0, 10.0, 20.0, 20.0, 0.0]
    accel = [0.0, 2.0, 2.0, 0.0, -2.0]
    frame = pd.DataFrame({"Speed": speed, "Acceleration": accel})

    d = trip_descriptors(frame)

    assert d["duration_s"] == 5
    assert d["avgspd"] == pytest.approx(np.mean(speed))
    # moving seconds: indices 1,2,3 (speed > 1.0 km/h threshold)
    assert d["runspd"] == pytest.approx(np.mean([10.0, 20.0, 20.0]))
    # idle: seconds with speed <= 1.0 km/h -> indices 0, 4
    assert d["idle_share"] == pytest.approx(2 / 5)
    # two separate idle runs (start, end) -> 2 stops, each length 1
    assert d["n_stops"] == 2
    assert d["mean_stop_duration_s"] == pytest.approx(1.0)
    # accel > 0.1 m/s^2 at indices 1, 2 -> mean of [2.0, 2.0]
    assert d["avgposacc"] == pytest.approx(2.0)
    assert d["rmsa"] == pytest.approx(np.sqrt(np.mean(np.array(accel) ** 2)))
    assert d["v95"] == pytest.approx(np.percentile(speed, 95))
    # distance: sum(speed)/3600
    assert d["distance_km"] == pytest.approx(sum(speed) / 3600.0)


def test_no_stops():
    frame = pd.DataFrame({"Speed": [10.0, 15.0, 20.0], "Acceleration": [1.0, 1.0, 0.0]})
    d = trip_descriptors(frame)
    assert d["n_stops"] == 0
    assert np.isnan(d["mean_stop_duration_s"])
    assert d["idle_share"] == 0.0


def test_entirely_idle():
    frame = pd.DataFrame({"Speed": [0.0, 0.5, 0.0], "Acceleration": [0.0, 0.0, 0.0]})
    d = trip_descriptors(frame)
    assert d["idle_share"] == 1.0
    assert d["n_stops"] == 1
    assert d["mean_stop_duration_s"] == pytest.approx(3.0)
    assert np.isnan(d["runspd"])


def test_tripkm_column_used_when_present():
    frame = pd.DataFrame({
        "Speed": [10.0, 10.0, 10.0],
        "Acceleration": [0.0, 0.0, 0.0],
        "TripKm": [0.0, 0.003, 0.006],
    })
    d = trip_descriptors(frame)
    assert d["distance_km"] == pytest.approx(0.006)


def test_missing_columns_raise():
    with pytest.raises(ValueError):
        trip_descriptors(pd.DataFrame({"Speed": [1.0, 2.0]}))
    with pytest.raises(ValueError):
        trip_descriptors(pd.DataFrame({"Acceleration": [1.0, 2.0]}))


def test_empty_frame_raises():
    with pytest.raises(ValueError):
        trip_descriptors(pd.DataFrame({"Speed": [], "Acceleration": []}))


def test_descriptors_table():
    f1 = pd.DataFrame({"Speed": [0.0, 10.0], "Acceleration": [0.0, 1.0]})
    f2 = pd.DataFrame({"Speed": [20.0, 20.0], "Acceleration": [0.0, 0.0]})
    table = descriptors_table([f1, f2], ids=["trip_a", "trip_b"])
    assert list(table.index) == ["trip_a", "trip_b"]
    assert len(table) == 2


def test_descriptors_table_id_length_mismatch_raises():
    f1 = pd.DataFrame({"Speed": [0.0, 10.0], "Acceleration": [0.0, 1.0]})
    with pytest.raises(ValueError):
        descriptors_table([f1], ids=["a", "b"])
