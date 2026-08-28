import numpy as np
import pandas as pd
import pytest

from drivecycle_stats.io_ved import clean_trip, iter_ved_trips


def _raw_trip(timestamps_ms, speeds):
    return pd.DataFrame(
        {
            "VehId": [1] * len(timestamps_ms),
            "Trip": [10] * len(timestamps_ms),
            "Timestamp(ms)": timestamps_ms,
            "Vehicle Speed[km/h]": speeds,
        }
    )


def test_clean_trip_resamples_irregular_timestamps():
    # irregular gaps: 0, 300, 1000, 1900, 3000 ms -> resample to 1 Hz grid
    raw = _raw_trip([0, 300, 1000, 1900, 3000], [0, 10, 20, 30, 40])
    frame, report = clean_trip(raw, min_duration_s=1)

    assert frame is not None
    assert list(frame.columns) == ["Speed", "Acceleration"]
    # duration 3.0s -> grid 0,1,2,3 -> 4 rows
    assert report["n_resampled_rows"] == 4
    assert report["n_dropped_missing_speed"] == 0


def test_clean_trip_drops_missing_speed():
    raw = _raw_trip([0, 1000, 2000, 3000], [0, np.nan, 20, 30])
    frame, report = clean_trip(raw, min_duration_s=1)
    assert report["n_dropped_missing_speed"] == 1
    assert frame is not None


def test_clean_trip_drops_short_trip():
    raw = _raw_trip([0, 500], [0, 5])
    frame, report = clean_trip(raw, min_duration_s=10)
    assert frame is None
    assert report["dropped_reason"] is not None


def test_clean_trip_clips_acceleration():
    # huge speed jump -> unrealistic acceleration -> must be clipped
    raw = _raw_trip([0, 1000, 2000], [0, 200, 0])
    frame, report = clean_trip(raw, min_duration_s=1, max_accel_ms2=5.0)
    assert frame["Acceleration"].abs().max() <= 5.0
    assert report["n_accel_clipped"] > 0


def test_clean_trip_missing_columns_raises_gracefully():
    raw = pd.DataFrame({"foo": [1, 2, 3]})
    frame, report = clean_trip(raw)
    assert frame is None
    assert report["dropped_reason"] == "missing required column"


def test_iter_ved_trips_groups_correctly():
    raw = pd.DataFrame(
        {
            "VehId": [1, 1, 2, 2],
            "Trip": [10, 10, 20, 20],
            "Timestamp(ms)": [1000, 0, 0, 1000],
            "Vehicle Speed[km/h]": [5, 0, 10, 15],
        }
    )
    groups = list(iter_ved_trips(raw))
    assert len(groups) == 2
    veh_ids = {g[0] for g in groups}
    assert veh_ids == {1, 2}
    # check sorting by timestamp within a group
    for veh_id, trip_id, frame in groups:
        assert frame["Timestamp(ms)"].is_monotonic_increasing


def test_iter_ved_trips_missing_columns_raises():
    with pytest.raises(ValueError):
        list(iter_ved_trips(pd.DataFrame({"foo": [1, 2]})))
