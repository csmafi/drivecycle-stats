import pandas as pd
import pytest

from drivecycle_stats.segment import segment_microtrips


def _frame(speeds):
    return pd.DataFrame({"Speed": speeds})


def test_three_known_stops():
    # seconds:      0  1  2  3  4  5  6  7  8  9 10
    speeds = [20, 20, 0, 0, 20, 20, 0, 0, 20, 0, 0]
    #                 stop@2        stop@6         stop@9
    frame = _frame(speeds)
    microtrips, summary = segment_microtrips(frame, min_stop_duration_s=2)

    # stops at indices 2 and 6 qualify (length 2 each); stop at 9 has
    # length 2 as well (indices 9,10) -> three stop events -> boundaries
    # 0, 2, 6, 9, 11 -> microtrips: [0:2), [2:6), [6:9), [9:11)
    assert len(microtrips) == 4
    assert summary["start_idx"].tolist() == [0, 2, 6, 9]
    assert summary["end_idx"].tolist() == [1, 5, 8, 10]


def test_threshold_parameter():
    speeds = [20, 20, 0.5, 0.5, 20, 20]
    frame = _frame(speeds)
    # with threshold 1.0, the 0.5 values count as stopped
    microtrips, _summary = segment_microtrips(
        frame, speed_threshold_kmh=1.0, min_stop_duration_s=2
    )
    assert len(microtrips) == 2

    # with threshold 0.0, the 0.5 values do NOT count as stopped -> no
    # qualifying stop -> whole trace is one microtrip
    microtrips2, _summary2 = segment_microtrips(
        frame, speed_threshold_kmh=0.0, min_stop_duration_s=2
    )
    assert len(microtrips2) == 1


def test_min_stop_duration_parameter():
    speeds = [20, 20, 0, 20, 20, 20]  # single-second stop at index 2
    frame = _frame(speeds)

    # min duration 1: the single-second stop qualifies
    mt1, _s1 = segment_microtrips(frame, min_stop_duration_s=1)
    assert len(mt1) == 2

    # min duration 2: the single-second stop does not qualify
    mt2, _s2 = segment_microtrips(frame, min_stop_duration_s=2)
    assert len(mt2) == 1


def test_no_stops():
    speeds = [10, 20, 30, 40, 50]
    frame = _frame(speeds)
    microtrips, summary = segment_microtrips(frame, min_stop_duration_s=1)
    assert len(microtrips) == 1
    assert summary["start_idx"].iloc[0] == 0
    assert summary["end_idx"].iloc[0] == 4


def test_entirely_idle():
    speeds = [0, 0, 0, 0]
    frame = _frame(speeds)
    microtrips, summary = segment_microtrips(frame, min_stop_duration_s=1)
    # the whole trace is one qualifying stop starting at 0 -> one
    # microtrip covering the whole trace (boundary at 0 only, since
    # there's no "next" stop start after position 0)
    assert len(microtrips) == 1
    assert summary["start_idx"].iloc[0] == 0
    assert summary["end_idx"].iloc[0] == 3


def test_min_distance_filter():
    # short microtrip: 1 second at 10 km/h -> distance ~0.00278 km
    speeds = [10, 0, 0, 50, 50, 50, 0, 0]
    frame = _frame(speeds)
    _microtrips, summary = segment_microtrips(
        frame, min_stop_duration_s=2, min_microtrip_distance_km=0.01
    )
    # first microtrip (10 km/h, 1 s) should be filtered out
    assert all(d >= 0.01 for d in summary["distance_km"])


def test_min_duration_filter():
    speeds = [10, 0, 0, 50, 50, 50, 0, 0]
    frame = _frame(speeds)
    _microtrips, summary = segment_microtrips(
        frame, min_stop_duration_s=2, min_microtrip_duration_s=2
    )
    assert all(n >= 2 for n in summary["n_seconds"])


def test_bad_input_raises():
    with pytest.raises(ValueError):
        segment_microtrips(pd.DataFrame({"Speed": []}))
    with pytest.raises(ValueError):
        segment_microtrips(pd.DataFrame({"NotSpeed": [1, 2, 3]}))
    with pytest.raises(ValueError):
        segment_microtrips(_frame([1, 2, 3]), speed_threshold_kmh=-1)
    with pytest.raises(ValueError):
        segment_microtrips(_frame([1, 2, 3]), min_stop_duration_s=0)
