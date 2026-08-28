import pandas as pd
import pytest

from drivecycle_stats.io_cycles import load_cycle_csv


def test_load_cycle_csv(tmp_path):
    csv_path = tmp_path / "cycle.csv"
    pd.DataFrame({"time_s": [2, 0, 1], "speed_kmh": [20.0, 0.0, 10.0]}).to_csv(
        csv_path, index=False
    )
    frame = load_cycle_csv(csv_path)
    assert list(frame.columns) == ["Speed"]
    # sorted by time_s -> 0,10,20
    assert frame["Speed"].tolist() == [0.0, 10.0, 20.0]


def test_load_cycle_csv_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_cycle_csv("does_not_exist.csv")


def test_load_cycle_csv_bad_columns_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_cycle_csv(csv_path)


def test_load_epa_cycle_txt_parses_and_converts(tmp_path):
    """EPA .txt files are tab-separated, have two header lines, and are in mph."""
    from drivecycle_stats.io_cycles import MPH_TO_KMH, load_epa_cycle_txt

    p = tmp_path / "ftpcol.txt"
    p.write_text(
        "FTPCOL.TXT\tFederal Test Procedure\n"
        "Test Time, secs\tTarget Speed, mph\n"
        "0\t0\n1\t10\n2\t25.5\n3\t0\n"
    )
    frame = load_epa_cycle_txt(p)

    assert list(frame.columns) == ["Speed"]
    assert len(frame) == 4
    assert frame["Speed"].iloc[0] == 0.0
    assert abs(frame["Speed"].iloc[1] - 10 * MPH_TO_KMH) < 1e-9
    assert abs(frame["Speed"].iloc[2] - 25.5 * MPH_TO_KMH) < 1e-9
    # 25.5 mph is about 41 km/h; if the conversion were skipped this fails.
    assert 40.0 < frame["Speed"].iloc[2] < 42.0


def test_load_epa_cycle_txt_missing_file(tmp_path):
    from drivecycle_stats.io_cycles import load_epa_cycle_txt

    with pytest.raises(FileNotFoundError):
        load_epa_cycle_txt(tmp_path / "nope.txt")
