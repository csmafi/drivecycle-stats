"""Loader and cleaner for the Vehicle Energy Dataset (VED).

VED: Oh, G., LeBlanc, D.J., Peng, H. (2020). "Vehicle Energy Dataset
(VED), A Large-scale Dataset for Vehicle Energy Consumption Research."
IEEE Transactions on Intelligent Transportation Systems.
Repository: https://github.com/gsoh/VED
Licence: Apache-2.0 (the dataset's own licence; this loader is MIT and
is independent of the data it reads).

VED is not shipped with this package. Download it yourself from the
repository above:

    Data/VED_DynamicData_Part1.7z
    Data/VED_DynamicData_Part2.7z
    Data/VED_Static_Data_ICE&HEV.xlsx
    Data/VED_Static_Data_PHEV&EV.xlsx

Each .7z extracts to a set of "VED_mmddyy_week.csv" files, one file per
week of logging. This loader reads the .csv files directly; unpack the
archives yourself with any 7z-capable tool before calling this module.

Confirmed real schema (verified against the raw files, not assumed):

    DayNum, VehId, Trip, Timestamp(ms), Latitude[deg], Longitude[deg],
    Vehicle Speed[km/h], MAF[g/sec], Engine RPM[RPM], Absolute Load[%],
    OAT[DegC], Fuel Rate[L/hr], Air Conditioning Power[kW],
    Air Conditioning Power[Watts], Heater Power[Watts],
    HV Battery Current[A], HV Battery SOC[%], HV Battery Voltage[V],
    Short Term Fuel Trim Bank 1[%], Short Term Fuel Trim Bank 2[%],
    Long Term Fuel Trim Bank 1[%], Long Term Fuel Trim Bank 2[%]

Two things this loader must handle that a naive read would miss:

1. Timestamps within a trip are NOT evenly spaced. Measured gaps
   between consecutive samples range from about 100 ms to about 2800
   ms. This loader resamples every trip to a uniform 1 Hz grid by
   linear interpolation on Timestamp(ms) before computing
   acceleration, so acceleration is not computed on an uneven time
   base.

2. The two static-data files use different column names for the same
   field: "Vehicle Type" in the ICE&HEV file versus "EngineType" in
   the PHEV&EV file. This loader renames both to a single
   "EngineType" column when merging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_SPEED_COL = "Vehicle Speed[km/h]"
MAX_PHYSICAL_ACCEL_MS2 = 5.0  # clip threshold, generous for a light-duty car


def load_ved_dynamic_csv(path: str | Path) -> pd.DataFrame:
    """Read one raw VED_mmddyy_week.csv file with no cleaning applied.

    Parameters
    ----------
    path : str or Path
        Path to one extracted "VED_mmddyy_week.csv" file.

    Returns
    -------
    pandas.DataFrame
        The file as-is, with original VED column names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VED file not found: {path}")
    return pd.read_csv(path)


def load_ved_static(ice_hev_path: str | Path, phev_ev_path: str | Path) -> pd.DataFrame:
    """Read and merge the two VED static vehicle-parameter files.

    Reconciles the column name mismatch between the two source files
    ("Vehicle Type" in the ICE&HEV file, "EngineType" in the PHEV&EV
    file) into a single "EngineType" column.

    Parameters
    ----------
    ice_hev_path : str or Path
        Path to "VED_Static_Data_ICE&HEV.xlsx".
    phev_ev_path : str or Path
        Path to "VED_Static_Data_PHEV&EV.xlsx".

    Returns
    -------
    pandas.DataFrame
        One row per vehicle, indexed by VehId, with a single
        "EngineType" column covering both source files.
    """
    ice_hev = pd.read_excel(ice_hev_path)
    phev_ev = pd.read_excel(phev_ev_path)

    if "Vehicle Type" in ice_hev.columns:
        ice_hev = ice_hev.rename(columns={"Vehicle Type": "EngineType"})

    combined = pd.concat([ice_hev, phev_ev], ignore_index=True, sort=False)
    return combined


def clean_trip(
    raw_trip: pd.DataFrame,
    speed_col: str = RAW_SPEED_COL,
    timestamp_col: str = "Timestamp(ms)",
    max_accel_ms2: float = MAX_PHYSICAL_ACCEL_MS2,
    min_duration_s: int = 10,
):
    """Clean one raw VED trip into a 1 Hz frame with Speed and Acceleration.

    Steps, in order, all logged in the returned report:

    1. Drop rows with a missing or negative speed value.
    2. Resample onto a uniform 1 Hz grid by linear interpolation of
       speed against elapsed time, since raw timestamps are not evenly
       spaced.
    3. Compute Acceleration by first-differencing the resampled speed.
    4. Clip acceleration to +/- ``max_accel_ms2``, a physically
       plausible bound for a light-duty car; this is deliberately
       conservative, not a data-driven limit.
    5. Drop the trip entirely if the resampled duration is shorter
       than ``min_duration_s``.

    Parameters
    ----------
    raw_trip : pandas.DataFrame
        Rows for one (VehId, Trip) pair from a raw VED file, in time
        order.
    speed_col : str, default "Vehicle Speed[km/h]"
        Name of the raw speed column.
    timestamp_col : str, default "Timestamp(ms)"
        Name of the raw timestamp column, milliseconds elapsed since
        trip start.
    max_accel_ms2 : float, default 5.0
        Acceleration clip threshold in m/s^2, applied symmetrically.
    min_duration_s : int, default 10
        Minimum resampled trip duration to keep. Trips shorter than
        this are dropped (returns ``None`` for the frame).

    Returns
    -------
    frame : pandas.DataFrame or None
        Columns ``Speed`` (km/h) and ``Acceleration`` (m/s^2) on a
        uniform 1 Hz grid, or ``None`` if the trip was dropped.
    report : dict
        ``n_raw_rows``, ``n_dropped_missing_speed``,
        ``n_resampled_rows``, ``n_accel_clipped``, ``dropped_reason``
        (None if kept).
    """
    n_raw_rows = len(raw_trip)
    report = {
        "n_raw_rows": n_raw_rows,
        "n_dropped_missing_speed": 0,
        "n_resampled_rows": 0,
        "n_accel_clipped": 0,
        "dropped_reason": None,
    }

    if speed_col not in raw_trip.columns or timestamp_col not in raw_trip.columns:
        report["dropped_reason"] = "missing required column"
        return None, report

    trip = raw_trip[[timestamp_col, speed_col]].copy()
    trip.columns = ["t_ms", "speed_raw"]

    before = len(trip)
    trip = trip.dropna(subset=["speed_raw"])
    trip = trip[trip["speed_raw"] >= 0]
    report["n_dropped_missing_speed"] = before - len(trip)

    if len(trip) < 2:
        report["dropped_reason"] = "fewer than 2 valid speed samples"
        return None, report

    trip = trip.sort_values("t_ms")
    t_start = trip["t_ms"].iloc[0]
    t_end = trip["t_ms"].iloc[-1]
    duration_s = (t_end - t_start) / 1000.0

    if duration_s < min_duration_s:
        report["dropped_reason"] = f"resampled duration {duration_s:.1f}s below minimum"
        return None, report

    grid_s = np.arange(0, int(duration_s) + 1, 1)
    grid_ms = t_start + grid_s * 1000.0

    speed_1hz = np.interp(grid_ms, trip["t_ms"].to_numpy(), trip["speed_raw"].to_numpy())

    accel = np.diff(speed_1hz, prepend=speed_1hz[0]) / 3.6  # km/h per s -> m/s^2
    accel[0] = 0.0

    n_clipped = int(np.sum(np.abs(accel) > max_accel_ms2))
    accel = np.clip(accel, -max_accel_ms2, max_accel_ms2)

    report["n_resampled_rows"] = len(speed_1hz)
    report["n_accel_clipped"] = n_clipped

    frame = pd.DataFrame({"Speed": speed_1hz, "Acceleration": accel})
    return frame, report


def iter_ved_trips(raw: pd.DataFrame):
    """Yield (VehId, Trip, raw_trip_frame) for each trip in a raw VED file.

    Parameters
    ----------
    raw : pandas.DataFrame
        A raw VED file as returned by :func:`load_ved_dynamic_csv`.

    Yields
    ------
    tuple
        ``(veh_id, trip_id, trip_frame)``, ``trip_frame`` sorted by
        ``Timestamp(ms)``.
    """
    if "VehId" not in raw.columns or "Trip" not in raw.columns:
        raise ValueError("raw frame must have VehId and Trip columns")

    for (veh_id, trip_id), group in raw.groupby(["VehId", "Trip"], sort=False):
        yield veh_id, trip_id, group.sort_values("Timestamp(ms)")
