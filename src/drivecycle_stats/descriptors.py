"""Trip and microtrip descriptors.

Different papers define these quantities slightly differently. This module
states its own definitions explicitly so results are reproducible and
comparable across studies that may use a different convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .vsp import vsp as _vsp

IDLE_SPEED_THRESHOLD_KMH = 1.0
MOVING_ACCEL_THRESHOLD_MS2 = 0.1


def trip_descriptors(frame: pd.DataFrame) -> dict:
    """Compute descriptors for one trip or microtrip.

    Parameters
    ----------
    frame : pandas.DataFrame
        A per-second (or resampled-to-per-second) speed trace with at
        least a ``Speed`` column (km/h) and an ``Acceleration`` column
        (m/s^2). An optional ``TripKm`` column, cumulative distance in
        kilometres, may be supplied; if absent, distance is obtained by
        integrating speed over time.

    Returns
    -------
    dict
        Keys and definitions:

        - ``distance_km``: total distance travelled.
        - ``duration_s``: number of seconds in the frame.
        - ``avgspd``: mean speed over ALL seconds, including idle
          seconds (speed at or below the idle threshold).
        - ``runspd``: mean speed over moving seconds only (idle seconds
          excluded).
        - ``avgposacc``: mean acceleration, averaged only over seconds
          where acceleration exceeds 0.1 m/s^2 (i.e. seconds that are
          meaningfully accelerating, not just above zero due to noise).
        - ``rmsa``: root-mean-square acceleration over all seconds.
        - ``idle_share``: fraction of seconds at or below 1 km/h.
        - ``n_stops``: number of stop events, defined as maximal runs
          of consecutive idle seconds.
        - ``mean_stop_duration_s``: mean length of those stop events,
          in seconds. NaN if there are no stops.
        - ``v95``: 95th percentile of speed over all seconds.
        - ``vsp_pos_mean``: mean vehicle-specific power, averaged only
          over seconds where VSP is positive.

    Raises
    ------
    ValueError
        If required columns are missing or the frame is empty.
    """
    if frame is None or len(frame) == 0:
        raise ValueError("frame must be a non-empty DataFrame")
    if "Speed" not in frame.columns:
        raise ValueError("frame must have a 'Speed' column (km/h)")
    if "Acceleration" not in frame.columns:
        raise ValueError("frame must have an 'Acceleration' column (m/s^2)")

    speed = frame["Speed"].to_numpy(dtype=float)
    accel = frame["Acceleration"].to_numpy(dtype=float)
    n = len(speed)

    duration_s = n

    if "TripKm" in frame.columns:
        distance_km = float(frame["TripKm"].iloc[-1] - frame["TripKm"].iloc[0])
    else:
        # Integrate speed (km/h) over 1 s steps -> km.
        distance_km = float(np.sum(speed) * (1.0 / 3600.0))

    idle_mask = speed <= IDLE_SPEED_THRESHOLD_KMH
    moving_mask = ~idle_mask

    avgspd = float(np.mean(speed))
    runspd = float(np.mean(speed[moving_mask])) if moving_mask.any() else float("nan")

    accel_mask = accel > MOVING_ACCEL_THRESHOLD_MS2
    avgposacc = float(np.mean(accel[accel_mask])) if accel_mask.any() else float("nan")

    rmsa = float(np.sqrt(np.mean(accel**2)))

    idle_share = float(np.mean(idle_mask))

    n_stops, mean_stop_duration_s = _stop_structure(idle_mask)

    v95 = float(np.percentile(speed, 95))

    vsp_values = _vsp(speed, accel)
    pos_mask = vsp_values > 0
    vsp_pos_mean = float(np.mean(vsp_values[pos_mask])) if pos_mask.any() else float("nan")

    return {
        "distance_km": distance_km,
        "duration_s": duration_s,
        "avgspd": avgspd,
        "runspd": runspd,
        "avgposacc": avgposacc,
        "rmsa": rmsa,
        "idle_share": idle_share,
        "n_stops": n_stops,
        "mean_stop_duration_s": mean_stop_duration_s,
        "v95": v95,
        "vsp_pos_mean": vsp_pos_mean,
    }


def _stop_structure(idle_mask: np.ndarray) -> tuple[int, float]:
    """Count stop events and their mean duration from a boolean idle mask.

    A stop event is a maximal run of consecutive True values.
    """
    if not idle_mask.any():
        return 0, float("nan")

    # Find run boundaries via difference of the boolean-as-int mask.
    padded = np.concatenate(([0], idle_mask.astype(int), [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    lengths = ends - starts

    return len(lengths), float(np.mean(lengths))


def descriptors_table(frames: list[pd.DataFrame], ids: list | None = None) -> pd.DataFrame:
    """Apply :func:`trip_descriptors` across a list of frames.

    Parameters
    ----------
    frames : list of pandas.DataFrame
        Each must satisfy the requirements of :func:`trip_descriptors`.
    ids : list, optional
        Identifiers for each frame, used as the index of the returned
        table. If omitted, a 0-based integer index is used.

    Returns
    -------
    pandas.DataFrame
        One row per input frame, columns as in :func:`trip_descriptors`.
    """
    rows = [trip_descriptors(f) for f in frames]
    table = pd.DataFrame(rows)
    if ids is not None:
        if len(ids) != len(frames):
            raise ValueError("ids must be the same length as frames")
        table.index = ids
    return table
