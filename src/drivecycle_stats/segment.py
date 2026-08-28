"""Microtrip segmentation.

A microtrip is the portion of a trip from the start of one stop to the
start of the next stop. A stop is a run of consecutive seconds at or
below a speed threshold, lasting at least a minimum duration. This is a
standard, widely used definition in driving-cycle literature; only the
implementation here is new.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_SPEED_THRESHOLD_KMH = 1.0
DEFAULT_MIN_STOP_DURATION_S = 1


def segment_microtrips(
    frame: pd.DataFrame,
    speed_col: str = "Speed",
    speed_threshold_kmh: float = DEFAULT_SPEED_THRESHOLD_KMH,
    min_stop_duration_s: int = DEFAULT_MIN_STOP_DURATION_S,
    min_microtrip_distance_km: float | None = None,
    min_microtrip_duration_s: float | None = None,
):
    """Split a per-second trip frame into microtrips.

    Parameters
    ----------
    frame : pandas.DataFrame
        Per-second trace, indexed 0..N-1 in time order, containing at
        least the speed column named by ``speed_col``.
    speed_col : str, default "Speed"
        Name of the speed column, in km/h.
    speed_threshold_kmh : float, default 1.0
        A second is "stopped" if speed is at or below this value.
        The private predecessor of this function tested for speed
        exactly equal to 0.0, which fails on any logger with sensor
        noise; this threshold is a deliberate improvement.
    min_stop_duration_s : int, default 1
        A run of stopped seconds must be at least this long to count
        as a stop event that ends a microtrip.
    min_microtrip_distance_km : float, optional
        If given, drop microtrips shorter than this distance. Distance
        is computed by integrating speed over time (km/h over 1 s
        steps), which assumes the frame is already at 1 Hz.
    min_microtrip_duration_s : float, optional
        If given, drop microtrips shorter than this duration in seconds.

    Returns
    -------
    microtrips : list of pandas.DataFrame
        One DataFrame per microtrip, each a contiguous slice of
        ``frame`` with its original index preserved.
    summary : pandas.DataFrame
        One row per microtrip, columns ``start_idx``, ``end_idx``,
        ``n_seconds``, ``distance_km``.

    Raises
    ------
    ValueError
        If ``frame`` is empty, ``speed_col`` is missing, or any
        parameter is out of range.
    """
    if frame is None or len(frame) == 0:
        raise ValueError("frame must be a non-empty DataFrame")
    if speed_col not in frame.columns:
        raise ValueError(f"frame must have a '{speed_col}' column")
    if speed_threshold_kmh < 0:
        raise ValueError("speed_threshold_kmh must be >= 0")
    if min_stop_duration_s < 1:
        raise ValueError("min_stop_duration_s must be >= 1")

    speed = frame[speed_col].to_numpy(dtype=float)
    n = len(speed)

    stopped = speed <= speed_threshold_kmh

    # Identify runs of consecutive stopped seconds that are long enough
    # to count as a "stop event". Use padding to catch runs touching
    # the array edges.
    padded = np.concatenate(([0], stopped.astype(int), [0]))
    diffs = np.diff(padded)
    run_starts = np.where(diffs == 1)[0]
    run_ends = np.where(diffs == -1)[0]  # exclusive end index
    run_lengths = run_ends - run_starts

    valid = run_lengths >= min_stop_duration_s
    stop_starts = run_starts[valid]

    # A microtrip runs from the start of one qualifying stop to the
    # start of the next qualifying stop. If the trace does not begin
    # in a stop, the first microtrip begins at index 0.
    boundaries = sorted(set([0] + stop_starts.tolist() + [n]))
    # Drop a leading 0 duplicate if the trace itself starts with a stop
    # (0 would already be in stop_starts in that case; the set handles it).

    microtrips = []
    rows = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        segment = frame.iloc[start:end]
        n_seconds = end - start
        seg_speed = speed[start:end]
        distance_km = float(np.sum(seg_speed) * (1.0 / 3600.0))

        if min_microtrip_duration_s is not None and n_seconds < min_microtrip_duration_s:
            continue
        if min_microtrip_distance_km is not None and distance_km < min_microtrip_distance_km:
            continue

        microtrips.append(segment)
        rows.append(
            {
                "start_idx": start,
                "end_idx": end - 1,
                "n_seconds": n_seconds,
                "distance_km": distance_km,
            }
        )

    summary = pd.DataFrame(rows)
    return microtrips, summary
