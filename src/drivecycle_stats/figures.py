"""Figures, in a fixed double-column journal style.

Every function returns a Matplotlib ``Figure`` so the caller decides
whether and where to save it. Style rules (serif font, bold labels,
600 dpi save target, no gridlines) are applied once at import time and
are not configurable per call, to keep every figure in this package
consistent.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "bold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "lines.markersize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})

DOUBLE_COLUMN_FIGSIZE = (7, 5)


def _place_legend_safely(ax, prefer="inside", ncol=1):
    if prefer == "inside":
        return ax.legend(loc="best", frameon=True, framealpha=0.9, ncol=ncol)
    elif prefer == "above":
        return ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=ncol, frameon=False)
    elif prefer == "below":
        return ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=ncol, frameon=False)
    else:
        raise ValueError("prefer must be 'inside', 'above', or 'below'")


def plot_speed_trace_with_microtrips(frame, microtrip_summary, title="Speed trace with microtrip boundaries"):
    """Figure 1: speed trace of one trip with microtrip boundaries marked.

    Parameters
    ----------
    frame : pandas.DataFrame
        Must have a ``Speed`` column, one row per second.
    microtrip_summary : pandas.DataFrame
        The ``summary`` table returned by
        :func:`drivecycle_stats.segment.segment_microtrips`, with a
        ``start_idx`` column giving the boundary seconds.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN_FIGSIZE)
    t = np.arange(len(frame))
    ax.plot(t, frame["Speed"].to_numpy(), color="tab:blue", label="Speed")

    for i, start in enumerate(microtrip_summary["start_idx"]):
        ax.axvline(start, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.7,
                   label="Microtrip boundary" if i == 0 else None)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title(title)
    _place_legend_safely(ax, prefer="inside")
    fig.tight_layout()
    return fig


def plot_speed_accel_density(trips, title="Speed-acceleration density"):
    """Figure 2: speed-acceleration scatter/density for a set of trips.

    Parameters
    ----------
    trips : list of pandas.DataFrame
        Each must have ``Speed`` and ``Acceleration`` columns.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    speeds = np.concatenate([t["Speed"].to_numpy() for t in trips])
    accels = np.concatenate([t["Acceleration"].to_numpy() for t in trips])

    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN_FIGSIZE)
    ax.hexbin(speeds, accels, gridsize=40, cmap="viridis", mincnt=1)
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_descriptor_distributions(real_values, reference_values, descriptor_name, title=None):
    """Figure 3: descriptor distribution, real trips against a reference cycle.

    Parameters
    ----------
    real_values : array-like
        Descriptor values from real trips (e.g. a column of a
        descriptors table).
    reference_values : array-like
        The same descriptor computed for one or more reference-cycle
        trips (may be a single value or a short array).
    descriptor_name : str
        Used for the x-axis label, e.g. "avgspd (km/h)".
    title : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    real_values = np.asarray(real_values, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)

    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN_FIGSIZE)
    ax.hist(real_values, bins=30, density=True, alpha=0.6, color="tab:blue", label="Real (VED)")
    for v in np.atleast_1d(reference_values):
        ax.axvline(v, color="tab:red", linewidth=2.0, label="Reference cycle")

    ax.set_xlabel(descriptor_name)
    ax.set_ylabel("Density")
    ax.set_title(title or f"Distribution of {descriptor_name}")
    _place_legend_safely(ax, prefer="inside")
    fig.tight_layout()
    return fig


def plot_permutation_null(null_distribution, observed_statistic, title="Permutation null distribution"):
    """Figure 4: permutation null with the observed statistic marked.

    Parameters
    ----------
    null_distribution : array-like
        The ``null_distribution`` array returned by
        :func:`drivecycle_stats.compare.energy_test`.
    observed_statistic : float
        The ``statistic`` value returned by the same call.
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    null_distribution = np.asarray(null_distribution, dtype=float)

    fig, ax = plt.subplots(figsize=DOUBLE_COLUMN_FIGSIZE)
    ax.hist(null_distribution, bins=40, color="tab:gray", alpha=0.8, label="Null distribution")
    ax.axvline(observed_statistic, color="tab:red", linewidth=2.0, label="Observed statistic")

    ax.set_xlabel("Energy distance")
    ax.set_ylabel("Count")
    ax.set_title(title)
    _place_legend_safely(ax, prefer="inside")
    fig.tight_layout()
    return fig
