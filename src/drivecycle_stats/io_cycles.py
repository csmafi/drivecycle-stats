"""Loader for legislated reference driving cycles.

These are public regulatory speed traces, not this package's own data.
This package does not redistribute them: US EPA cycles (FTP-75, US06,
HWFET) are published by the US Environmental Protection Agency, and
WLTC is published by UNECE as part of the Global Technical Regulation
No. 15. Sources:

- US EPA drive cycles: https://www.epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
- WLTC: https://unece.org/transport/documents/2021/06/standards/gtr-no15-worldwide-harmonized-light-vehicles-test

Two loaders are provided, because the two sources have different formats:

- :func:`load_cycle_csv` reads a cycle you already hold as a two-column
  (time_s, speed_kmh) CSV. Use this for WLTC, which UNECE publishes as a
  table inside GTR No. 15 rather than as a downloadable file, so you must
  transcribe it into a CSV yourself.
- :func:`load_epa_cycle_txt` reads the raw EPA ``.txt`` files as published:
  tab-separated, two header lines, and speed in **miles per hour**. It
  converts to km/h for you. Pair it with :func:`download_epa_cycle`.

The EPA route is the one that works end to end with no manual steps, so
prefer it unless you specifically need WLTC.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

EPA_CYCLE_URLS = {
    "FTP75": "https://www.epa.gov/sites/default/files/2015-10/ftpcol.txt",
    "US06": "https://www.epa.gov/sites/default/files/2015-10/us06col.txt",
    "HWFET": "https://www.epa.gov/sites/default/files/2015-10/hwycol.txt",
}


def load_cycle_csv(path: str | Path, time_col: str = "time_s", speed_col: str = "speed_kmh") -> pd.DataFrame:
    """Load a reference cycle from a local two-column CSV.

    Parameters
    ----------
    path : str or Path
        CSV with a time column (seconds) and a speed column (km/h).
    time_col : str, default "time_s"
    speed_col : str, default "speed_kmh"

    Returns
    -------
    pandas.DataFrame
        Columns renamed to ``Speed`` (km/h), indexed by second, ready
        to hand to :func:`drivecycle_stats.descriptors.trip_descriptors`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"cycle file not found: {path}")

    raw = pd.read_csv(path)
    if time_col not in raw.columns or speed_col not in raw.columns:
        raise ValueError(
            f"expected columns '{time_col}' and '{speed_col}', got {raw.columns.tolist()}"
        )

    raw = raw.sort_values(time_col).reset_index(drop=True)
    return pd.DataFrame({"Speed": raw[speed_col].to_numpy()})


MPH_TO_KMH = 1.609344


def load_epa_cycle_txt(path: str | Path) -> pd.DataFrame:
    """Load a raw EPA drive-cycle ``.txt`` file as published by the EPA.

    The EPA files are not CSVs. Their layout, verified against the live
    files on 2026-08-28, is:

    - Line 1: a title, e.g. ``FTPCOL.TXT<TAB>Federal Test Procedure``
    - Line 2: column headings, ``Test Time, secs<TAB>Target Speed, mph``
    - Line 3 onward: tab-separated time in seconds and **speed in mph**

    This function skips the two header lines, reads the tab-separated
    values, and converts mph to km/h. Use it instead of
    :func:`load_cycle_csv` for anything fetched by
    :func:`download_epa_cycle`.

    Parameters
    ----------
    path : str or Path
        Path to a downloaded EPA ``.txt`` file.

    Returns
    -------
    pandas.DataFrame
        One column, ``Speed``, in km/h, one row per second.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file does not parse as two numeric columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"EPA cycle file not found: {path}")

    raw = pd.read_csv(path, sep="\t", skiprows=2, header=None,
                      names=["time_s", "speed_mph"])
    raw = raw.apply(pd.to_numeric, errors="coerce").dropna()
    if raw.empty or raw.shape[1] != 2:
        raise ValueError(
            f"{path} did not parse as two numeric columns; the EPA file "
            "layout may have changed. Check the first three lines by hand."
        )

    raw = raw.sort_values("time_s").reset_index(drop=True)
    return pd.DataFrame({"Speed": raw["speed_mph"].to_numpy() * MPH_TO_KMH})


def download_epa_cycle(name: str, dest: str | Path) -> Path:
    """Download one of the standard EPA drive cycles to a local file.

    The saved file is the EPA's own ``.txt`` format. Read it with
    :func:`load_epa_cycle_txt`, not :func:`load_cycle_csv`.

    This function requires network access and is not exercised by the
    test suite. If it fails (site layout changes, no network), download
    the file manually from the URL in ``EPA_CYCLE_URLS``.

    Parameters
    ----------
    name : {"FTP75", "US06", "HWFET"}
        Which EPA cycle to fetch.
    dest : str or Path
        Local path to save the raw file to.

    Returns
    -------
    Path
        The path the file was saved to.
    """
    if name not in EPA_CYCLE_URLS:
        raise ValueError(f"unknown cycle name {name!r}, choose from {list(EPA_CYCLE_URLS)}")

    import urllib.request

    url = EPA_CYCLE_URLS[name]
    dest = Path(dest)
    urllib.request.urlretrieve(url, dest)
    return dest
