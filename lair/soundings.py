"""
Upper air sounding data.
"""

from collections import deque
import datetime as dt
import logging
import os
import numpy as np
import pandas as pd
import requests
from time import sleep
import xarray as xr

from lair.air import wind_direction
from lair.config import get_data_dir
from lair._optional import import_optional_dependency

# Optional dependency
siphon = import_optional_dependency("siphon")
from siphon.simplewebservice.wyoming import WyomingUpperAir

logger = logging.getLogger(__name__)


#: Environment variable holding the sounding archive root (one subdirectory per station)
SOUNDING_DIR_ENV = "LAIR_SOUNDING_DIR"


class Sounding:
    """
    Upper air sounding data.

    Attributes
    ----------
    path : str
        The path to the sounding data.
    filename : str
        The filename of the sounding data.
    station : str
        The station identifier.
    time : datetime
        The date and time of the sounding.
    data : pd.DataFrame
        The sounding data.

    Methods
    -------
    interpolate(start=1289, stop=5000, interval=10)
        Interpolate sounding data to regular height intervals.
    """

    _attrs = [
        "station",
        "time",
        "station_number",
        "latitude",
        "longitude",
        "elevation",
        "pw",
    ]

    units = {
        "pressure": "hPa",
        "height": "meter",
        "temperature": "degC",
        "dewpoint": "degC",
        "direction": "degrees",
        "speed": "knot",
        "u_wind": "knot",
        "v_wind": "knot",
        "latitude": "degrees",
        "longitude": "degrees",
        "elevation": "meter",
        "pw": "millimeter",
    }

    def __init__(self, path: str):
        """
        Initialize a Sounding object.

        Strips the station identifier and time from the filename and reads the data.

        Parameters
        ----------
        path : str
            The path to the sounding data.
        """
        self.path = path
        self.filename = os.path.basename(path)

        self.station = self.filename.split("_")[0]
        self.time = dt.datetime.strptime(
            self.filename.split("_")[1].split(".")[0], "%Y%m%d%H"
        )

        self.data = pd.read_csv(path, parse_dates=["time"])
        self.data.drop(columns=["station", "time"], inplace=True)

        for attr in self._attrs:
            if attr in self.data.columns:
                setattr(self, attr, self.data[attr].iloc[0])
                self.data.drop(columns=attr, inplace=True)

    def interpolate(self, start=1289, stop=5000, interval=10):
        """Interpolate sounding data to a specified height.

        Parameters
        ----------
        start : int
            The starting height.
        stop : int
            The stopping height.
        step : int
            The height step.

        Returns
        -------
        xr.Dataset
            The interpolated sounding data.
        """
        height = pd.Index(range(start, stop, interval), dtype=float, name="height")
        raw = self.data.dropna(subset="height").set_index("height").sort_index()
        # One row per raw level (keep the first of any repeats) so a target
        # height that equals a raw level shares its row and takes its values.
        # A separate target row there would sort before or after the raw row
        # and be left NaN at the bottom or top level (#45).
        raw = raw[~raw.index.duplicated()]

        # Only fill between observed levels, so heights above the sounding top
        # stay NaN instead of being extrapolated
        data = raw.reindex(raw.index.union(height))
        data = data.interpolate(method="index", limit_area="inside")
        data = data.reindex(height)

        # Direction can't be interpolated linearly (350 and 10 deg would give
        # 180), so rebuild direction and speed from the interpolated u/v
        if "u_wind" in data.columns and "v_wind" in data.columns:
            data["direction"] = wind_direction(data["u_wind"], data["v_wind"])
            data["speed"] = np.hypot(data["u_wind"], data["v_wind"])

        ds = data.to_xarray().expand_dims(time=[self.time])
        ds["pw"] = (("time",), [getattr(self, "pw", float("nan"))])

        attrs = {
            attr: getattr(self, attr)
            for attr in self._attrs
            if hasattr(self, attr) and attr not in ["time", "pw"]
        }
        ds.attrs.update(attrs)
        ds.attrs["interpolation_interval"] = interval
        ds.attrs["notes"] = f"Interpolated to {start}-{stop}m at {interval}m intervals."

        return ds

    def plot(self):
        # TODO
        raise NotImplementedError


def merge(soundings: list) -> pd.DataFrame:
    """
    Merge a list of sounding data.

    Parameters
    ----------
    soundings : list
        A list of Sounding objects.

    Returns
    -------
    pd.DataFrame
        The merged sounding data.
    """
    dfs = []
    for sounding in soundings:
        df = sounding.data.copy()
        for attr in sounding._attrs:
            if hasattr(sounding, attr):
                df[attr] = getattr(sounding, attr)
        dfs.append(df)

    return pd.concat(dfs)


def _naive_utc(t) -> pd.Timestamp:
    """Timestamp in naive UTC (the sounding file times); naive input is taken as UTC."""
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t


def download_sounding(station, date, dst=None) -> str:
    """
    Download an upper air sounding from the Wyoming archive.

    Parameters
    ----------
    station : str
        The 3-letter station identifier (e.g. ``'SLC'``) passed to the Wyoming
        service via siphon. Also used as the file-name prefix.
    date : datetime
        The date and time.
    dst : str
        The destination directory. Defaults to ``$LAIR_SOUNDING_DIR/<station>``.

    Returns
    -------
    str
        The path to the downloaded sounding data.
    """
    if dst is None:
        dst = os.path.join(get_data_dir(SOUNDING_DIR_ENV), station)
    os.makedirs(dst, exist_ok=True)

    path = os.path.join(dst, f"{station}_{date:%Y%m%d%H}.csv")
    if not os.path.exists(path):
        logger.info("Downloading %s on %s...", station, f"{date:%Y-%m-%d %H:%M}")
        df = WyomingUpperAir.request_data(date, station)
        df.to_csv(path, index=False)
    else:
        logger.debug("%s on %s already exists.", station, f"{date:%Y-%m-%d %H:%M}")

    return path


def download_soundings(station, start, end, dst=None, months=None):
    """
    Download upper air soundings from the Wyoming archive.

    Parameters
    ----------
    station : str
        The 3-letter station identifier (e.g. ``'SLC'``).
    start : datetime
        The start date and time. Soundings are at 00 and 12 UTC, so the first
        one requested is the first of those at or after ``start``.
    end : datetime
        The end date and time (inclusive). Naive times are taken as UTC;
        tz-aware times are converted.
    dst : str
        The destination directory.
    months : list
        The months to download.
    """
    logger.info("Downloading soundings...")

    # Snap to the 00/12 UTC synoptic times; date_range would otherwise anchor
    # on start (start=06:00 -> 06Z, 18Z, ...)
    start = _naive_utc(start).ceil("12h")
    end = _naive_utc(end).floor("12h")
    dates = pd.date_range(start, end, freq="12h")

    if months:
        dates = dates[dates.month.isin(months)]

    to_download = deque(dates)
    while len(to_download) > 0:
        date = to_download.popleft()
        try:
            download_sounding(station, date, dst)
        except IndexError as e:
            logger.warning(
                "Error downloading %s on %s: %s", station, f"{date:%Y-%m-%d %H:%M}", e
            )
            continue
        except ValueError as e:
            logger.warning(
                "Error downloading %s on %s: %s", station, f"{date:%Y-%m-%d %H:%M}", e
            )
            if "No data available" in str(e):
                continue
            else:
                raise e
        except requests.exceptions.HTTPError as e:
            logger.warning(
                "Error downloading %s on %s: %s", station, f"{date:%Y-%m-%d %H:%M}", e
            )
            if "Please try again later" in str(e):
                logger.info("Trying again in 2 seconds...")
                to_download.appendleft(date)
                sleep(2)
            else:
                raise e


def get_soundings(
    station="SLC",
    start=None,
    end=None,
    sounding_dir=None,
    months=None,
    driver="xarray",
    **kwargs,
):
    """
    Get upper air soundings from the Wyoming archive.

    Parameters
    ----------
    station : str
        The 3-letter station identifier (e.g. ``'SLC'``). Names the
        subdirectory of ``$LAIR_SOUNDING_DIR`` and the file prefix.
    start : datetime | str, optional
        The start date and time (inclusive). Naive times are taken as UTC;
        tz-aware times are converted.
    end : datetime | str, optional
        The end date and time (inclusive).
    sounding_dir : str
        The directory containing the station's sounding files. Defaults to
        ``$LAIR_SOUNDING_DIR/<station>``.
    months : list[int], optional
        Only use (and download) soundings from these months.
    driver : {'xarray', 'pandas'}, optional
        Return interpolated profiles as an xarray Dataset (default), or the
        merged raw soundings as a pandas DataFrame.

    Returns
    -------
    xr.Dataset | pd.DataFrame
        The sounding data.
    """
    # File times are naive UTC, so compare against naive UTC bounds
    start = _naive_utc(start).to_pydatetime() if start is not None else None
    end = _naive_utc(end).to_pydatetime() if end is not None else None

    if sounding_dir is None:
        sounding_dir = os.path.join(get_data_dir(SOUNDING_DIR_ENV), station)

    files = os.listdir(sounding_dir) if os.path.isdir(sounding_dir) else []
    if len(files) == 0:
        logger.info("No soundings found. Downloading...")

        if not all([start, end]):
            raise ValueError(
                "start and end must be specified if no soundings are found."
            )
        download_soundings(station, start, end, sounding_dir, months)
        files = os.listdir(sounding_dir)

    soundings = []
    for file in files:
        # Skip anything that isn't a <station>_<YYYYmmddHH>.csv sounding
        try:
            date = dt.datetime.strptime(file.split("_")[1].split(".")[0], "%Y%m%d%H")
        except (IndexError, ValueError):
            continue

        # Skip files that don't match the date range
        if start:
            if date < start:
                continue
        if end:
            if date > end:
                continue
        if months and date.month not in months:
            continue

        path = os.path.join(sounding_dir, file)
        try:
            soundings.append(Sounding(path))
        except Exception as e:
            logger.warning("Error reading file %s: %s", path, e)
            continue

    if not soundings:
        raise ValueError(
            f"No soundings found in {sounding_dir} for the requested period."
        )

    if driver in ["xarray", "nc"]:
        data = xr.concat(
            [sounding.interpolate() for sounding in soundings], dim="time"
        ).sortby("time")
    elif driver in ["pandas", "csv"]:
        data = merge(soundings)
    else:
        raise ValueError(f"Invalid driver: {driver}")

    return data
