"""
Upper-air soundings from the University of Wyoming's archive.

:class:`Sounding` reads one sounding file and interpolates it to regular
heights; :func:`get_soundings` reads a station's soundings for a period, all
on the same heights, and :func:`download_soundings` fetches them from Wyoming.
The folder is an argument or ``$LAIR_SOUNDING_DIR/<station>``.
:doc:`/examples/soundings` reads two soundings and computes their potential
temperature. The module needs the ``requests`` extra.
"""

import datetime as dt
import logging
import os
from collections import deque
from time import sleep

import numpy as np
import pandas as pd
import xarray as xr

from lair import units
from lair._optional import import_optional_dependency
from lair.air import wind_direction
from lair.config import get_data_dir

# Optional dependencies
requests = import_optional_dependency("requests")
siphon = import_optional_dependency("siphon")
from siphon.simplewebservice.wyoming import WyomingUpperAir

logger = logging.getLogger(__name__)


#: Environment variable holding the sounding archive root (one subdirectory per station)
#:
#: .. seealso:: :ref:`data-locations`, which lists every variable;
#:    :doc:`/examples/soundings` shows the folder.
SOUNDING_DIR_ENV = "LAIR_SOUNDING_DIR"

#: The height [m] at which some saved soundings' heights start again from 0
_HEIGHT_WRAP = 10_000.0

#: A fall [m] between two consecutive levels larger than this is a wrap, not
#: a balloon pushed down (those fall 10 to 50 m)
_WRAP_FALL = 9_000.0


class Sounding:
    """
    Upper air sounding data.

    One sounding file, as :func:`download_sounding` saves it: a CSV named
    ``<station>_<YYYYmmddHH>.csv`` for the UTC hour of the sounding, one row per
    level, with the station's attributes repeated in columns of their own.

    Parameters
    ----------
    path : str
        The path to the sounding file. The station and time are read from its
        name.

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
        The sounding data, one row per level in the order of the file.
        Heights that wrap at 10 km are :ref:`unwrapped
        <soundings-wrapped-heights>` as the file is read.
    units : dict[str, str]
        The unit of each column of ``data`` and of each station attribute,
        keyed by name: ``pressure`` in hPa; ``height`` and ``elevation`` in
        meters; ``temperature`` and ``dewpoint`` in degrees Celsius
        (``degC``); ``direction``, ``latitude`` and ``longitude`` in degrees;
        ``speed``, ``u_wind`` and ``v_wind`` in knots; ``pw`` in millimeters.
        :func:`download_sounding` saves every file in these units.

    Methods
    -------
    interpolate(start=None, stop=5000, interval=10)
        Interpolate sounding data to regular height intervals.

    See Also
    --------
    get_soundings : A station's soundings for a period, on the same heights.

    Examples
    --------
    A made-up sounding with two levels, saved as Wyoming's are:

    >>> import pathlib
    >>> import tempfile
    >>> import pandas as pd
    >>> from lair.soundings import Sounding
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> levels = pd.DataFrame(
    ...     {
    ...         "pressure": [870.0, 700.0],
    ...         "height": [1289.0, 3000.0],
    ...         "temperature": [-5.0, -12.0],
    ...         "dewpoint": [-8.0, -20.0],
    ...         "direction": [180.0, 270.0],
    ...         "speed": [5.0, 10.0],
    ...         "u_wind": [0.0, 10.0],
    ...         "v_wind": [5.0, 0.0],
    ...         "station": "SLC",
    ...         "time": "2024-01-15 12:00",
    ...         "elevation": 1289.0,
    ...     }
    ... )
    >>> levels.to_csv(folder / "SLC_2024011512.csv", index=False)
    >>> sounding = Sounding(str(folder / "SLC_2024011512.csv"))
    >>> sounding.station, sounding.time, float(sounding.elevation)
    ('SLC', datetime.datetime(2024, 1, 15, 12, 0), 1289.0)
    >>> list(sounding.data.columns)
    ['pressure', 'height', 'temperature', 'dewpoint', 'direction', 'speed', 'u_wind', 'v_wind']
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
        Initialize a Sounding object; the class docstring describes the parameters.

        Strips the station identifier and time from the filename and reads the data.
        """
        self.path = path
        self.filename = os.path.basename(path)

        self.station = self.filename.split("_")[0]
        self.time = dt.datetime.strptime(
            self.filename.split("_")[1].split(".")[0], "%Y%m%d%H"
        )

        self.data = pd.read_csv(path, parse_dates=["time"])
        self.data = self.data.drop(columns=["station", "time"])
        # Undo heights that wrap at 10 km before anything sorts by height
        if "height" in self.data.columns:
            self.data["height"] = _unwrap_heights(self.data["height"], path)

        for attr in self._attrs:
            if attr in self.data.columns:
                setattr(self, attr, self.data[attr].iloc[0])
                self.data = self.data.drop(columns=attr)

    def _surface_height(self) -> float:
        """Station elevation [m], or the first level's height when the file has none."""
        elevation = getattr(self, "elevation", None)
        if elevation is not None and not pd.isna(elevation):
            return float(elevation)
        return _first_level_height(self.data)

    def interpolate(self, start=None, stop=5000, interval=10):
        """
        Interpolate sounding data to regular height levels.

        Parameters
        ----------
        start : float, optional
            The lowest height, in meters above sea level. Default is the
            ground at the station: its ``elevation``, or, in a file without
            one (soundings downloaded from Wyoming today), the height of the
            first level, which Wyoming lists from the ground up.
        stop : float, optional
            The height to stop at, in meters. Like the end of a ``range``, it
            is not itself a level. Default is 5000.
        interval : float, optional
            The spacing between levels, in meters. Default is 10.

        Returns
        -------
        xr.Dataset
            The interpolated sounding data.

        See Also
        --------
        get_soundings : Interpolates every sounding of a period to the same
            heights.

        Notes
        -----
        The levels are sorted by height before they are interpolated. Heights
        that wrap at 10 km were :ref:`unwrapped <soundings-wrapped-heights>`
        when the file was read; a level reported lower than the one before it,
        by a balloon pushed down, keeps its height.

        Examples
        --------
        Levels every 10 m from the station's ground up to 4999 m above sea
        level (Salt Lake City's ground is 1289 m)::

            ds = Sounding("SLC_2024011512.csv").interpolate()
            ds.height.values[:3]  # array([1289., 1299., 1309.])
        """
        if start is None:
            start = self._surface_height()
        n = int(np.ceil((stop - start) / interval))
        height = pd.Index(
            start + interval * np.arange(max(n, 0)), dtype=float, name="height"
        )
        # A stable sort keeps repeated levels in file order, so "last" below
        # means the row that comes later in the file
        raw = (
            self.data.dropna(subset="height")
            .set_index("height")
            .sort_index(kind="stable")
        )
        # One row per raw level (keep the last of any repeats) so a target
        # height that equals a raw level shares its row and takes its values.
        # A separate target row there would sort before or after the raw row
        # and be left NaN at the bottom or top level (#45).
        raw = raw[~raw.index.duplicated(keep="last")]

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
        ds.attrs["notes"] = (
            f"Interpolated to {start:g}-{stop:g}m at {interval:g}m intervals."
        )

        return ds

    def plot(self):
        """
        Plot the sounding (not implemented yet).

        Raises
        ------
        NotImplementedError
            Always.

        See Also
        --------
        Sounding.interpolate : The profiles to plot with xarray, for now.
        """
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
        The merged sounding data: the levels of every sounding, one row per
        level, with each sounding's station attributes (``station``, ``time``,
        ``elevation``, ...) as columns.

    See Also
    --------
    get_soundings : With ``driver="pandas"``, merges a period's soundings.
    """
    dfs = []
    for sounding in soundings:
        df = sounding.data.copy()
        for attr in sounding._attrs:
            if hasattr(sounding, attr):
                df[attr] = getattr(sounding, attr)
        dfs.append(df)

    return pd.concat(dfs)


def _unwrap_heights(height: pd.Series, source: str) -> pd.Series:
    """
    Heights [m] with any wrap at 10 km undone, the levels in file order.

    A fall of more than 9 km from one level to the next is a wrap (9998 m,
    then 96 m): 10 km is added to that level and every level after it, and
    another 10 km after a second wrap. A smaller fall is kept as it is. A level
    without a height stays NaN and is skipped. Logs a warning that names
    ``source`` when a wrap is undone.
    """
    valid = height.dropna()
    wraps = valid.diff() < -_WRAP_FALL
    if not wraps.any():
        return height
    first = int(np.flatnonzero(wraps.to_numpy())[0])
    logger.warning(
        "%s: the heights fall from %g m to %g m between two levels, taken as a "
        "wrap at %g m; %g m is added to every level after each wrap (%d in the "
        "file).",
        source,
        valid.iloc[first - 1],
        valid.iloc[first],
        _HEIGHT_WRAP,
        _HEIGHT_WRAP,
        int(wraps.sum()),
    )
    unwrapped = height.astype(float)
    unwrapped.loc[valid.index] = valid + _HEIGHT_WRAP * wraps.cumsum()
    return unwrapped


def _first_level_height(data: pd.DataFrame) -> float:
    """
    Height [m] of the first level of a sounding that has one.

    Wyoming lists the levels from the ground up, so this is the ground at the
    station. The first level, not the lowest: a level can carry a wrong height
    (one SLC file has 15 m).
    """
    return float(data["height"].dropna().iloc[0])


#: The columns of a saved sounding that are speeds, saved in knots
_WIND_COLUMNS = ("speed", "u_wind", "v_wind")


def _winds_to_knots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wind columns of a siphon response to knots.

    siphon reports the unit of each column in ``df.units``. It sets them by its
    own version, not from the response: 0.9 and 0.10 say knots (and send
    ``elevation``), 0.11, for Wyoming's current service, says m/s. A wind
    column whose unit is missing, or is not a speed pint knows, raises a
    ``ValueError`` that names the column and the unit.
    """
    reported = getattr(df, "units", None) or {}
    df = df.copy()
    for column in _WIND_COLUMNS:
        if column not in df.columns:
            continue
        unit = reported.get(column)
        if not isinstance(unit, str) or not unit.strip():
            raise ValueError(
                f"siphon reports no unit for the sounding's {column!r} column "
                f"(units: {unit!r}); it cannot be converted to knots."
            )
        try:
            factor = float(units.Quantity(1, unit).to("knot").magnitude)
        except Exception as e:
            # Not one error type: pint raises UndefinedUnitError for a name it
            # lacks, DimensionalityError for a unit that is not a speed, and
            # AssertionError, ValueError or TypeError for a malformed string
            raise ValueError(
                f"siphon reports the sounding's {column!r} column in {unit!r}, "
                "which is not a speed unit that pint knows; it cannot be "
                "converted to knots."
            ) from e
        df[column] = df[column] * factor
    return df


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
        The date and time. A time without a time zone follows :ref:`the UTC
        rule <utc-rule>`; one with a time zone is converted to UTC. Wyoming
        reads the time as UTC, and the file is named for the UTC hour
        (``SLC_2024011512.csv`` for 12 UTC).
    dst : str
        The destination directory. Defaults to ``$LAIR_SOUNDING_DIR/<station>``.

    Returns
    -------
    str
        The path to the downloaded sounding data.

    Raises
    ------
    ValueError
        If the unit siphon reports for ``speed``, ``u_wind`` or ``v_wind`` is
        missing or is not a speed unit that pint knows. No file is written.

    Notes
    -----
    Every file is saved in the units of ``Sounding.units``, whatever
    Wyoming sent. Wyoming sends the winds in m/s today and sent knots before,
    so ``speed``, ``u_wind`` and ``v_wind`` are converted to knots from the
    unit siphon reports for each. Wyoming no longer sends the station's
    ``elevation``, so it is saved as the height of the first level, the ground.
    ``pw`` and ``station_number`` are saved only if Wyoming sends them, which
    it no longer does. A file that is already in ``dst`` is not downloaded
    again or changed.

    See Also
    --------
    download_soundings : Every sounding of a period.
    Sounding : Reads the file.

    Examples
    --------
    The 12 UTC sounding of Salt Lake City on 2024-01-15, into a folder of your
    own. It needs the network, so it is not run when the docs build:

    >>> import datetime as dt
    >>> from lair.soundings import download_sounding
    >>> download_sounding(  # doctest: +SKIP
    ...     "SLC", dt.datetime(2024, 1, 15, 12), dst="soundings/SLC"
    ... )
    'soundings/SLC/SLC_2024011512.csv'
    """
    date = _naive_utc(date)
    if dst is None:
        dst = os.path.join(get_data_dir(SOUNDING_DIR_ENV), station)
    os.makedirs(dst, exist_ok=True)

    path = os.path.join(dst, f"{station}_{date:%Y%m%d%H}.csv")
    if not os.path.exists(path):
        logger.info("Downloading %s on %s...", station, f"{date:%Y-%m-%d %H:%M}")
        df = _winds_to_knots(WyomingUpperAir.request_data(date, station))
        if "elevation" not in df.columns:
            df["elevation"] = _first_level_height(df)
        df.to_csv(path, index=False)
    else:
        logger.debug("%s on %s already exists.", station, f"{date:%Y-%m-%d %H:%M}")

    return path


def download_soundings(station, start, end, dst=None, months=None):
    """
    Download upper air soundings from the Wyoming archive.

    Each one is saved as :func:`download_sounding` saves it, in the units of
    ``Sounding.units``.

    Parameters
    ----------
    station : str
        The 3-letter station identifier (e.g. ``'SLC'``).
    start : datetime
        The start date and time. Soundings are at 00 and 12 UTC, so the first
        one requested is the first of those at or after ``start``. A time
        without a time zone follows :ref:`the UTC rule <utc-rule>`; one with a
        time zone is converted to UTC.
    end : datetime
        The end date and time (inclusive), read like ``start``.
    dst : str
        The destination directory. Defaults to ``$LAIR_SOUNDING_DIR/<station>``.
    months : list
        The months to download (1 to 12), by default every month.

    Raises
    ------
    ValueError
        If the unit siphon reports for a wind is not one :func:`download_sounding`
        can convert, or Wyoming answers with an error other than "No data
        available" (a sounding Wyoming does not have is logged and skipped).
    requests.exceptions.HTTPError
        For an HTTP error other than Wyoming's "Please try again later", after
        which the same sounding is asked for again 2 seconds later.

    See Also
    --------
    get_soundings : Reads them, with ``download=True`` downloading first.
    download_sounding : One sounding.

    Examples
    --------
    The soundings of January 2024, at 00 and 12 UTC. It needs the network, so
    it is not run when the docs build:

    >>> from lair.soundings import download_soundings
    >>> download_soundings(  # doctest: +SKIP
    ...     "SLC", "2024-01-01", "2024-01-31 12:00", dst="soundings/SLC"
    ... )
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
    download=False,
):
    """
    Get upper air soundings from the Wyoming archive.

    Reads the soundings that are in the station's folder. It downloads only
    when asked: see :ref:`soundings-download`.

    Parameters
    ----------
    station : str
        The 3-letter station identifier (e.g. ``'SLC'``). Names the
        subdirectory of ``$LAIR_SOUNDING_DIR`` and the file prefix.
    start : datetime | str, optional
        The start date and time (inclusive). A time without a time zone follows
        :ref:`the UTC rule <utc-rule>`; one with a time zone is converted to
        UTC.
    end : datetime | str, optional
        The end date and time (inclusive).
    sounding_dir : str
        The directory containing the station's sounding files. Defaults to
        ``$LAIR_SOUNDING_DIR/<station>``.
    months : list[int], optional
        Only use soundings from these months (with ``download``, only download
        those).
    driver : {'xarray', 'pandas'}, optional
        Return interpolated profiles as an xarray Dataset (default), or the
        merged raw soundings as a pandas DataFrame. The profiles share one set
        of heights (:ref:`soundings-heights` says which).
    download : bool, optional
        Download the soundings the folder lacks, from ``start`` to ``end``,
        before reading them. The default, False, downloads nothing.

    Returns
    -------
    xr.Dataset | pd.DataFrame
        The sounding data.

    Raises
    ------
    ValueError
        If no sounding is found for the period (the message says to pass
        ``download=True`` when it is False); if ``download`` is True without
        ``start`` and ``end``; or for an invalid ``driver``.

    See Also
    --------
    download_soundings : Download soundings without reading them.

    Examples
    --------
    Two made-up soundings in a folder, read as one dataset on the same
    heights (:doc:`/examples/soundings` reads real ones):

    >>> import pathlib
    >>> import tempfile
    >>> import pandas as pd
    >>> from lair.soundings import get_soundings
    >>> folder = pathlib.Path(tempfile.mkdtemp())
    >>> for hour, ground in [("2024011500", -8.0), ("2024011512", -5.0)]:
    ...     pd.DataFrame(
    ...         {
    ...             "height": [1289.0, 5100.0],
    ...             "temperature": [ground, -30.0],
    ...             "station": "SLC",
    ...             "time": pd.to_datetime(hour, format="%Y%m%d%H"),
    ...             "elevation": 1289.0,
    ...         }
    ...     ).to_csv(folder / f"SLC_{hour}.csv", index=False)
    >>> ds = get_soundings(sounding_dir=str(folder))
    >>> ds.temperature.sizes
    Frozen({'time': 2, 'height': 372})
    >>> ds.temperature.sel(height=1289.0).values
    array([-8., -5.])
    """
    # File times are naive UTC, so compare against naive UTC bounds
    start = _naive_utc(start).to_pydatetime() if start is not None else None
    end = _naive_utc(end).to_pydatetime() if end is not None else None

    if sounding_dir is None:
        sounding_dir = os.path.join(get_data_dir(SOUNDING_DIR_ENV), station)

    if download:
        if start is None or end is None:
            raise ValueError("start and end must be specified to download soundings.")
        download_soundings(station, start, end, sounding_dir, months)

    files = os.listdir(sounding_dir) if os.path.isdir(sounding_dir) else []

    soundings = []
    for file in files:
        # Skip anything that isn't a <station>_<YYYYmmddHH>.csv sounding
        try:
            date = dt.datetime.strptime(file.split("_")[1].split(".")[0], "%Y%m%d%H")
        except (IndexError, ValueError):
            continue

        # Skip files that don't match the date range
        if start and date < start:
            continue
        if end and date > end:
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
        message = f"No soundings found in {sounding_dir} for the requested period."
        if not download:
            message += " Pass download=True (with start and end) to download them."
        raise ValueError(message)

    if driver in ["xarray", "nc"]:
        # One set of heights for every sounding, from the lowest surface, so
        # the profiles line up and stay evenly spaced
        lowest = min(sounding._surface_height() for sounding in soundings)
        data = xr.concat(
            [sounding.interpolate(start=lowest) for sounding in soundings], dim="time"
        ).sortby("time")
    elif driver in ["pandas", "csv"]:
        data = merge(soundings)
    else:
        raise ValueError(f"Invalid driver: {driver}")

    return data
