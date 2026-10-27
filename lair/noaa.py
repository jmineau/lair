"""
NOAA greenhouse gas data: CarbonTracker and NOAA GML's measurement records.

:class:`CarbonTrackerCH4` and :class:`CarbonTrackerCO2` download and read the
mole fractions of NOAA's CarbonTracker model, and sample them at points, such as
the ends of STILT trajectories, for a background
(:meth:`~CarbonTracker.background`). :class:`GMLData` downloads and reads NOAA
GML's records of one gas at one site, event by event or as monthly means. The
data roots are ``$LAIR_CARBONTRACKER_DIR`` and ``$LAIR_GML_DIR`` (see
:ref:`data-locations`), and the cache is in ``$LAIR_CACHE_DIR``.
:doc:`/examples/carbontracker` samples CarbonTracker at four points.
"""

import datetime as dt
import re
from abc import ABCMeta
from collections.abc import Sequence
from concurrent.futures.process import BrokenProcessPool
from contextlib import nullcontext
from functools import cached_property
from pathlib import Path
from typing import Literal, cast

import dask
import numpy as np
import pandas as pd
import xarray as xr
from xarray.backends.locks import HDF5_LOCK, NETCDFC_LOCK, combine_locks

from lair.config import get_cache_dir, get_data_dir
from lair.parallel import _allowed_cpus
from lair.records import Cacher, ftp_download, list_files

#: Environment variable holding the CarbonTracker data root
#:
#: .. seealso:: :ref:`data-locations`, which lists every variable;
#:    :ref:`carbontracker-data` shows the folder.
CARBONTRACKER_DIR_ENV = "LAIR_CARBONTRACKER_DIR"

#: Environment variable holding the NOAA GML data root
#:
#: .. seealso:: :ref:`data-locations`, which lists every variable;
#:    :class:`GMLData`'s Notes show the folder.
GML_DIR_ENV = "LAIR_GML_DIR"


def _open_mfdataset(files, preprocess=None, parallel: bool = False) -> xr.Dataset:
    """
    Open ``files`` as one dataset, combined the way xarray does it today.

    xarray is changing the defaults of the four arguments named here (it warns
    since 2025.08), and it warns that the new ``data_vars`` changes what the
    CarbonTracker mole fractions load. Naming today's values keeps the result
    the same on any xarray, and keeps the warning quiet.

    With ``parallel=True`` the files are opened in worker processes, one for
    each CPU this process may use (no more than there are files), not on
    dask's default threads. netCDF-C and HDF5 are not thread-safe, and xarray
    holds its lock only while it opens a file, not while it reads the file's
    variables and attributes, so opening files on threads at once crashes the
    process or fails with an HDF error (issue #121). Worker processes start by
    importing the caller's script again, so a script needs the
    ``if __name__ == "__main__":`` guard.

    The workers are given this process's netCDF lock. A file opened in a
    worker comes back with the lock it was opened with, and each process's own
    lock is a different one, so without this the files from different workers
    would be read at once, and crash, when the data is loaded here.
    """
    if parallel:
        # The lock xarray's netCDF4 backend uses for local files
        lock = combine_locks([NETCDFC_LOCK, HDF5_LOCK])
        workers = max(1, min(len(files), _allowed_cpus()))
        # Takes effect as it is made; the `with` below restores the old config
        scheduler = dask.config.set(scheduler="processes", num_workers=workers)
    else:
        lock = None  # xarray's own choice
        scheduler = nullcontext()
    try:
        with scheduler:
            return xr.open_mfdataset(
                files,
                preprocess=preprocess,
                parallel=parallel,
                lock=lock,
                combine="by_coords",
                compat="no_conflicts",
                data_vars="all",
                coords="different",
                join="outer",
            )
    except BrokenProcessPool as e:
        raise RuntimeError(
            "A worker process opening the files stopped. In a script, put the "
            'code that runs under `if __name__ == "__main__":` (each worker '
            "imports the script again), or pass parallel_parse=False to open "
            "the files one at a time."
        ) from e


#: The day a CarbonTracker file covers: the date its name ends with
_DAY_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})\.nc$")


def _check_one_file_per_day(files: Sequence[str]) -> None:
    """
    Raise if two of the daily molefraction ``files`` cover the same day.

    The day is the date a file name ends with (``..._2015-01-01.nc``), so no
    file is opened. A name that does not end with a date is not checked.

    Raises
    ------
    ValueError
        Names the first day with more than one file, two of its files, and how
        many days have more than one.
    """
    by_day: dict[str, list[str]] = {}
    for file in sorted(files):
        match = _DAY_IN_NAME.search(Path(file).name)
        if match:
            by_day.setdefault(match.group(1), []).append(file)
    repeated = sorted(day for day, names in by_day.items() if len(names) > 1)
    if not repeated:
        return
    day = repeated[0]
    first, second = by_day[day][:2]
    message = (
        f"Two molefraction files cover {day}: {first} and {second}. Reading "
        "both would read every value twice. Keep one file for each day in the "
        "folder, for example by moving the files of an older release out of it."
    )
    if len(repeated) > 1:
        message += (
            f" {len(repeated)} days in all have more than one file, from "
            f"{day} to {repeated[-1]}."
        )
    raise ValueError(message)


def _cache_file(*parts: str) -> Path:
    """
    Path of a file in the lair cache, with its folder made.

    The cache folder (``LAIR_CACHE_DIR``, default ``~/.cache/lair``) is read
    and made here, the first time something is cached, and not when lair is
    imported.

    Parameters
    ----------
    *parts : str
        Path of the file below the cache folder.

    Returns
    -------
    Path
        The cache file (not created).

    Raises
    ------
    OSError
        If the folder cannot be made (a :exc:`PermissionError` when the
        location is read-only). The message names the folder and says to set
        ``LAIR_CACHE_DIR``.
    """
    cache_dir = get_cache_dir()
    path = cache_dir.joinpath(*parts)
    try:
        # exist_ok: many processes (e.g. SLURM tasks) may make it at once
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        # OSError(errno, ...) gives the matching subclass, e.g. PermissionError
        raise OSError(
            e.errno,
            f"Cannot make the cache folder {path.parent} ({e.strerror}). Set "
            f"the LAIR_CACHE_DIR environment variable (now {cache_dir}) to a "
            "folder you can write to, or pass cache=False.",
        ) from e
    return path


class CarbonTracker(metaclass=ABCMeta):
    """
    NOAA CarbonTracker

    The base of :class:`CarbonTrackerCH4` and :class:`CarbonTrackerCO2`, which
    are the classes to make (or :meth:`from_version`).

    Parameters
    ----------
    version : str
        The version of CarbonTracker data to download.
        Visit https://gml.noaa.gov/aftp/products/carbontracker/ to see available versions.
    carbon_tracker_directory : str, optional
        The CarbonTracker data root, by default ``$LAIR_CARBONTRACKER_DIR``.
    cache : bool, optional
        Whether to cache the data, by default True. The cache is a file in
        ``LAIR_CACHE_DIR`` (default ``~/.cache/lair``), and the folder is
        made the first time the data is read, so ``cache=False`` needs no
        writable folder.

    Raises
    ------
    TypeError
        If made as ``CarbonTracker`` itself, which has no gas.

    Attributes
    ----------
    specie : Literal['ch4', 'co2']
        The greenhouse gas specie.
    version : str
        The CarbonTracker version.
    directory : str
        The directory for the version: ``<root>/<specie>/<version>``.
    cache : bool
        Whether to cache the data (in ``LAIR_CACHE_DIR``).

    Methods
    -------
    get_specie_from_version(version)
        Get the specie from the version.
    from_version(version, carbon_tracker_directory=None)
        Create a CarbonTracker object from the version.
    download(sub_dirs=['fluxes', 'molefractions'], pattern=None)
        Download CarbonTracker data from the NOAA GML FTP server.

    See Also
    --------
    CarbonTrackerCH4 : CarbonTracker-CH4.
    CarbonTrackerCO2 : CarbonTracker for CO2.
    """

    specie: Literal["ch4", "co2"]
    #: Units of the molefraction field ('ppb' for CH4, 'ppm' for CO2)
    #:
    #: .. seealso:: :meth:`CarbonTracker.sample`, whose ``ct_<specie>_<units>``
    #:    column is in them.
    units: Literal["ppb", "ppm"]

    def __init__(
        self,
        version: str,
        carbon_tracker_directory: str | Path | None = None,
        cache: bool = True,
    ):
        """Initialize a CarbonTracker; the class docstring describes the parameters."""
        if not hasattr(self, "specie"):
            raise TypeError(
                "Use CarbonTrackerCH4 / CarbonTrackerCO2, or "
                "CarbonTracker.from_version(version)."
            )
        self.version = version

        carbon_tracker_directory = get_data_dir(
            CARBONTRACKER_DIR_ENV, carbon_tracker_directory
        )
        self.directory = carbon_tracker_directory / self.specie / version

        self.cache = cache

    def __repr__(self):
        return f"{self.__class__.__name__}(version={self.version}, directory={self.directory})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.version})"

    @staticmethod
    def get_specie_from_version(version: str) -> Literal["ch4", "co2"]:
        """
        Get the specie from the version.

        Parameters
        ----------
        version : str
            The version of CarbonTracker data.

        Returns
        -------
        Literal['ch4', 'co2']
            The specie: ``'ch4'`` if the version's name has "ch4" in it (in any
            case), else ``'co2'``.

        See Also
        --------
        CarbonTracker.from_version : Makes the class of that gas.

        Examples
        --------
        >>> from lair.noaa import CarbonTracker
        >>> CarbonTracker.get_specie_from_version("CT-CH4-2025")
        'ch4'
        >>> CarbonTracker.get_specie_from_version("CT2019B")
        'co2'
        """
        return "ch4" if "ch4" in version.lower() else "co2"

    @staticmethod
    def from_version(version: str, carbon_tracker_directory: str | None = None):
        """
        Create a CarbonTracker object from the version.

        Parameters
        ----------
        version : str
            The version of CarbonTracker data to download.
            Visit https://gml.noaa.gov/aftp/products/carbontracker/ to see available versions.
        carbon_tracker_directory : str, optional
            The CarbonTracker data root, by default ``$LAIR_CARBONTRACKER_DIR``.

        Returns
        -------
        CarbonTracker
            The CarbonTracker object: a :class:`CarbonTrackerCH4` or a
            :class:`CarbonTrackerCO2`, with the other arguments at their defaults.

        See Also
        --------
        CarbonTracker.get_specie_from_version : How the gas is told.

        Examples
        --------
        Nothing is read until the data is used:

        >>> from lair.noaa import CarbonTracker
        >>> print(CarbonTracker.from_version("CT2019B", "carbontracker"))
        CarbonTrackerCO2(CT2019B)
        """
        specie = CarbonTracker.get_specie_from_version(version)
        if specie == "co2":
            return CarbonTrackerCO2(version, carbon_tracker_directory)
        elif specie == "ch4":
            return CarbonTrackerCH4(version, carbon_tracker_directory)
        else:
            raise ValueError("Invalid specie")

    def download(
        self,
        sub_dirs: Sequence[str] | None = ("fluxes", "molefractions"),
        pattern: str | None = None,
    ):
        """
        Download CarbonTracker data from the NOAA GML FTP server.

        Parameters
        ----------
        sub_dirs : list of str, optional
            The subdirectories to download data from, by default ['fluxes', 'molefractions'].
            If None, download the entire version data.
        pattern : str, optional
            Only download matching files: a glob against the remote path if it
            has wildcards (e.g. ``'*2015-06*'``), else a substring. By default None.

        See Also
        --------
        lair.records.ftp_download : Does the download.

        Notes
        -----
        The files go into ``directory``, in the server's folder layout below the
        version. Without ``pattern`` every file of the folders is downloaded:
        the CT-CH4-2025 mole fractions of 2015 to 2023 alone are 127 GB.

        Examples
        --------
        One day of mole fractions, as :doc:`/examples/carbontracker` downloads
        it. It needs the network (about five minutes), so it is not run when the
        docs build:

        >>> from lair.noaa import CarbonTrackerCH4
        >>> ct = CarbonTrackerCH4(carbon_tracker_directory="carbontracker")
        >>> ct.download(  # doctest: +SKIP
        ...     sub_dirs=["molefractions"], pattern="*2020-01-15*"
        ... )
        """
        host = "ftp.gml.noaa.gov"
        parent = "/products/carbontracker"

        # Build list of remote paths to download
        path = f"{parent}/{self.specie}/{self.version}"
        if sub_dirs is None:
            paths = [path]
        else:
            paths = [f"{path}/{sub_dir}" for sub_dir in sub_dirs]

        # Download the data
        ftp_download(host, paths, str(self.directory), prefix=path, pattern=pattern)
        return None

    # --- sampling the concentration field at trajectory points ---------------

    @property
    def molefractions_dir(self) -> Path:
        """
        Directory of the molefraction (concentration-field) files.

        See Also
        --------
        CarbonTrackerCH4.molefractions : Reads them.

        Examples
        --------
        >>> from lair.noaa import CarbonTrackerCH4
        >>> ct = CarbonTrackerCH4(carbon_tracker_directory="carbontracker")
        >>> ct.molefractions_dir.as_posix()
        'carbontracker/ch4/CT-CH4-2025/molefractions'
        """
        return self.directory / "molefractions"

    @staticmethod
    def _preprocess_molefractions(ds: xr.Dataset) -> xr.Dataset:
        """
        Label each time step of a molefraction file by its period's center.

        Each step is a mean over a period (3 hours in the files lair reads),
        and the file's ``time_components`` (year, month, day, hour, minute,
        second, UTC) hold the center of it: 01:30, 04:30, ... in every
        release. ``time`` does not always: CT-CH4-2023 has none, and in
        CT-CH4-2025 it is the end of each period (03:00, 06:00, ...), its
        ``time_bounds`` 1.5 hours late too, though the file's ``notes`` say
        the times are centers. So when a file has ``time_components`` the
        time comes from them (as ``datetime64[ns]``, the resolution xarray
        decodes ``time`` to), ``time_bounds`` move with it, and
        ``time_components`` are dropped.
        """
        if "time_components" not in ds:
            return ds
        centers = np.array(
            [dt.datetime(*row) for row in ds["time_components"].values],
            dtype="datetime64[ns]",
        )
        if "time_bounds" in ds and np.issubdtype(ds["time"].dtype, np.datetime64):
            shift = xr.DataArray(ds["time"].values - centers, dims="time")
            ds = ds.assign(time_bounds=ds["time_bounds"] - shift)
        return ds.assign_coords(time=centers).drop_vars("time_components")

    def _molefraction_file_for_date(self, date) -> Path | None:
        """Molefraction file covering a UTC ``date`` (newest match), if present."""
        date = pd.Timestamp(date).date()
        month_dir = self.molefractions_dir / f"{date.year}" / f"{date.month:02d}"
        matches = sorted(month_dir.glob(f"*molefrac*_{date}.nc"))
        return matches[-1] if matches else None

    @staticmethod
    def _sample_field(
        points: pd.DataFrame, ds: xr.Dataset, variable: str, units: str = "ppb"
    ) -> pd.DataFrame:
        """
        Evaluate ``ds[variable]`` at each row's (datetime, lati, long, zagl).

        Vertical placement uses the geopotential-height (``gph``) layer bounds:
        the particle's height-above-ground is added to the surface gph and the
        enclosing model level selected. Returns the input rows (index reset) with
        ``ct_<variable>_<units>`` plus the CT cell/level actually used. Points
        outside the horizontal grid (e.g. a regional CO2 grid) are NaN.
        """
        dim = "points"
        t = (
            pd.DatetimeIndex(pd.to_datetime(points["datetime"], utc=True))
            .tz_convert(None)
            .values
        )
        sel = ds.sel(
            time=xr.DataArray(t, dims=dim),
            latitude=xr.DataArray(points["lati"].to_numpy(), dims=dim),
            longitude=xr.DataArray(points["long"].to_numpy(), dims=dim),
            method="nearest",
        )
        zagl = xr.DataArray(points["zagl"].to_numpy(), dims=dim)
        gph = sel["gph"]
        gph_lo = (
            gph.isel(boundary=slice(None, -1))
            .rename({"boundary": "level"})
            .assign_coords(level=ds["level"].values)
        )
        gph_hi = (
            gph.isel(boundary=slice(1, None))
            .rename({"boundary": "level"})
            .assign_coords(level=ds["level"].values)
        )
        z_asl = zagl + gph_lo.isel(level=0)
        mask = (z_asl >= gph_lo) & (z_asl < gph_hi)

        # Nearest-cell selection would clamp points outside the grid to its edge
        in_domain = np.ones(len(points), dtype=bool)
        for coord, col in (("latitude", "lati"), ("longitude", "long")):
            centers = ds[coord].values
            if centers.size > 1:
                half = np.abs(np.diff(centers)).max() / 2
                p = points[col].to_numpy()
                in_domain &= (p >= centers.min() - half) & (p <= centers.max() + half)
        has_layer = mask.any(dim="level") & xr.DataArray(in_domain, dims=dim)
        values = (
            sel[variable].where(mask).max(dim="level", skipna=True).where(has_layer)
        )
        level_idx = mask.argmax(dim="level")
        used_level = (
            xr.DataArray(ds["level"].values, dims="level")
            .isel(level=level_idx)
            .where(has_layer)
        )

        out = points.reset_index(drop=True).copy()
        out[f"ct_{variable}_{units}"] = values.load().to_numpy()
        out["ct_time"] = pd.to_datetime(sel["time"].load().to_numpy())
        out["ct_latitude"] = sel["latitude"].load().to_numpy()
        out["ct_longitude"] = sel["longitude"].load().to_numpy()
        out["ct_level"] = used_level.load().to_numpy()
        return out

    def sample(self, points: pd.DataFrame) -> pd.DataFrame:
        """
        Sample the molefraction field at ``points``.

        ``points`` is a DataFrame with columns ``datetime`` (UTC sample time),
        ``lati``, ``long``, ``zagl``, as in a PYSTILT particle table (for
        example ``particles.stilt.endpoints()``); any other columns (e.g.
        ``indx``, ``receptor``) are carried through for downstream grouping. Returns one row
        per input point with ``ct_<specie>_<units>`` and the CT cell/level used.
        Points whose UTC date has no molefraction file are dropped.

        Each point takes the time step whose center is nearest: the steps are
        3-hour means labeled by their centers (``ct_time``: 01:30, 04:30, ...
        UTC), so a point takes the mean of the period it is in, and a point on
        the boundary of two periods (03:00) takes the later one.

        Parameters
        ----------
        points : pd.DataFrame
            The points, with the columns ``datetime`` (UTC), ``lati`` and
            ``long`` (degrees) and ``zagl`` (meters above ground).

        Returns
        -------
        pd.DataFrame
            The points, with ``ct_<specie>_<units>`` (``ct_ch4_ppb``) and the
            cell, time and level it came from (``ct_latitude``,
            ``ct_longitude``, ``ct_time``, ``ct_level``). A point outside the
            grid gets NaN.

        See Also
        --------
        CarbonTracker.background : Their mean.

        Examples
        --------
        :ref:`The CarbonTracker example <carbontracker-data>` samples four
        points west of Salt Lake City; it reads the folder of one downloaded
        day::

            points = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2020-01-15 17:00"] * 2),  # UTC
                    "lati": [41.5, 39.5],
                    "long": [-114.5, -114.5],
                    "zagl": [100.0, 100.0],
                }
            )
            ct.sample(points)[["lati", "ct_ch4_ppb", "ct_time", "ct_level"]]
        """
        points = points.reset_index(drop=True)
        if points.empty:
            return points
        dates = pd.DatetimeIndex(
            pd.to_datetime(points["datetime"], utc=True)
        ).normalize()
        file_for = {d: self._molefraction_file_for_date(d) for d in dates.unique()}
        keep = dates.map(lambda d: file_for[d] is not None).to_numpy()
        points = points.loc[keep].reset_index(drop=True)
        files = sorted({f for f in file_for.values() if f is not None})
        if points.empty or not files:
            return points
        with _open_mfdataset(files, preprocess=self._preprocess_molefractions) as ds:
            return self._sample_field(points, ds, self.specie, self.units)

    def background(self, points: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
        """
        Background mole fraction [ppm]: mean of the sampled field over ``points``.

        Samples the field at every point (e.g. trajectory endpoints) and averages
        over them. With ``by`` (e.g. ``'receptor'``) the mean and 1-sigma spread
        are returned per group; otherwise a single-row summary. Output is ppm
        (CT-CH4 mole fractions are stored in ppb, CO2 in ppm).

        Parameters
        ----------
        points : pd.DataFrame
            The points, as :meth:`sample` takes them.
        by : str, optional
            A column of ``points`` to group by, such as ``'receptor'``. By
            default None: one row for all the points.

        Returns
        -------
        pd.DataFrame
            ``background_ppm``, ``sigma_ppm`` (the standard deviation over the
            points) and ``n_endpoints`` (the points that had a value), one row
            per group.

        See Also
        --------
        CarbonTracker.sample : The value at each point.

        Examples
        --------
        With the points of :meth:`sample`, as :ref:`the CarbonTracker example
        <carbontracker-background>` shows::

            ct.background(points)  # one row
            ct.background(points, by="receptor")  # one row per receptor
        """
        sampled = self.sample(points)
        col = f"ct_{self.specie}_{self.units}"
        if col not in sampled:
            # sample() returns no rows (and no ct_ columns) when no
            # molefraction file covers the points
            sampled[col] = pd.Series(dtype=float)
        ppm = sampled[col] / (1000.0 if self.units == "ppb" else 1.0)
        if by is None:
            return pd.DataFrame(
                {
                    "background_ppm": [ppm.mean()],
                    "sigma_ppm": [ppm.std()],
                    "n_endpoints": [int(ppm.notna().sum())],
                }
            )
        grouped = ppm.groupby(sampled[by].to_numpy())
        return pd.DataFrame(
            {
                "background_ppm": grouped.mean(),
                "sigma_ppm": grouped.std(),
                "n_endpoints": grouped.count(),
            }
        )


class CarbonTrackerCH4(CarbonTracker):
    """
    NOAA CarbonTracker-CH4

    Mole fractions are in ppb. :ref:`The CarbonTracker example
    <carbontracker-data>` shows the folder.

    Each parameter is also an attribute of the same name.

    Parameters
    ----------
    version : str, optional
        The version, by default ``'CT-CH4-2025'``.
    carbon_tracker_directory : str, optional
        The CarbonTracker data root, by default ``$LAIR_CARBONTRACKER_DIR``.
    cache : bool, optional
        Whether to cache the data, by default True (see :class:`CarbonTracker`).
    parallel_parse : bool, optional
        Whether ``molefractions`` opens the files in worker processes, one for
        each CPU this process may use (default True), or one at a time. The
        workers import the calling script again, so a script that reads
        ``molefractions`` needs the ``if __name__ == "__main__":`` guard.

    Methods
    -------
    calc_molefractions_pressure(molefractions)
        Calculate the pressure at each level in the molefractions Dataset.

    See Also
    --------
    CarbonTrackerCO2 : The same for CO2.

    Examples
    --------
    Nothing is read until the data is used:

    >>> from lair.noaa import CarbonTrackerCH4
    >>> print(CarbonTrackerCH4(carbon_tracker_directory="carbontracker"))
    CarbonTrackerCH4(CT-CH4-2025)
    """

    specie = "ch4"
    units = "ppb"

    def __init__(
        self,
        version="CT-CH4-2025",
        carbon_tracker_directory=None,
        cache=True,
        parallel_parse=True,
    ):
        super().__init__(version, carbon_tracker_directory, cache)
        self.parallel_parse = parallel_parse

    @cached_property
    def molefractions(self) -> xr.Dataset:
        """
        Molefractions Dataset of every file of the version. Cached property.

        Raises
        ------
        ValueError
            If two files cover one day, such as the same day from two
            releases (their names end with the same ``YYYY-MM-DD.nc``). The
            message names the day and both files. The files are not opened,
            and nothing is cached.

        See Also
        --------
        CarbonTracker.sample : Reads only the days of the points.
        CarbonTrackerCH4.calc_molefractions_pressure : Adds the pressure of
            each level.

        Notes
        -----
        Each time step is labeled by the center of its 3-hour period (01:30,
        04:30, ... UTC). It opens every file of the version; with
        ``cache=True`` a later read loads the cache instead.
        """
        path = self.directory / "molefractions"

        files = list_files(str(path), "*nc", full_names=True, recursive=True)
        _check_one_file_per_day(files)

        if self.cache:
            cache_file = _cache_file(
                "carbontracker", self.specie, self.version, "molefractions.pkl"
            )
            open_mfdataset = Cacher(_open_mfdataset, str(cache_file))
        else:
            open_mfdataset = _open_mfdataset
        ds = open_mfdataset(
            files,
            preprocess=CarbonTrackerCH4._preprocess_molefractions,
            parallel=self.parallel_parse,
        )
        return ds

    @staticmethod
    def calc_molefractions_pressure(molefractions) -> xr.Dataset:
        """
        Calculate the pressure at each level in the molefractions Dataset.

        Parameters
        ----------
        molefractions : xr.Dataset
            The molefractions Dataset.

        Returns
        -------
        xr.Dataset
            The molefractions Dataset with the pressure calculated: a variable
            ``P``, in hPa, from the hybrid coefficients ``at`` and ``bt`` and the
            surface pressure (``at + bt * surf_pressure``). The Dataset passed is
            changed too.

        See Also
        --------
        CarbonTrackerCH4.molefractions : The Dataset it takes.

        Examples
        --------
        Two levels over one cell, with a surface pressure of 850 hPa:

        >>> import xarray as xr
        >>> from lair.noaa import CarbonTrackerCH4
        >>> mf = xr.Dataset(
        ...     {
        ...         "at": ("boundary", [0.0, 5000.0]),
        ...         "bt": ("boundary", [1.0, 0.5]),
        ...         "surf_pressure": 85000.0,
        ...     }
        ... )
        >>> CarbonTrackerCH4.calc_molefractions_pressure(mf)["P"].values
        array([850., 475.])
        """
        molefractions["P"] = (
            molefractions.at + molefractions.bt * molefractions.surf_pressure
        )
        molefractions["P"] /= 100  # Convert to hPa
        molefractions["P"].attrs = {
            "long_name": "Pressure",
            "units": "hPa",
            "comment": "Calculated from hybrid sigma-pressure coefficients and surface pressure.",
        }
        return molefractions


class CarbonTrackerCO2(CarbonTracker):
    """
    NOAA CarbonTracker (CO2)

    Mole fractions are in ppm (micromol mol-1). Releases such as CT2019B keep
    the daily molefraction files flat in ``molefractions/co2_total/``, one file
    per grid (``CT2019B.molefrac_nam1x1_2015-06-01.nc``, ``..._glb3x2_...``).

    Each parameter is also an attribute of the same name.

    Parameters
    ----------
    version : str
        The version, such as ``'CT2019B'``.
    carbon_tracker_directory : str, optional
        The CarbonTracker data root, by default ``$LAIR_CARBONTRACKER_DIR``.
    cache : bool, optional
        Whether to cache the data, by default True (see :class:`CarbonTracker`).
    grid : str, optional
        The molefraction grid to read, e.g. ``'nam1x1'`` (North America,
        1x1 deg, the default) or ``'glb3x2'`` (global, 3x2 deg). Points outside
        the grid sample as NaN.
    parallel_parse : bool, optional
        Whether ``molefractions`` opens the files in worker processes, one for
        each CPU this process may use (default True), or one at a time. The
        workers import the calling script again, so a script that reads
        ``molefractions`` needs the ``if __name__ == "__main__":`` guard.

    See Also
    --------
    CarbonTrackerCH4 : The same for methane.

    Examples
    --------
    Nothing is read until the data is used:

    >>> from lair.noaa import CarbonTrackerCO2
    >>> ct = CarbonTrackerCO2("CT2019B", "carbontracker", grid="glb3x2")
    >>> ct.molefractions_dir.as_posix()
    'carbontracker/co2/CT2019B/molefractions/co2_total'
    """

    specie = "co2"
    units = "ppm"

    def __init__(
        self,
        version: str,
        carbon_tracker_directory=None,
        cache=True,
        grid: str = "nam1x1",
        parallel_parse=True,
    ):
        super().__init__(version, carbon_tracker_directory, cache)
        self.grid = grid
        self.parallel_parse = parallel_parse

    @property
    def molefractions_dir(self) -> Path:
        """
        Directory of the total-CO2 molefraction files.

        See Also
        --------
        CarbonTrackerCO2.molefractions : Reads them.
        """
        return self.directory / "molefractions" / "co2_total"

    def _molefraction_file_for_date(self, date) -> Path | None:
        """Molefraction file on ``self.grid`` covering a UTC ``date``, if present."""
        date = pd.Timestamp(date).date()
        matches = sorted(
            self.molefractions_dir.glob(f"*molefrac_{self.grid}_{date}.nc")
        )
        return matches[-1] if matches else None

    @cached_property
    def molefractions(self) -> xr.Dataset:
        """
        Molefractions Dataset on ``self.grid`` (lazy). Cached property.

        Raises
        ------
        FileNotFoundError
            If the folder has no file on ``self.grid``.
        ValueError
            If two files on ``self.grid`` cover one day, such as the same day
            from two releases (their names end with the same
            ``YYYY-MM-DD.nc``). The message names the day and both files. The
            files are not opened, and nothing is cached.

        See Also
        --------
        CarbonTracker.sample : Reads only the days of the points.
        CarbonTrackerCH4.molefractions : The same for methane.
        """
        files = sorted(
            str(f) for f in self.molefractions_dir.glob(f"*molefrac_{self.grid}_*.nc")
        )
        if not files:
            raise FileNotFoundError(
                f"No {self.grid} molefraction files in {self.molefractions_dir}"
            )
        _check_one_file_per_day(files)

        if self.cache:
            cache_file = _cache_file(
                "carbontracker",
                self.specie,
                self.version,
                f"molefractions_{self.grid}.pkl",
            )
            open_mfdataset = Cacher(_open_mfdataset, str(cache_file))
        else:
            open_mfdataset = _open_mfdataset
        return open_mfdataset(
            files,
            preprocess=self._preprocess_molefractions,
            parallel=self.parallel_parse,
        )

    @staticmethod
    def _preprocess_molefractions(ds: xr.Dataset) -> xr.Dataset:
        # The times as for CH4; decimal_date repeats them as fractional years
        ds = CarbonTracker._preprocess_molefractions(ds)
        return ds.drop_vars("decimal_date", errors="ignore")


class GMLData:
    """
    NOAA GML Data

    One file of NOAA GML's measurements of one gas at one site: every sample
    (``frequency='event'``) or monthly means (``'month'``).

    Each parameter is also an attribute of the same name.

    Parameters
    ----------
    specie : str
        The greenhouse gas specie, in lower case as in NOAA's file names
        (``'ch4'``).
    site : str
        The site where the flask samples were collected, its code in lower
        case (``'uta'``).
    platform : str, optional
        The platform where the flask samples were collected, by default 'surface'.
    sample_type : str, optional
        The sample type, by default 'flask'. Use 'pfp' for Portable Flask
        Package data.
    lab_id : int, optional
        The lab ID, by default 1.
    measurement_group : str, optional
        The measurement group, by default 'ccgg'.
    frequency : str, optional
        The frequency of the measurements, by default 'event'.
    driver : str, optional
        The driver to use to read the data, by default 'pandas': a DataFrame
        from the ``.txt`` file, or ``'xarray'``: a Dataset from the ``.nc`` file.
    gml_dir : str, optional
        The NOAA GML data root, by default ``$LAIR_GML_DIR``.

    Attributes
    ----------
    directory : str
        The directory for the data.
    filename : str
        The filename for the data.
    filepath : str
        The filepath for the data.
    data : pd.DataFrame | xr.Dataset
        The data.
    file_template : str
        The template for the filename.
    driver_ext : dict
        The driver extensions.

    Notes
    -----
    The file is read from, and ``download()`` saves it to, one folder per
    species and sample type in the data root. ``specie`` and ``site`` are
    names in the file name, so spell them as NOAA does, in lower case
    (``'ch4'``, ``'uta'``)::

        $LAIR_GML_DIR/
          ch4/
            flask/
              ch4_uta_surface-flask_1_ccgg_event.txt
              ch4_uta_surface-flask_1_ccgg_month.txt
            pfp/
              ch4_mbo_surface-pfp_1_ccgg_event.txt

    Each name is ``{specie}_{site}_{platform}-{sample_type}_{lab_id}_{measurement_group}_{frequency}``
    and ``.txt`` for ``driver='pandas'`` or ``.nc`` for ``driver='xarray'``.
    ``download()`` copies the file of that name from NOAA GML's FTP server
    (``/data/trace_gases/{specie}/{sample_type}/{platform}/{ext}/``).

    See Also
    --------
    GMLData.apply_qaqc : Keeps the samples with good flags.
    CarbonTracker : NOAA's model, for a background.

    Examples
    --------
    Making one reads nothing; it names the file:

    >>> from lair.noaa import GMLData
    >>> uta = GMLData("ch4", "uta", gml_dir="gml")
    >>> uta.filepath.as_posix()
    'gml/ch4/flask/ch4_uta_surface-flask_1_ccgg_event.txt'

    Then ``uta.download()`` fetches it and ``uta.data`` reads it.
    """

    file_template = "{specie}_{site}_{platform}-{sample_type}_{lab_id}_{measurement_group}_{frequency}.{ext}"

    driver_ext = {"pandas": "txt", "xarray": "nc"}

    def __init__(
        self,
        specie: str,
        site: str,
        platform: Literal["surface", "shipboard"] = "surface",
        sample_type: Literal["flask", "pfp"] = "flask",
        lab_id: int = 1,
        measurement_group: Literal["ccgg", "sil"] = "ccgg",
        frequency: Literal["event", "month"] = "event",
        driver: Literal["pandas", "xarray"] = "pandas",
        gml_dir: str | Path | None = None,
    ):
        """Initialize a GMLData; the class docstring describes the parameters."""
        self.specie = specie
        self.site = site
        self.platform = platform
        self.sample_type = sample_type
        self.lab_id = lab_id
        self.measurement_group = measurement_group
        self.frequency = frequency
        self.driver = driver
        self.ext = self.driver_ext[driver]
        self.gml_dir = get_data_dir(GML_DIR_ENV, gml_dir)
        self.directory = self.gml_dir / specie / sample_type
        self.filename = self.file_template.format(**self.__dict__)
        self.filepath = self.directory / self.filename

    def __repr__(self):
        return f"GMLData(specie={self.specie}, site={self.site}, platform={self.platform}, sample_type={self.sample_type}, lab_id={self.lab_id}, measurement_group={self.measurement_group}, frequency={self.frequency}, driver={self.driver})"

    def __str__(self):
        return f"NOAA GML Data({self.specie}, {self.site}, {self.sample_type})"

    def download(self):
        """
        Download the file from NOAA GML's FTP server and return its path.

        See Also
        --------
        GMLData.data : Reads the file.

        Examples
        --------
        It needs the network, so it is not run when the docs build:

        >>> from lair.noaa import GMLData
        >>> GMLData("ch4", "uta", gml_dir="gml").download()  # doctest: +SKIP
        'gml/ch4/flask/ch4_uta_surface-flask_1_ccgg_event.txt'
        """
        host = "ftp.gml.noaa.gov"
        path = f"/data/trace_gases/{self.specie}/{self.sample_type}/{self.platform}/{self.ext}/{self.filename}"
        ftp_download(host, path, str(self.directory))
        return str(self.filepath)

    @cached_property
    def data(self):
        """
        The file's data, read once.

        With ``driver='pandas'``, a DataFrame indexed by ``datetime``, sorted:
        the sample times of an event file (naive, as NOAA writes them in UTC),
        or the first of each month for a monthly file. With ``'xarray'``, the
        NetCDF file's Dataset, with ``time`` along ``obs``.

        See Also
        --------
        GMLData.apply_qaqc : Keeps the samples with good flags.
        GMLData.download : Fetches the file first.
        """
        if self.driver == "pandas" and self.frequency == "month":
            data = self._read_monthly()
        elif self.driver == "pandas":
            data = pd.read_csv(
                self.filepath, sep=" ", comment="#", parse_dates=["datetime"]
            )
            # pandas-stubs type `.dt` of an untyped column as the generic accessor
            # pyrefly: ignore[missing-attribute]
            data["datetime"] = data.datetime.dt.tz_localize(None)
            data = data.dropna(subset=["datetime"]).set_index("datetime").sort_index()
        elif self.driver == "xarray":
            data = xr.open_dataset(self.filepath)
            times = data.time.values
            data = data.drop_vars("time").assign_coords(time=("obs", times))
        else:
            raise ValueError("Invalid driver")

        return data

    def _read_monthly(self) -> pd.DataFrame:
        """
        Read a monthly-mean file, indexed by the first of each month.

        Unlike event files, monthly files have no header row: the column names
        are only given in a ``# data_fields:`` comment, and columns are aligned
        with runs of spaces. There is no ``qcflag`` column.
        """
        fields = None
        with open(self.filepath) as f:
            for line in f:
                if not line.startswith("#"):
                    break
                if line.startswith("# data_fields:"):
                    fields = line.split(":", 1)[1].split()
        if fields is None:
            raise ValueError(f'No "# data_fields:" header line in {self.filepath}')

        data = pd.read_csv(
            self.filepath, sep=r"\s+", comment="#", header=None, names=fields
        )
        data["datetime"] = pd.to_datetime(
            dict(year=data["year"], month=data["month"], day=1)
        )
        return data.set_index("datetime").sort_index()

    @staticmethod
    def apply_qaqc(
        data: pd.DataFrame | xr.Dataset,
        flags: None | str | list[str] = None,
        driver: str = "pandas",
    ):
        """
        Apply QA/QC filtering.

        Parameters
        ----------
        data : pd.DataFrame | xr.Dataset
            The data.
        flags : None | str | list of str, optional
            The allowed QA/QC flags. If None, only keep good data (qcflag == '...').
            By default None.
        driver : str, optional
            The driver to use to read the data, by default 'pandas'.

        Returns
        -------
        pd.DataFrame | xr.Dataset
            The filtered data.

        See Also
        --------
        GMLData.data : The data to filter.

        Examples
        --------
        NOAA marks a good sample ``...``; a sample with a flag is kept only
        when that flag is allowed:

        >>> import pandas as pd
        >>> from lair.noaa import GMLData
        >>> samples = pd.DataFrame(
        ...     {"value": [1900.0, 2400.0], "qcflag": ["...", "..P"]}
        ... )
        >>> GMLData.apply_qaqc(samples)["value"].tolist()
        [1900.0]
        >>> GMLData.apply_qaqc(samples, flags="..P")["value"].tolist()
        [1900.0, 2400.0]
        """
        allowed_flags = ["..."]
        if flags is not None:
            if isinstance(flags, str):
                flags = [flags]
            allowed_flags.extend(flags)

        if driver == "pandas":
            data = data[data.qcflag.isin(allowed_flags)]
        elif driver == "xarray":
            data = cast(xr.Dataset, data)
            qcflag = data.qcflag
            if qcflag.dtype.kind == "S":
                # netCDF char arrays (no _Encoding) decode to bytes, e.g. b'...'
                qcflag = qcflag.str.decode("ascii")
            data = data.where(qcflag.isin(allowed_flags), drop=True)
        else:
            raise ValueError("Invalid driver")
        return data
