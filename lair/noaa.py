"""
NOAA greenhouse gas data.
"""

from abc import ABCMeta
import datetime as dt
from functools import cached_property
from pathlib import Path
import pandas as pd
from typing import Literal, Union

import numpy as np
import xarray as xr

from lair.config import get_data_dir
from lair.records import ftp_download, list_files, Cacher

#: Environment variable holding the CarbonTracker data root
CARBONTRACKER_DIR_ENV = 'LAIR_CARBONTRACKER_DIR'

#: Environment variable holding the NOAA GML data root
GML_DIR_ENV = 'LAIR_GML_DIR'


class CarbonTracker(metaclass=ABCMeta):
    """
    NOAA CarbonTracker

    Attributes
    ----------
    specie : Literal['ch4', 'co2']
        The greenhouse gas specie.
    version : str
        The CarbonTracker version.
    directory : str
        The directory for the version.
    cache : bool
        Whether to cache the data.

    Methods
    -------
    get_specie_from_version(version)
        Get the specie from the version.
    from_version(version, carbon_tracker_directory=None)
        Create a CarbonTracker object from the version.
    download(sub_dirs=['fluxes', 'molefractions'], pattern=None)
        Download CarbonTracker data from the NOAA GML FTP server.
    """
    specie: Literal['ch4', 'co2']

    def __init__(self, version: str, carbon_tracker_directory: str | Path | None=None,
                 cache: bool=True):
        """
        Initialize a CarbonTracker object.

        Parameters
        ----------
        version : str
            The version of CarbonTracker data to download.
            Visit https://gml.noaa.gov/aftp/products/carbontracker/ to see available versions.
        carbon_tracker_directory : str, optional
            The CarbonTracker data root, by default ``$LAIR_CARBONTRACKER_DIR``.
        cache : bool, optional
            Whether to cache the data, by default True.
        """
        self.version = version

        carbon_tracker_directory = get_data_dir(CARBONTRACKER_DIR_ENV, carbon_tracker_directory)
        self.directory = carbon_tracker_directory / self.specie / version

        self.cache = cache

    def __repr__(self):
        return f"{self.__class__.__name__}(version={self.version}, directory={self.directory})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.version})"

    @staticmethod
    def get_specie_from_version(version: str) -> Literal['ch4', 'co2']:
        """
        Get the specie from the version.

        Parameters
        ----------
        version : str
            The version of CarbonTracker data.

        Returns
        -------
        Literal['ch4', 'co2']
            The specie.
        """
        return 'ch4' if 'ch4' in version.lower() else 'co2'

    @staticmethod
    def from_version(version: str, carbon_tracker_directory: str | None=None):
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
            The CarbonTracker object.
        """
        specie = CarbonTracker.get_specie_from_version(version)
        if specie == 'co2':
            raise ValueError("CarbonTrackerCO2 not yet implemented")
        elif specie == 'ch4':
            return CarbonTrackerCH4(version, carbon_tracker_directory)
        else:
            raise ValueError("Invalid specie")

    def download(self, sub_dirs: list[str] | None=('fluxes', 'molefractions'),
                 pattern: str=None):
        """
        Download CarbonTracker data from the NOAA GML FTP server.

        Parameters
        ----------
        sub_dirs : list of str, optional
            The subdirectories to download data from, by default ['fluxes', 'molefractions'].
            If None, download the entire version data.
        pattern : str, optional
            The pattern to match against the files, by default None
        """
        host = 'ftp.gml.noaa.gov'
        parent = '/products/carbontracker'

        # Build list of remote paths to download
        path = f'{parent}/{self.specie}/{self.version}'
        if sub_dirs is None:
            paths = [path]
        else:
            paths = [f'{path}/{sub_dir}' for sub_dir in sub_dirs]

        # Download the data
        ftp_download(host, paths, str(self.directory), prefix=path, pattern=pattern)
        return None

    # --- sampling the concentration field at trajectory points ---------------

    @property
    def molefractions_dir(self) -> Path:
        """Directory of the molefraction (concentration-field) files."""
        return self.directory / 'molefractions'

    @staticmethod
    def _preprocess_molefractions(ds: xr.Dataset) -> xr.Dataset:
        """Normalize a molefraction dataset on open. Overridden per specie."""
        return ds

    def _molefraction_file_for_date(self, date) -> Path | None:
        """Molefraction file covering a UTC ``date`` (newest match), if present."""
        date = pd.Timestamp(date).date()
        month_dir = self.molefractions_dir / f'{date.year}' / f'{date.month:02d}'
        matches = sorted(month_dir.glob(f'*molefrac*_{date}.nc'))
        return matches[-1] if matches else None

    @staticmethod
    def _sample_field(points: pd.DataFrame, ds: xr.Dataset, variable: str) -> pd.DataFrame:
        """Evaluate ``ds[variable]`` at each row's (time, lati, long, zagl).

        Vertical placement uses the geopotential-height (``gph``) layer bounds:
        the particle's height-above-ground is added to the surface gph and the
        enclosing model level selected. Returns the input rows (index reset) with
        ``ct_<variable>_ppb`` plus the CT cell/level actually used.
        """
        dim = 'points'
        t = pd.DatetimeIndex(pd.to_datetime(points['time'], utc=True)).tz_convert(None).values
        sel = ds.sel(
            time=xr.DataArray(t, dims=dim),
            latitude=xr.DataArray(points['lati'].to_numpy(), dims=dim),
            longitude=xr.DataArray(points['long'].to_numpy(), dims=dim),
            method='nearest',
        )
        zagl = xr.DataArray(points['zagl'].to_numpy(), dims=dim)
        gph = sel['gph']
        gph_lo = gph.isel(boundary=slice(None, -1)).rename({'boundary': 'level'}).assign_coords(level=ds['level'].values)
        gph_hi = gph.isel(boundary=slice(1, None)).rename({'boundary': 'level'}).assign_coords(level=ds['level'].values)
        z_asl = zagl + gph_lo.isel(level=0)
        mask = (z_asl >= gph_lo) & (z_asl < gph_hi)
        has_layer = mask.any(dim='level')
        values = sel[variable].where(mask).max(dim='level', skipna=True).where(has_layer)
        level_idx = mask.argmax(dim='level')
        used_level = xr.DataArray(ds['level'].values, dims='level').isel(level=level_idx).where(has_layer)

        out = points.reset_index(drop=True).copy()
        out[f'ct_{variable}_ppb'] = values.load().to_numpy()
        out['ct_time'] = pd.to_datetime(sel['time'].load().to_numpy())
        out['ct_latitude'] = sel['latitude'].load().to_numpy()
        out['ct_longitude'] = sel['longitude'].load().to_numpy()
        out['ct_level'] = used_level.load().to_numpy()
        return out

    def sample(self, points: pd.DataFrame) -> pd.DataFrame:
        """Sample the molefraction field at ``points``.

        ``points`` is a DataFrame with columns ``time`` (UTC sample time),
        ``lati``, ``long``, ``zagl``; any other columns (e.g. ``indx``,
        ``run_time``) are carried through for downstream grouping. Returns one row
        per input point with ``ct_<specie>_ppb`` and the CT cell/level used.
        Points whose UTC date has no molefraction file are dropped.
        """
        points = points.reset_index(drop=True)
        if points.empty:
            return points
        dates = pd.DatetimeIndex(pd.to_datetime(points['time'], utc=True)).normalize()
        file_for = {d: self._molefraction_file_for_date(d) for d in dates.unique()}
        keep = dates.map(lambda d: file_for[d] is not None).to_numpy()
        points = points.loc[keep].reset_index(drop=True)
        files = sorted({f for f in file_for.values() if f is not None})
        if points.empty or not files:
            return points
        with xr.open_mfdataset(
            files, preprocess=self._preprocess_molefractions,
            data_vars='all', combine='by_coords',
        ) as ds:
            return self._sample_field(points, ds, self.specie)

    def background(self, points: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
        """Background mole fraction [ppm]: mean of the sampled field over ``points``.

        Samples the field at every point (e.g. trajectory endpoints) and averages
        over them. With ``by`` (e.g. ``'run_time'``) the mean and 1-sigma spread
        are returned per group; otherwise a single-row summary. Output is ppm
        (CarbonTracker mole fractions are stored in ppb).
        """
        sampled = self.sample(points)
        col = f'ct_{self.specie}_ppb'
        if col not in sampled:
            # sample() returns no rows (and no ct_ columns) when no
            # molefraction file covers the points
            sampled[col] = pd.Series(dtype=float)
        ppm = sampled[col] / 1000.0
        if by is None:
            return pd.DataFrame({'background_ppm': [ppm.mean()],
                                 'sigma_ppm': [ppm.std()],
                                 'n_endpoints': [int(ppm.notna().sum())]})
        grouped = ppm.groupby(sampled[by].to_numpy())
        return pd.DataFrame({'background_ppm': grouped.mean(),
                             'sigma_ppm': grouped.std(),
                             'n_endpoints': grouped.count()})


class CarbonTrackerCH4(CarbonTracker):
    """
    NOAA CarbonTracker-CH4

    Attributes
    ----------
    molefractions : xr.Dataset
        The molefractions Dataset.

    Methods
    -------
    calc_molefractions_pressure(molefractions)
        Calculate the pressure at each level in the molefractions Dataset.
    """
    specie = 'ch4'

    def __init__(self, version='CT-CH4-2025', carbon_tracker_directory=None, cache=True,
                 parallel_parse=True):
        super().__init__(version, carbon_tracker_directory, cache)
        self.parallel_parse = parallel_parse

    @staticmethod
    def _preprocess_molefractions(ds):
        # CT-CH4-2025+ ships with a proper datetime64 time coordinate already;
        # older versions (CT-CH4-2023) encode time in a `time_components` variable.
        if not np.issubdtype(ds['time'].dtype, np.datetime64):
            time_components = ds['time_components'].values
            time = [dt.datetime(*row) for row in time_components]
            ds = ds.assign_coords(time=time)
        ds = ds.drop_vars('time_components', errors='ignore')
        return ds

    @cached_property
    def molefractions(self) -> xr.Dataset:
        'Molefractions Dataset. Cached property.'
        path = self.directory / 'molefractions'

        files = list_files(str(path), '*nc', full_names=True, recursive=True)

        if self.cache:
            from lair.config import CACHE_DIR
            cache_file = Path(CACHE_DIR) / 'carbontracker' / self.specie / self.version / 'molefractions.pkl'
            open_mfdataset = Cacher(xr.open_mfdataset, str(cache_file))
        else:
            open_mfdataset = xr.open_mfdataset
        ds = open_mfdataset(files, preprocess=CarbonTrackerCH4._preprocess_molefractions, 
                            parallel=self.parallel_parse)
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
            The molefractions Dataset with the pressure calculated.
        """
        molefractions['P'] = (molefractions.at 
                              + molefractions.bt * molefractions.surf_pressure)
        molefractions['P'] /= 100  # Convert to hPa
        molefractions['P'].attrs = {'long_name': 'Pressure', 'units': 'hPa',
                                    'comment': 'Calculated from hybrid sigma-pressure coefficients and surface pressure.'}
        return molefractions


class GMLData:
    """
    NOAA GML Data

    Attributes
    ----------
    specie : str
        The greenhouse gas specie.
    site : str
        The site where the flask samples were collected.
    platform : str, optional
        The platform where the flask samples were collected, by default 'surface'.
    lab_id : int, optional
        The lab ID, by default 1.
    measurement_group : str, optional
        The measurement group, by default 'ccgg'.
    frequency : str, optional
        The frequency of the measurements, by default 'event'.
    driver : str, optional
        The driver to use to read the data, by default 'pandas'.
    gml_dir : str, optional
        The NOAA GML data root, by default ``$LAIR_GML_DIR``.
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
    """
    file_template = '{specie}_{site}_{platform}-{sample_type}_{lab_id}_{measurement_group}_{frequency}.{ext}'

    driver_ext = {
        'pandas': 'txt',
        'xarray': 'nc'
    }

    def __init__(self, specie: str, site: str,
                 platform: Literal['surface', 'shipboard']='surface',
                 sample_type: Literal['flask', 'pfp']='flask',
                 lab_id: int=1,
                 measurement_group: Literal['ccgg', 'sil']='ccgg',
                 frequency: Literal['event', 'month']='event',
                 driver: Literal['pandas', 'xarray']='pandas',
                 gml_dir: str | Path | None=None):
        """
        Initialize a GMLData object.

        Parameters
        ----------
        specie : str
            The greenhouse gas specie.
        site : str
            The site where the flask samples were collected.
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
            The driver to use to read the data, by default 'pandas'.
        gml_dir : str, optional
            The NOAA GML data root, by default ``$LAIR_GML_DIR``.
        """
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
        return f'GMLData(specie={self.specie}, site={self.site}, platform={self.platform}, sample_type={self.sample_type}, lab_id={self.lab_id}, measurement_group={self.measurement_group}, frequency={self.frequency}, driver={self.driver})'

    def __str__(self):
        return f'NOAA GML Data({self.specie}, {self.site}, {self.sample_type})'

    def download(self):
        host = 'ftp.gml.noaa.gov'
        path = f'/data/trace_gases/{self.specie}/{self.sample_type}/{self.platform}/{self.ext}/{self.filename}'
        ftp_download(host, path, str(self.directory))
        return str(self.filepath)

    @cached_property
    def data(self):
        if self.driver == 'pandas':
            data = pd.read_csv(self.filepath, sep=' ', comment='#',
                               parse_dates=['datetime'])
            data['datetime'] = data.datetime.dt.tz_localize(None)
            data = data.dropna(subset=['datetime']).set_index('datetime').sort_index()
        elif self.driver == 'xarray':
            data = xr.open_dataset(self.filepath)
            times = data.time.values
            data = data.drop_vars('time').assign_coords(time=('obs', times))
        else:
            raise ValueError("Invalid driver")

        return data

    @staticmethod
    def apply_qaqc(data: Union[pd.DataFrame, xr.Dataset], flags: None | str | list[str]=None,
                   driver: str='pandas'):
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
        """
        allowed_flags = ['...']
        if flags is not None:
            if isinstance(flags, str):
                flags = [flags]
            allowed_flags.extend(flags)

        if driver == 'pandas':
            data = data[data.qcflag.isin(allowed_flags)]
        elif driver == 'xarray':
            data = data.where(data.qcflag.isin(allowed_flags), drop=True)
        else:
            raise ValueError("Invalid driver")
        return data
