"""
NOAA greenhouse gas data.
"""

from abc import ABCMeta
import datetime as dt
from functools import cached_property
import os
import pandas as pd
from typing import Literal, Union
import xarray as xr

from lair.config import GROUP_DIR
from lair.records import ftp_download, list_files, Cacher

#: CarbonTracker data directory
CARBONTRACKER_DIR = os.path.join(GROUP_DIR, 'carbontracker')

#: NOAA GML data directory
GML_DIR = os.path.join(GROUP_DIR, 'gml')


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

    def __init__(self, version: str, carbon_tracker_directory: str | None=None,
                 cache: bool=True):
        """
        Initialize a CarbonTracker object.

        Parameters
        ----------
        version : str
            The version of CarbonTracker data to download.
            Visit https://gml.noaa.gov/aftp/products/carbontracker/ to see available versions.
        carbon_tracker_dir : str, optional
            The directory to download the data to, by default CARBONTRACKER_DIR.
        cache : bool, optional
            Whether to cache the data, by default True.
        """
        self.version = version

        carbon_tracker_directory = carbon_tracker_directory or CARBONTRACKER_DIR
        self.directory = os.path.join(carbon_tracker_directory, self.specie, version)

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
            The directory to download the data to, by default CARBONTRACKER_DIR.

        Returns
        -------
        CarbonTracker
            The CarbonTracker object.
        """
        specie = CarbonTracker.get_specie_from_version(version)
        if specie == 'co2':
            raise ValueError("CarbonTrackerCO2 not yet implemented")
            return CarbonTrackerCO2(version, directory)
        elif specie == 'ch4':
            return CarbonTrackerCH4(version, carbon_tracker_directory)
        else:
            raise ValueError("Invalid specie")

    def download(self, sub_dirs: list[str]=['fluxes', 'molefractions'], 
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
        paths = [f'{path}/{sub_dir}' for sub_dir in sub_dirs]

        # Download the data
        ftp_download(host, paths, self.directory, prefix=path, pattern=pattern)
        return None


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

    def __init__(self, version='CT-CH4-2023', carbon_tracker_directory=None, cache=True,
                 parallel_parse=True):
        super().__init__(version, carbon_tracker_directory, cache)
        self.parallel_parse = parallel_parse

    @staticmethod
    def _preprocess_molefractions(ds):
        time_components = ds['time_components'].values
        time = [dt.datetime(*row) for row in time_components]

        ds = ds.assign_coords(time=time)
        ds = ds.drop_vars('time_components')

        return ds

    @cached_property
    def molefractions(self) -> xr.Dataset:
        'Molefractions Dataset. Cached property.'
        path = os.path.join(self.directory, 'molefractions')

        files = list_files(path, '*nc', full_names=True, recursive=True)

        if self.cache:
            from lair.config import CACHE_DIR
            cache_file = os.path.join(CACHE_DIR, 'carbontracker', self.specie, self.version, 'molefractions.pkl')
            open_mfdataset = Cacher(xr.open_mfdataset, cache_file)
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


class Flask:
    """
    NOAA GML Flask

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
        The NOAA GML directory to download the data to, by default GML_DIR.
    directory : str
        The directory for the Flask data.
    filename : str
        The filename for the Flask data.
    filepath : str
        The filepath for the Flask data.
    data : pd.DataFrame | xr.Dataset
        The Flask data.
    file_template : str
        The template for the Flask filename.
    driver_ext : dict
        The driver extensions.
    """
    file_template = '{specie}_{site}_{platform}-flask_{lab_id}_{measurement_group}_{frequency}.{ext}'

    driver_ext = {
        'pandas': 'txt',
        'xarray': 'nc'
    }

    def __init__(self, specie: str, site: str,
                 platform: Literal['surface', 'shipboard']='surface',
                 lab_id: int=1,
                 measurement_group: Literal['ccgg', 'sil']='ccgg',
                 frequency: Literal['event', 'month']='event',
                 driver: Literal['pandas', 'xarray']='pandas',
                 gml_dir: str|None=None):
        """
        Initialize a Flask object.
        
        Parameters
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
            The NOAA GML directory to download the data to, by default GML_DIR.
        """
        self.specie = specie
        self.site = site
        self.platform = platform
        self.lab_id = lab_id
        self.measurement_group = measurement_group
        self.frequency = frequency
        self.driver = driver
        self.ext = self.driver_ext[driver]
        self.gml_dir = gml_dir or GML_DIR
        self.directory = os.path.join(self.gml_dir, specie, 'flask')
        self.filename = self.file_template.format(**self.__dict__)
        self.filepath = os.path.join(self.directory, self.filename)

    def __repr__(self):
        return f'Flask(specie={self.specie}, site={self.site}, platform={self.platform}, lab_id={self.lab_id}, measurement_group={self.measurement_group}, frequency={self.frequency}, driver={self.driver})'

    def __str__(self):
        return f'NOAA GML Flask({self.specie}, {self.site})'

    def download(self):
        host = 'ftp.gml.noaa.gov'
        path = f'/data/trace_gases/{self.specie}/flask/surface/{self.ext}/{self.filename}'
        ftp_download(host, path, self.directory)
        return self.filepath

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
        Apply QA/QC to the Flask data.

        Parameters
        ----------
        data : pd.DataFrame | xr.Dataset
            The Flask data.
        flags : None | str | list of str, optional
            The allowed QA/QC flags. If None, only keep good data (qcflag == '...').
            By default None.
        driver : str, optional
            The driver to use to read the data, by default 'pandas'.

        Returns
        -------
        pd.DataFrame | xr.Dataset
            The Flask data with QA/QC applied.
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
