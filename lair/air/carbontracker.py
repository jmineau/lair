"""
lair.air.carbontracker
~~~~~~~~~~~~~~~~~~~~~~

Module for reading CarbonTracker data.
"""

from abc import ABCMeta
import datetime as dt
from functools import cached_property
import os
from typing import Literal
import xarray as xr
from lair.config import GROUP_DIR
from lair.utils.records import ftp_download, list_files, Cacher

CARBONTRACKER_DIR = os.path.join(GROUP_DIR, 'carbontracker')
CO2_DIR = os.path.join(CARBONTRACKER_DIR, 'co2')
CH4_DIR = os.path.join(CARBONTRACKER_DIR, 'ch4')


def get_specie_from_version(version) -> Literal['ch4'] | Literal['co2']:
    return 'ch4' if 'ch4' in version.lower() else 'co2'


def download_carbontracker(version, carbon_tracker_dir=CARBONTRACKER_DIR,
                           sub_dirs = ['fluxes', 'molefractions'], 
                           pattern=None):
    """
    Download CarbonTracker data from the NOAA GML FTP server.

    Parameters
    ----------
    version : str
        The version of CarbonTracker data to download.
        Visit https://gml.noaa.gov/aftp/products/carbontracker/ to see available versions.
    carbon_tracker_dir : str, optional
        The directory to download the data to, by default CARBONTRACKER_DIR.
    sub_dirs : list of str, optional
        The subdirectories to download data from, by default ['fluxes', 'molefractions'].
        If None, download the entire version data.
    pattern : str, optional
        The pattern to match against the files, by default None
    """
    host = 'ftp.gml.noaa.gov'
    parent = '/products/carbontracker'

    # Determine the specie from the version
    specie = get_specie_from_version(version)

    # Get the local version directory to download to
    version_dir = os.path.join(carbon_tracker_dir, specie, version)

    # Build list of remote paths to download
    path = f'{parent}/{specie}/{version}'
    paths = [f'{path}/{sub_dir}' for sub_dir in sub_dirs]

    # Download the data
    ftp_download(host, paths, version_dir, prefix=path, pattern=pattern)
    return None


class CarbonTracker(metaclass=ABCMeta):
    specie: Literal['ch4', 'co2']

    def __init__(self, version, carbon_tracker_directory=None, cache=True):
        self.version = version

        carbon_tracker_directory = carbon_tracker_directory or CARBONTRACKER_DIR
        self.directory = os.path.join(carbon_tracker_directory, self.specie, version)

        self.cache = cache

    def __repr__(self):
        return f"{self.__class__.__name__}(version={self.version}, directory={self.directory})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.version})"

    @staticmethod
    def from_version(version, carbon_tracker_directory=None):
        specie = get_specie_from_version(version)
        if specie == 'co2':
            raise ValueError("CarbonTrackerCO2 not yet implemented")
            return CarbonTrackerCO2(version, directory)
        elif specie == 'ch4':
            return CarbonTrackerCH4(version, carbon_tracker_directory)
        else:
            raise ValueError("Invalid specie")


class CarbonTrackerCH4(CarbonTracker):
    specie = 'ch4'

    def __init__(self, version='CT-CH4-2023', carbon_tracker_directory=None, parallel_parse=True):
        super().__init__(version, carbon_tracker_directory)
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
