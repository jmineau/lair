"""
SODAR data from MesoWest (Horel Group).

Reads SODAR (SOnic Detection And Ranging) wind-profile data archived as HDF5 by
the Horel group.

lair is a toolkit, so the data location is **not** hard-coded. Provide it either
explicitly (``Sodar(SID, mesowest_dir=...)``) or via the ``LAIR_MESOWEST_DIR``
environment variable. On CHPC this is
``/uufs/chpc.utah.edu/common/home/horel-group/oper/mesowest``.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

from lair._optional import import_optional_dependency
from lair.clock import TimeRange
from lair.parallel import parallelize


#: Environment variable pointing at the MesoWest data directory.
MESOWEST_DIR_ENV = 'LAIR_MESOWEST_DIR'


def resolve_mesowest_dir(mesowest_dir: str | None = None) -> str:
    """
    Resolve the MesoWest data directory.

    Parameters
    ----------
    mesowest_dir : str, optional
        Explicit path. If omitted, falls back to the ``LAIR_MESOWEST_DIR``
        environment variable.

    Returns
    -------
    str
        The resolved directory path.

    Raises
    ------
    ValueError
        If neither an argument nor the environment variable is set.
    """
    mesowest_dir = mesowest_dir or os.environ.get(MESOWEST_DIR_ENV)
    if not mesowest_dir:
        raise ValueError(
            'No MesoWest data directory configured. Pass `mesowest_dir=...` or '
            f'set the {MESOWEST_DIR_ENV} environment variable '
            '(on CHPC: /uufs/chpc.utah.edu/common/home/horel-group/oper/mesowest).'
        )
    return mesowest_dir


class Sodar:
    """
    Horel Group SODAR wind profiler.

    Parameters
    ----------
    SID : str
        Station identifier.
    mesowest_dir : str, optional
        Path to the MesoWest data directory. Defaults to the
        ``LAIR_MESOWEST_DIR`` environment variable (see :func:`resolve_mesowest_dir`).
    """

    model = 'sodar'
    species_measured = ('wind',)

    def __init__(self, SID: str, mesowest_dir: str | None = None):
        self.SID = SID.upper()

        mesowest_dir = resolve_mesowest_dir(mesowest_dir)
        self.sodar_dir = os.path.join(mesowest_dir, 'sodar_data')
        self.archive_dir = os.path.join(self.sodar_dir, 'hdf5archive')

        self.metafile = os.path.join(self.archive_dir,
                                     f'{self.SID}_full_metadata_log.h5')
        self.meta = pd.read_hdf(self.metafile, key='metagroup/metadata')
        self.variables = pd.read_hdf(self.metafile, key='metagroup/variables')
        self.variables.set_index('SHORTNAME', inplace=True)

    def get_files(self, lvl: str = 'raw', time_range=(None, None)) -> list[str]:
        """
        List archived SODAR HDF5 files for this site within a time range.

        Parameters
        ----------
        lvl : str, optional
            Processing level. Currently only ``'raw'`` is supported.
        time_range : tuple | TimeRange, optional
            Restrict to files whose month overlaps this range.

        Returns
        -------
        list[str]
            Matching file paths, sorted by name.
        """
        time_range = TimeRange(time_range)

        files = []
        for file in sorted(os.listdir(self.archive_dir)):
            if not (file.startswith(self.SID) and file.endswith('sodar.h5')):
                continue
            # Filename embeds the month; slice matches the archive convention.
            date_str = file[6:13].replace('_', '-')
            period = pd.Period(date_str, freq='M')
            # Keep files whose month overlaps the requested range.
            if time_range.start is not None and period.end_time < pd.Timestamp(time_range.start):
                continue
            if time_range.stop is not None and period.start_time > pd.Timestamp(time_range.stop):
                continue
            files.append(os.path.join(self.archive_dir, file))
        return files

    @staticmethod
    def parse(file, variables) -> xr.Dataset:
        """
        Parse a single SODAR HDF5 file into an xarray Dataset.

        Requires the optional PyTables (``tables``) dependency.
        """
        tables = import_optional_dependency('tables')  # PyTables

        with tables.open_file(file, mode='r') as f:
            table = f.root['obsdata/observations']
            data = table.read()

        var_names = data.dtype.names

        time = pd.to_datetime(data['DATTIM'], unit='s')

        data_vars = {'STATION_ID': (['Time_UTC'], data['STATION_ID']),
                    **{var: (['Time_UTC', 'level'], data[var])
                        for var in var_names
                        if var not in ['DATTIM', 'STATION_ID']}}

        ds = xr.Dataset(
            coords={'Time_UTC': time},
            data_vars=data_vars)

        ds = ds.where(ds != -9999)

        for variable in variables.index:
            ds[variable] /= variables.loc[variable, 'MULT']

        return ds

    def read_data(self, lvl='raw', time_range=(None, None), num_processes=1):
        """
        Read and merge SODAR data over a time range.
        """
        files = self.get_files(lvl, time_range)
        if not files:
            raise FileNotFoundError(
                f'No {lvl} SODAR files for {self.SID} in {self.archive_dir} '
                f'overlapping {time_range}')
        read_files = parallelize(self.parse, num_processes=num_processes)
        data = xr.merge(read_files(files, variables=self.variables))

        time_range = TimeRange(time_range)
        return data.sel(Time_UTC=slice(time_range.start, time_range.stop))

    @staticmethod
    def get_winds_at_height(data, height):
        """
        Extract wind direction/speed at a given height from parsed SODAR data.
        """
        level = np.where(data.HEIGHT[0] == height)[0][0]
        winds = data.sel(level=level).to_pandas()[['WD', 'WS']]

        winds.rename(columns={'WD': 'direction', 'WS': 'speed'}, inplace=True)

        winds.dropna(how='all', inplace=True)

        return winds
