"""
This module contains classes and functions for working in the CHPC UATAQ filesystem.
"""

from abc import ABCMeta, abstractmethod
import os
import pandas as pd
from typing import Literal, Type

from lair.config import vprint
from lair.uataq import errors
from lair.utils.clock import TimeRange
from lair.utils.parallel import parallelize


lvls: dict = {
    'raw':        1,
    'qaqc':       2,
    'calibrated': 3,
    'final':      4
}


class DataFile(metaclass=ABCMeta):
    """
    Abstract base class for data files.

    Attributes
    ----------
    path : str
        The file path.
    period : pd.Period
        The period of the data file.
    logger : str
        The logger name.
    date_slicer : slice
        A slice object to extract the date from the file name.
    file_freq : str
        The file frequency.
    ext : str
        The file extension.

    Methods
    -------
    parse()
        Parse the data file.
    """
    logger: str
    date_slicer: slice
    file_freq: str
    ext: str

    def __init__(self, path: str):
        """
        Initialize the DataFile object.
        Determines the period of the data file from the file name.

        Parameters
        ----------
        path : str
            The file path.
        """
        self.path = path

        # Get date from file name
        fname = os.path.basename(path)
        date_str = fname[self.date_slicer].replace('_', '-')
        self.period = pd.Period(date_str, freq=self.file_freq)

    def __str__(self) -> str:
        return self.path

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.path})'

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """
        Parse the data file. Must be implemented by subclasses.
        """
        raise NotImplementedError


def filter_datafiles(files: list[DataFile], time_range: TimeRange,
                    pattern: str | None = None) -> list[DataFile]:
    """
    Filter a list of files by a given time range.

    Parameters
    ----------
    files : list
        A list of DataFile objects.
    time_range : TimeRange
        A TimeRange object representing the time range to filter by.
    pattern : str, optional
        A string pattern to filter the file paths. Defaults to None.

    Returns
    -------
    list[DataFile]
        A list of DataFile objects that match the given time range.
    """
    vprint(f'Filtering files to time range: {time_range}')

    df = pd.DataFrame([(file, file.period) for file in files], columns=['file', 'period'])
    df = df.set_index('period').sort_index()

    # Filter by time range
    filtered_files = df.loc[time_range.start: time_range.stop, 'file'].tolist()

    # Filter by pattern
    if pattern:
        filtered_files = [file for file in filtered_files if pattern in file.path]

    if len(filtered_files) > 0 and filtered_files[-1].period.start_time == time_range.stop:
        # Drop the last file if it starts at the end of the time range
        filtered_files = filtered_files[:-1]

    if len(filtered_files) == 0:
        raise errors.ReaderError('No files found within the specified time range.')

    return filtered_files


def _parse_datafile(datafile: DataFile) -> pd.DataFrame | None:
    """
    Private function to parse a single data file.

    Parameters
    ----------
    datafile : DataFile
        A DataFile object.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the parsed data.
    """
    try:
        return datafile.parse()
    except errors.ParserError as e:
        vprint(f'Error parsing {datafile}: {e}')


def parse_datafiles(files: list[DataFile], time_range: TimeRange,
                    num_processes: int | Literal['max'] = 1,
                    driver: Literal['pandas'] | Literal['xarray'] = 'pandas'):
    """
    Read and parse data files using multiple processes.

    Parameters
    ----------
    files : list
        A list of DataFile objects.
    time_range : TimeRange
        A TimeRange object representing the time range to filter by.
    num_processes : int, optional
        The number of processes to use. Defaults to 1.
    driver : str, optional
        The data driver to use. Defaults to 'pandas'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the parsed data.
    """
    # Parse the data files in parallel
    datasets =  parallelize(_parse_datafile, num_processes)(files)

    # Concatenate the datasets
    vprint('Concatenating datasets and reducing rows to time range...')
    if driver == 'pandas':
        data = pd.concat(datasets)

        # Set time as index and filter to time_range
        data = data.dropna(subset='Time_UTC').set_index('Time_UTC').sort_index()
        data = data.loc[time_range.start: time_range.stop]
    elif driver == 'xarray':
        # TODO
        raise NotImplementedError('xarray driver not implemented yet.')
    else:
        raise ValueError(f'Invalid driver: {driver}')

    return data


class GroupSpace(metaclass=ABCMeta):
    """
    Abstract base class for group spaces.

    Attributes
    ----------
    name : str
        The group name.
    datafiles : dict[str, Type[DataFile]]
        A dictionary of datafile keys and DataFile classes.

    Methods
    -------
    get_highest_lvl(SID, instrument)
        Get the highest data level for a given site and instrument.
    get_files(SID, instrument, lvl, logger)
        Get list of file paths for a given site, instrument, and level.
    get_datafile_key(instrument, lvl, logger)
        Get the datafile key based on the instrument, level, and logger.
    get_datafile_class(instrument, lvl, logger)
        Get the DataFile class based on the instrument, level, and logger.
    get_datafiles(SID, instrument, lvl, logger, time_range, pattern)
        Returns a list of data files for a given level and time range.
    """
    name: str  # group name
    datafiles: dict[str, Type[DataFile]]  # datafile key: DataFile class

    def __repr__(self):
        return f'{self.__class__.__name__}()'

    def __str__(self):
        return f'{self.name.capitalize()} GroupSpace'

    @staticmethod
    @abstractmethod
    def get_highest_lvl(SID: str, instrument: str) -> str:
        """
        Get the highest data level for a given site and instrument.

        Parameters
        ----------
        SID : str
            The site ID.
        instrument : str
            The instrument name.

        Returns
        -------
        str
            The highest data level.
        """
        raise NotImplementedError

    @abstractmethod
    def get_files(self, SID: str, instrument: str, lvl: str, logger: str) -> list[str]:
        """
        Get list of file paths for a given site, instrument, and level.

        Parameters
        ----------
        SID : str
            The site ID.
        instrument : str
            The instrument name.
        lvl : str
            The data level.
        logger : str
            The logger name.

        Returns
        list[str]
            A list of file paths.
        """
        raise NotImplementedError

    @abstractmethod
    def get_datafile_key(self, instrument: str, lvl: str, logger: str) -> str:
        """
        Get the datafile key based on the instrument, level, and logger.

        Parameters
        ----------
        instrument : str
            The instrument name.
        lvl : str
            The data level.
        logger : str
            The logger name.

        Returns
        -------
        str
            The datafile key.
        """
        raise NotImplementedError

    def get_datafile_class(self, instrument: str, lvl: str, logger: str
                           ) -> Type[DataFile]:
        """
        Get the DataFile class based on the instrument, level, and logger.

        Parameters
        ----------
        instrument : str
            The instrument name.
        lvl : str
            The data level.
        logger : str
            The logger name.

        Returns
        -------
        Type[DataFile]
            The DataFile class.
        """
        key = self.get_datafile_key(instrument, lvl, logger)
        DataFileClass = self.datafiles.get(key)
        if DataFileClass is None:
            raise ValueError(f'DataFile class not found for key: {key} in {self.name} group.')
        return DataFileClass

    def get_datafiles(self, SID: str, instrument: str, lvl: str, logger: str,
                      time_range: TimeRange, pattern: str | None = None) -> list[DataFile]:
        """
        Returns a list of data files for a given level and time range.

        Parameters
        ----------
        SID : str
            The site ID.
        instrument : str
            The instrument name.
        lvl : str
            The data level.
        logger : str
            The logger name.
        time_range : TimeRange
            The time range to filter by.
        pattern : str, optional
            A string pattern to filter the file paths. Defaults to None.

        Returns
        -------
        list[DataFile]
            A list of DataFile objects.
        """
        DataFileClass = self.get_datafile_class(instrument, lvl, logger)

        file_paths = self.get_files(SID, instrument, lvl, logger)

        # Initialize DataFile objects from file paths
        datafiles = []
        for path in file_paths:
            if path.endswith(DataFileClass.ext):
                try:
                    datafiles.append(DataFileClass(path))
                except errors.DataFileInitializationError as e:
                    vprint(f'Unable to initialize {DataFileClass.__name__} from {path}: {e}')
            continue

        return filter_datafiles(datafiles, time_range, pattern)

    @staticmethod
    @abstractmethod
    def standardize_data(instrument: str, data: pd.DataFrame
                         ) -> pd.DataFrame:
        """
        Manipulate the data to a standard format between research groups,
        renaming columns, converting units, mapping values, etc. as needed.

        Parameters
        ----------s
        instrument : str
            The instrument model.
        data : pd.DataFrame
            The data to standardize.

        Returns
        -------
        pd.DataFrame
            The standardized data.
        """
        raise NotImplementedError


groups: dict = {}
