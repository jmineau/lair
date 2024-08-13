"""
This module implements UATAQ instruments as classes.

Each instrument class is a subclass of the `Instrument` abstract base class and
implements methods for reading and parsing data files.

The `Instrument` class provides a common interface for all instrument classes
and defines abstract methods that must be implemented by each subclass.
"""

from abc import ABCMeta
import json
import pandas as pd
from typing import Literal, Iterator, Type

from lair.config import vprint
from lair.uataq import errors, filesystem
from lair.utils.clock import TimeRange

# TODO
# TRX01 aeth & no2 from horel-group

class Instrument(metaclass=ABCMeta):
    """
    Abstract base class for instrument objects.

    Attributes
    ----------
    model : str
        Model of the instrument.
    SID : str
        Site ID where the instrument is installed.
    name : str
        Name of the instrument.
    groups : list[str]
        Research groups that operate the instrument.
    loggers : set[str]
        Loggers used by the research groups to record data.
    config : dict
        Configuration settings for the instrument.
    
    Methods
    -------
    get_files(group: str, lvl: str) -> list[str]
        Get list of file paths for a given level.
    read_data(group: str, lvl: str, time_range: TimeRange, num_processes: int, file_pattern: str) -> pd.DataFrame
        Read and parse group data files for the given level and time range using multiple processes.
    """

    model: str

    def __init__(self, SID: str, name: str, loggers: dict, config: dict):
        """
        Initialize the Instrument object.

        Parameters
        ----------
        SID : str
            Site ID where the instrument is installed.
        name : str
            Name of the instrument.
        loggers : dict
            Dictionary of loggers used by different research groups.
        config : dict
            Configuration settings for the instrument.
        """
        self.SID = SID
        self.name = name
        self._loggers = loggers
        self.config = config

        self.groups = list(loggers.keys())
        self.loggers = set(loggers.values())

    def __str__(self):
        return f'{self.name}@{self.SID}'

    def __repr__(self):
        name = f", name='{self.name}" if self.name != self.model else ''
        config = json.dumps(self.config, indent=4)
        return (f"{self.__class__.__name__}({self.SID}"
                f"{name}, loggers={self._loggers}, config={config})")

    def _get_groupspace(self, group: str) -> filesystem.GroupSpace:
        """
        Get the groupspace object for the given group.

        Parameters
        ----------
        group : str
            The research group whose groupspace to retrieve.

        Returns
        -------
        GroupSpace
            The groupspace object.
        """
        if group not in filesystem.groups:
            raise errors.InvalidGroupError(f'{group} groupspace not found in filesystem')
        elif group not in self.groups:
            raise errors.InvalidGroupError(f'{group} group invalid for {self}')
        return filesystem.groups[group]

    def get_highest_lvl(self, group: str) -> str:
        """
        Get the highest data level for the instrument.

        Parameters
        ----------
        group : str
            The research group whose data to retrieve.

        Returns
        -------
        str
            The highest data level.
        """
        vprint('No level specified. Determining highest level...')
        groupspace = self._get_groupspace(group)
        return groupspace.get_highest_lvl(self.SID, self.name)

    def get_files(self, group: str, lvl: str) -> list[str]:
        """
        Get list of file paths for a given level.

        Parameters
        ----------
        group : str
            The research group whose data to retrieve.
        lvl : str
            The level of the data to retrieve.

        Returns
        -------
        list[str]
            A list of file paths.
        """
        groupspace = self._get_groupspace(group)
        logger = self._loggers[group]
        return groupspace.get_files(self.SID, self.name, lvl, logger)

    def get_datafiles(self, group: str, lvl: str,
                      time_range: TimeRange,
                      pattern: str | None = None) -> list[filesystem.DataFile]:
        """
        Get data files for the given level and time range from the groupspace.

        Parameters
        ----------
        group : str
            The research group whose data to retrieve.
        lvl : str
            The level of the data to retrieve.
        time_range : TimeRange
            The time range of the data to retrieve.
        pattern : str
            A string pattern to filter the file paths.

        Returns
        -------
        list[DataFile]
            A list of data files.
        """
        # Check if instrument is active during the time range
        start, end = time_range
        installation_date = pd.to_datetime(self.config['installation_date'])
        removal_date = pd.to_datetime(self.config.get('removal_date', pd.Timestamp.max))
        if (start and start > removal_date) or (end and end < installation_date):
            raise errors.InactiveInstrumentError(self)

        groupspace = self._get_groupspace(group)
        logger = self._loggers[group]
        return groupspace.get_datafiles(self.SID, self.name, lvl, logger, time_range, pattern)

    def standardize_data(self, group: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Manipulate the data to a standard format between research groups,
        renaming columns, converting units, mapping values, etc. as needed.

        Parameters
        ----------
        group : str
            The research group whose data to standardize.
        data : pandas.DataFrame
            The data to standardize.

        Returns
        -------
        pandas.DataFrame
            The standardized data.
        """
        groupspace = self._get_groupspace(group)
        return groupspace.standardize_data(self.model, data)

    def read_data(self, group: str, lvl: str | None = None,
                  time_range: TimeRange | TimeRange._input_types = None,
                  num_processes: int | Literal['max'] = 1,
                  file_pattern: str | None = None) -> pd.DataFrame:
        """
        Read and parse data files for the given level and time range,
        using multiple processes if specified.

        Parameters
        ----------
        group : str
            The research group whose data to read.
        lvl : str
            The level of the data to read.
        time_range : str | list[Union[str, dt.datetime, None]] | tuple[Union[str, dt.datetime, None], Union[str, dt.datetime, None]] | slice | None
            The time range to read data. Default is None which reads all available data.
        num_processes : int | 'max'
            The number of processes to use for parallelization.
        file_pattern : str
            A string pattern to filter the file paths.

        Returns
        -------
        pandas.DataFrame
            A concatenated DataFrame containing the parsed data from files.
        """
        vprint(f'Reading data for {self} from the {group} groupspace...')

        # Format lvl & time_range
        lvl = lvl.lower() if lvl else self.get_highest_lvl(group)
        assert lvl in filesystem.lvls, f"Invalid data level '{lvl}'. Must be one of {filesystem.lvls}."
        time_range = TimeRange(time_range)

        vprint(f'Getting {lvl} files...')
        datafiles = self.get_datafiles(group, lvl, time_range, file_pattern)
        data = filesystem.parse_datafiles(datafiles, time_range, num_processes)
        vprint('Mapping columns to UATAQ names...')
        data = self.standardize_data(group, data)
        vprint('done.')
        return data


def configure_instrument(SID: str, name: str, config: dict, loggers: dict | None = None
                         ) -> Instrument:
    """
    Configure an instrument object based on the given configuration settings.

    Parameters
    ----------
    SID : str
        Site ID where the instrument is installed.
    name : str
        Name of the instrument.
    config : dict
        Configuration settings for the instrument.
    loggers : dict, optional
        Dictionary of loggers used by different research groups.

    Returns
    -------
    Instrument
        An instrument object configured with the given settings.

    Raises
    ------
    ValueError
        If the instrument model is not found in the catalog.
    ValueError
        If no loggers are found for the instrument at the site.
    """
    model = config.get('model', name)

    loggers = config.get('loggers') or loggers
    if not loggers:
        raise ValueError(f"No loggers found for instrument {name} at site {SID}.")

    InstrumentClass = catalog.get(model)
    if not InstrumentClass:
        raise ValueError(f"Model '{model}' not found in the instrument catalog.")

    return InstrumentClass(SID, name, loggers, config)


class InstrumentEnsemble:
    """
    Container for an ensemble of instruments at a site.

    Attributes
    ----------
    SID : str
        Site ID of the ensemble.
    configs : dict[str, dict]
        Dictionary of configuration settings for each instrument.
    names : list[str]
        List of instrument names in the ensemble.
    loggers : set[str]
        Set of loggers used by the research groups.
    groups : set[str]
        Set of research groups that operate the instruments.
    pollutants : set[str]
        Set of pollutants measured by the instruments.
    """
    def __init__(self, SID: str, configs: dict, loggers: dict | None = None):
        """
        Initialize the InstrumentEnsemble object.

        Parameters
        ----------
        SID : str
            Site ID of the ensemble.
        configs : dict[instrument, config]
            Dictionary of configuration settings for each instrument.
        loggers : dict[group, logger], optional
            Dictionary of loggers used by different research groups.
        """
        self.SID = SID
        self.configs = configs
        self._loggers = loggers

        self.names = list(configs.keys())

        # Configure instruments
        self._instruments = {name: configure_instrument(SID, name, config, loggers)
                             for name, config in configs.items()}

        # Gather ensemble attributes
        self.loggers = set()
        self.groups = set()
        self.pollutants = set()
        for instrument in self._instruments.values():
            self.loggers.update(instrument.loggers)
            self.groups.update(instrument.groups)

            if hasattr(instrument, 'pollutants'):
                self.pollutants.update(p.upper() for p in instrument.pollutants)

    def __repr__(self):
        configs = json.dumps(self.configs, indent=4)
        loggers = f', loggers={self._loggers}' if self._loggers else ''
        return f'InstrumentEnsemble("{self.SID}", configs={configs}{loggers})'

    def __str__(self):
        return f"InstrumentEnsemble({self.SID}, instruments={self.names})"

    def __getattr__(self, name: str) -> Instrument:
        return self._instruments[name]

    def __getitem__(self, name: str) -> Instrument:
        return self._instruments[name]

    def __contains__(self, name: str) -> bool:
        return name in self.names

    def __iter__(self) -> Iterator[Instrument]:
        return iter(self._instruments.values())


class SensorMixin:
    """
    Mixin for instrument objects that measure a pollutant.

    Attributes:
        pollutants (tuple): Tuple of pollutants measured by the instrument.
    """
    pollutants: tuple[str, ...]


class BB_205(Instrument, SensorMixin):
    model = '2b_205'
    pollutants = ('O3',)


class BB_405(Instrument, SensorMixin):
    model = '2b_405'
    pollutants = ('NO', 'NO2', 'NOx')


class CR1000(Instrument):
    model = 'cr1000'


class GPS(Instrument):
    model = 'gps'

    def read_data(self, group: str, lvl: str, 
                  time_range: TimeRange | TimeRange._input_types = [None, None],
                  num_processes: int | Literal['max'] = 1,
                  file_pattern: str | None = None) -> pd.DataFrame:
        # Read GPS data
        data = super().read_data(group, lvl, time_range, num_processes, file_pattern)

        if 'Speed_kt' in data.columns:
            # convert knots to m/s
            data['Speed_kt'] = data.Speed_kt * 0.514444
            data.rename(columns={'Speed_kt': 'Speed_m_s'}, inplace=True)

        return data


class LGR_NO2(Instrument, SensorMixin):
    model = 'lgr_no2'
    pollutants = ('NO2',)


class LGR_UGGA(Instrument, SensorMixin):
    model = 'lgr_ugga'
    pollutants = ('CO2', 'CH4')


class Licor_6262(Instrument, SensorMixin):
    model = 'licor_6262'
    pollutants = ('CO2',)


class Licor_7000(Licor_6262):
    model = 'licor_7000'


class Magee_AE33(Instrument, SensorMixin):
    model = 'magee_ae33'
    pollutants = ('BC',)


class MetOne_ES405(Instrument, SensorMixin):
    model = 'metone_es405'
    pollutants = ('PM1', 'PM2.5', 'PM4', 'PM10')


class MetOne_ES642(Instrument, SensorMixin):
    model = 'metone_es642'
    pollutants = ('PM2.5',)


class Teledyne_T200(Instrument, SensorMixin):
    model = 'teledyne_t200'
    pollutants = ('NO', 'NO2', 'NOx')


class Teledyne_T300(Instrument, SensorMixin):
    model = 'teledyne_t300'
    pollutants = ('CO',)


class Teledyne_T400(Instrument, SensorMixin):
    model = 'teledyne_t400'
    pollutants = ('O3',)


class Teledyne_T500u(Instrument, SensorMixin):
    model = 'teledyne_t500u'
    pollutants = ('NO2',)


class Teom_1400ab(Instrument, SensorMixin):
    model = 'teom_1400ab'
    pollutants = ('PM2.5',)


#: Instrument catalog
catalog: dict[str, Type[Instrument]] = {
    '2b_205':         BB_205,
    '2b_405':         BB_405,
    'cr1000':         CR1000,
    'gps':            GPS,
    'lgr_no2':        LGR_NO2,
    'lgr_ugga':       LGR_UGGA,
    'licor_6262':     Licor_6262,
    'licor_7000':     Licor_7000,
    'magee_ae33':     Magee_AE33,
    'metone_es405':   MetOne_ES405,
    'metone_es642':   MetOne_ES642,
    'teledyne_t200':  Teledyne_T200,
    'teledyne_t300':  Teledyne_T300,
    'teledyne_t400':  Teledyne_T400,
    'teledyne_t500u': Teledyne_T500u,
    'teom_1400ab':    Teom_1400ab
}
