"""
lair.uataq.sites
~~~~~~~~~~~~~~~~

This module provides classes and functions for working with UATAQ sites.
"""

from collections import defaultdict
import datetime as dt
from datetime import timezone
import geopandas as gpd
import json
import pandas as pd
from typing import Sequence, Union, Literal

from lair.config import vprint
from lair.uataq import errors, filesystem, instruments
from lair.utils.clock import TimeRange

_all_or_mult_strs = Union[Literal['all'], str, list[str], tuple[str, ...], set[str]]


class Site:
    """
    A class representing a site where atmospheric measurements are taken.

    Attributes
    ----------
    SID : str
        The site identifier.
    config : dict
        A dictionary containing configuration information for the site.
    instruments : InstrumentEnsemble
        An instance of the InstrumentEnsemble class representing the instruments at the site.
    groups : set of str
        The research groups that collect data at the site.
    loggers : set of str
        The loggers used by research groups that record data at a site.
    pollutants : set of str
        The pollutants measured at the site.

    Methods
    -------
    read_data(instruments='all', lvl=None, time_range=None, num_processes=1, file_pattern=None)
        Read data for each instrument for specified level.
    read_obs(pollutants='all', format='wide', time_range=None, num_processes=1)
        Read observations for each pollutant, combining instruments by pollutants.
    get_recent_obs(recent=dt.timedelta(days=10), lvl='qaqc')
        Get recent observations from site instruments.
    """

    def __init__(self, SID: str, config: dict, instruments: instruments.InstrumentEnsemble):
        """
        Initializes a Site object with the given site ID.

        Parameters
        ----------
        SID : str
            The site identifier.
        config : dict
            A dictionary containing configuration information for the site:
            {
                name: str,
                is_active: bool,
                is_mobile: bool,
                latitude: float,
                longitude: float,
                zagl: float,
                loggers: dict,
                instruments: {
                    instrument: {
                        loggers: dict
                        installation_date: str,
                        removal_date: str,
                    }
                }
            }
        instruments : InstrumentEnsemble
            An instance of the InstrumentEnsemble class representing the instruments at the site.
        """
        self.SID = SID
        self.config = config
        self.instruments = instruments
        self.groups = instruments.groups
        self.loggers = instruments.loggers
        self.pollutants = instruments.pollutants

        # Build pollutant: instruments lookup table
        self.pollutant_instruments = defaultdict(list)
        for instrument in self.instruments:
            if hasattr(instrument, 'pollutants'):
                for pollutant in getattr(instrument, 'pollutants'):
                    self.pollutant_instruments[pollutant].append(instrument)

    def __repr__(self):
        cls = self.__class__.__name__
        config = json.dumps(self.config, indent=4)
        instruments = repr(self.instruments)
        return f'{cls}(SID="{self.SID}", config={config}, instruments={instruments})'

    def __str__(self):
        return f"{self.__class__.__name__}: {self.SID}"

    def read_data(self, instruments: _all_or_mult_strs = 'all',
                  group: str | None = None,
                  lvl: str | None = None, 
                  time_range: TimeRange | TimeRange._input_types = None,
                  num_processes: int | Literal['max'] = 1,
                  file_pattern: str | None = None) -> dict[str, pd.DataFrame]:
        """
        Read data for the specified instruments and level.

        Parameters
        ----------
        instruments : str or list of str or 'all'
            The instrument(s) to read data from. If 'all', read data from all instruments.
            Default is 'all'.
        group : str, optional
            The research group to read data from. Default is None which uses the default group.
        lvl : str, optional
            The data level to read. Default is None which reads the highest level available.
        time_range : str | list[Union[str, dt.datetime, None]] | tuple[Union[str, dt.datetime, None], Union[str, dt.datetime, None]] | slice | None
            The time range to read data. Default is None which reads all available data.
        num_processes : int or 'max'
            The number of processes to use for reading data. Default is 1.
        file_pattern : str, optional
            The file pattern to use for filtering files. Default is None.

        Returns
        -------
        dict[str, pandas.DataFrame]
            A dictionary containing the data for each instrument.

        Raises
        ------
        ReaderError
            If no data is found for the specified instruments.
        """
        # Format instruments
        if instruments == 'all':
            instruments = self.instruments.names
        elif isinstance(instruments, str):
            instruments = [instruments.lower()]
        elif isinstance(instruments, Sequence):
            instruments = [i.lower() for i in instruments]

        # Determine group to read data from
        group = filesystem.get_group(group)

        # Read data for each instrument and store in dictionary
        data = {}
        for name in instruments:
            if name not in self.instruments:
                raise errors.InstrumentNotFoundError(name, self.instruments)

            instrument = self.instruments[name]

            try:
                data[name] = instrument.read_data(group, lvl, time_range,
                                                  num_processes, file_pattern)
            except errors.ReaderError as e:
                vprint(f'Error reading {instrument} data from {group} groupspace: {e}')

        if not data:
            raise errors.ReaderError(f'No data found for {instruments} at {self.SID} in {group} groupspace.')

        return data

    def get_obs(self, pollutants: _all_or_mult_strs = 'all',
                format: Literal['wide'] | Literal['long'] = 'wide',
                group: str | None = None,
                time_range: TimeRange._input_types = None,
                num_processes: int | Literal['max'] = 1
                ) -> pd.DataFrame:
        """
        Get observations for each pollutant, combining instruments by pollutants. 

        Parameters
        ----------
        pollutants : str or list of str, optional
            pollutants to read. If 'all', read all pollutants. Default is 'all'.
        format : str, optional
            Format of the data to return. Default is 'wide'.
        group : str, optional
            Research group to read data from. Default is None which uses the default group.
        time_range : str | list[Union[str, dt.datetime, None]] | tuple[Union[str, dt.datetime, None], Union[str, dt.datetime, None]] | slice | None
            The time range to read data. Default is None which reads all available data.
        num_processes : int, optional
            Number of processes to use for reading data. Default is 1.

        Returns
        -------
        Union[Dict[str, pandas.DataFrame], pandas.DataFrame]
            A dictionary of dataframes, one for each level of data read, or a single dataframe if only one level was read.
            The keys of the dictionary are the names of the levels ('calibrated', 'qaqc', 'raw'), and the values are the
            corresponding dataframes. If only one level was read, the method returns the corresponding dataframe directly.
        """
        lvl = 'final'

        if pollutants == 'all':
            pollutants = self.pollutants
        elif isinstance(pollutants, str):
            pollutants = [pollutants.upper()]
        elif isinstance(pollutants, Sequence):
            pollutants = [p.upper() for p in pollutants]

        if any(p not in self.pollutants for p in pollutants):
            raise ValueError(f"Invalid pollutant(s): '{set(pollutants) - set(self.pollutants)}'")

        # Get instruments for each pollutant
        instruments_to_read = {
            instrument.name for pollutant in pollutants 
            for instrument in self.pollutant_instruments[pollutant]
        }

        # Read data
        data = self.read_data(instruments_to_read, group,
                              lvl, time_range, num_processes)

        # Reshape data
        vprint('Combining data by pollutant...')
        if format == 'wide':
            obs = pd.concat(data.values())
            obs = obs.filter(regex='|'.join(pollutants))  # filter columns by pollutants
            obs = obs.dropna(how='all')
        elif format == 'long':
            melted_dfs = []
            for instrument, df in data.items():
                df_reset = df.reset_index()
                melted_df = df_reset.melt(id_vars='Time_UTC', value_vars=df.columns,
                                var_name='pollutant', value_name='value')
                melted_dfs.append(melted_df)
            obs = pd.concat(melted_dfs)
            # Filter columns by pollutants
            obs = obs[obs['pollutant'].str.contains('|'.join(pollutants))]
            obs.dropna(subset='value', inplace=True)
            obs.set_index('Time_UTC', inplace=True)
        else:
            raise ValueError(f"Invalid format '{format}'. Must be 'wide' or 'long'.")

        return obs.sort_index()

    def get_recent_obs(self, recent: str | dt.timedelta = dt.timedelta(days=10),
                       pollutants: _all_or_mult_strs = 'all',
                       format: Literal['wide'] | Literal['long'] = 'wide',
                       group: str | None = None) -> pd.DataFrame:
        '''
        Get recent observations from site instruments.

        Parameters
        ----------
        recent : str or datetime.timedelta, optional
            Time range to get recent observations. Default is 10 days.
        pollutants : str or list of str, optional
            Pollutants to read. If 'all', read all pollutants. Default is 'all'.
        format : str, optional
            Format of the data to return. Default is 'wide'.
        group : str, optional
            Research group to read data from. Defaults to None which uses the default group.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing recent observations from site instruments.
        '''
        if isinstance(recent, str):
            recent = pd.to_timedelta(recent)
        start_time = dt.datetime.now(timezone.utc).replace(tzinfo=None) - recent
        return self.get_obs(pollutants, format, group, [start_time, None])


class MobileSite(Site):
    """
    A class representing a mobile site where atmospheric measurements are taken.
    
    Parameters
    ----------
    SID : str
        The site identifier.
    config : dict
        A dictionary containing configuration information for the site:
        SID: {
            is_mobile: true,
            instruments: {
                instrument: {...}
            }
            ...
        }
    """
    _pilot_sites = ['trx01', 'trx02']

    @staticmethod
    def merge_gps(obs: pd.DataFrame, gps: pd.DataFrame,
                    on: str | None = None,
                    obs_on: str | None = None, gps_on: str | None = None
                    ) -> pd.DataFrame:
        """
        Merge observation data with location data from GPS.

        Parameters
        ----------
        obs (pd.DataFrame): The observation data.
        gps (pd.DataFrame): The GPS location data.
        on (str, optional): The column name to merge on. Defaults to 'Time_UTC'.
        obs_on (str, optional): The column name in the observation data to merge on. If not specified, it will use the value of 'on'.
        gps_on (str, optional): The column name in the GPS data to merge on. If not specified, it will use the value of 'on'.

        Returns
        -------
        pd.DataFrame: The merged data with added location information.
        """
        def truncate(time):
            return time.dt.floor('s')

        vprint('Merging obs data with location data from gps...')

        # Reset datetime index
        obs = obs.reset_index()
        gps = gps.reset_index()

        # Merge on Time_UTC by default unless specified
        if on is None:
            on = 'Time_UTC'
        obs_on = obs_on or on
        gps_on = gps_on or on

        # Convert to datetime
        obs[obs_on] = pd.to_datetime(obs[obs_on], errors='coerce')
        gps[gps_on] = pd.to_datetime(gps[gps_on], errors='coerce')

        # Drop rows with missing obs, time, or location
        obs.dropna(how='all', inplace=True)
        gps.dropna(subset=[gps_on, 'Latitude_deg', 'Longitude_deg'],
                    inplace=True)

        # Truncate time to seconds
        obs[obs_on] = truncate(obs[obs_on])
        gps[gps_on] = truncate(gps[gps_on])

        # Perform merge
        obs = obs.merge(gps, how='inner', left_on=obs_on, right_on=gps_on,
                        suffixes=('', '_gps'))

        # Set Time_UTC as index
        obs.set_index('Time_UTC', inplace=True)

        # Convert to geodataframe
        obs = gpd.GeoDataFrame(obs, crs='EPSG:4326',
                               geometry=gpd.points_from_xy(obs.Longitude_deg,
                                                           obs.Latitude_deg))

        return obs

    def get_obs(self, pollutants: _all_or_mult_strs = 'all',
                format: Literal['wide'] | Literal['long'] = 'wide',
                group: str | None = None, 
                time_range: TimeRange._input_types = None,
                num_processes: int | Literal['max'] = 1
                ) -> pd.DataFrame:
        """
        Get mobile site observations for each pollutant, combining instruments by pollutants,
        and merging location data from GPS.

        Parameters
        ----------
        pollutants : str or list of str, optional
            pollutants to read. If 'all', read all pollutants. Default is 'all'.
        format : str, optional
            Format of the data to return. Default is 'wide'.
        group : str, optional
            Research group to read data from. Default is None which uses the default group.
        time_range : list of str, optional
            Time range to read data. Default is None.
        num_processes : int, optional
            Number of processes to use for reading data. Default is 1.

        Returns
        -------
        pandas.DataFrame
            A dataframe containing mobile site observations for each pollutant
            with location data merged.
        """
        # Determine group to read data from
        group = filesystem.get_group(group)

        # Read data
        obs = super().get_obs(pollutants, format, group, time_range, num_processes)
        gps = self.read_data('gps', group, 'final', time_range, num_processes)['gps']

        # Merge gps data with obs data
        if group == 'lin':
            # Can't always trust the pi's time for lin-group mobile data
            # but Pi_Time connects the gps data to the obs data
            # so we'll merge on Pi_Time and use Time_UTC from gps as the time
            merge_on = 'Pi_Time'
            obs.index.name = 'Pi_Time'
        elif group == 'horel':
            merge_on = 'Time_UTC'
        else:  # FIXME just check for lin group?
            raise ValueError(f"Invalid group '{group}'. Must be 'lin' or 'horel'.")
        obs = MobileSite.merge_gps(obs, gps, on=merge_on)

        if group == 'lin':
            obs.drop(columns=['Pi_Time'], inplace=True)

        return obs

    @staticmethod
    def plot(obs, ax=None):
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        # FIXME is this the best way to do this?
        obs['lon'] = obs.Longitude_deg.round(3)
        obs['lat'] = obs.Latitude_deg.round(3)

        # keep only most recent for each lat/lon

        if ax is not None:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

        # ax.set_extent([SLV_bounds[0], SLV_bounds[2],
        #                SLV_bounds[1], SLV_bounds[3]], crs=ccrs.PlateCarree())
        
        return ax

