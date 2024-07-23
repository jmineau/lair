"""
John Horel group - MesoWest, TRAX/eBUS, etc.

This module contains classes and functions for working with the Horel group data in the CHPC UATAQ filesystem.
"""

from abc import ABCMeta
import numpy as np
import os
import pandas as pd
import tables as pytbls
from typing import Dict, List

from lair.config import HOME, vprint
from lair.uataq import errors
import lair.uataq.filesystem._filesystem as filesystem
from lair import units
from lair.utils.clock import TimeRange
from lair.utils.records import list_files

#: Horel group directory
HOREL_DIR: str = os.path.join(HOME, 'horel-group')

# UUTRAX directories
#   Consider UUTRAX project as part of UATAQ (includes TRX, BUS, RAIL, HAWTH)
#   TODO add rail and hawth
#   Includes a pilot phase which does not have QAQC data
#: UUTRAX directory
UUTRAX_DIR: str = os.path.join(HOREL_DIR, 'uutrax')
#: UUTRAX pilot directory
UUTRAX_PILOT_DIR: str = os.path.join(HOREL_DIR, 'uutrax_pilot')

#: Pilot phase time ranges for UUTRAX data
PILOT_PHASE: dict[str, TimeRange] = {
    'TRX01': TimeRange(start='2014-11-11',
                       stop=pd.Timestamp('2018-11-19T20:03:58')),
    'TRX02': TimeRange(start='2016-02-04',
                       stop=pd.Timestamp('2018-11-19T18:53:52'))
}

#: Data levels for UUTRAX data
lvl_data_dirs: dict[str, list] = {
    'raw': [UUTRAX_PILOT_DIR, UUTRAX_DIR],
    'qaqc': [UUTRAX_DIR],
    'final': [UUTRAX_DIR]
}

#: Horel to UATAQ column mapping
column_mapping: dict[str, dict[str, str]] = {
    '2b_205': {
        'OZNE':                        'O3_ppb',
        '2B_Ozone_Concentration':      'O3_ppb',
        'FL2B':                        'Flow_Lpm',
        '2B_Air_Flow_Rate':            'Flow_Lpm',
        'TC2B':                        'Internal_T_C',
        '2B_Internal_Air_Temperature': 'Internal_T_C',
        'PS2B':                        'Internal_P_hPa',
        '2B_Internal_Air_Pressure':    'Internal_P_hPa',
        'Ozone_Data_Flagged':          'QAQC_Flag'
    },
    '2b_405': {
        'NO1C':                           'NO_ppb',
        '2B405_NO_Concentration':         'NO_ppb',
        'NO2C':                           'NO2_ppb',
        '2B405_NO2_Concentration':        'NO2_ppb',
        'NOXC':                           'NOx_ppb',
        '2B405_NOX_Concentration':        'NOx_ppb',
        'TCNO':                           'Internal_T_C',
        '2B405_Internal_Air_Temperature': 'Internal_T_C',
        'PSNO':                           'Internal_P_hPa',
        '2B405_Internal_Air_Pressure':    'Internal_P_hPa',
        'FLNO':                           'Flow_Lpm',
        '2B405_Air_Flow_Rate':            'Flow_Lpm',
        'FO3N':                           'O3_Flow_mLpm',
        '2B405_Cell_O3_Flow_Rate':        'O3_Flow_mLpm'
    },
    'cr1000': {
        'VOLT':                        'Battery_Voltage_V',
        'Battery_Voltage':             'Battery_Voltage_V',
        'TICC':                        'Logger_T_C',
        'Bus_Box_Temperature':         'Logger_T_C',
        'Train_Box_Temperature':       'Logger_T_C',
        'TRNT':                        'Ambient_T_C',
        'Bus_Top_Temperature':         'Ambient_T_C',
        'Train_Top_Temperature':       'Ambient_T_C',
        'TRNR':                        'Ambient_RH_pct',
        'Bus_Top_Relative_Humidity':   'Ambient_RH_pct',
        'Train_Top_Relative_Humidity': 'Ambient_RH_pct'
    },
    'gps': {
        'GTIM':             'Instrument_Time',
        'GLAT':             'Latitude_deg',
        'Latitude':         'Latitude_deg',
        'GLON':             'Longitude_deg',
        'Longitude':        'Longitude_deg',
        'GELV':             'Altitude_msl',
        'Elevation':        'Altitude_msl',
        'RSPD':             'Speed_kt',
        'GPS_Speed':        'Speed_kt', 
        'RDIR':             'Course_deg',
        'GPS_Direction':    'Course_deg',
        'NSAT':             'N_Satellites',
        'RSTS':             'Status',
        'GPS_RMC_Valid':    'Status',
        'GPS_Data_Flagged': 'QAQC_Flag',
    },
    'metone_es405': {
        'PM01':                             'PM1_ugm3',
        'ES405_PM1_Concentration':          'PM1_ugm3',
        'PM25':                             'PM2.5_ugm3',
        'ES405_PM2.5_Concentration':        'PM2.5_ugm3',
        'PM04':                             'PM4_ugm3',
        'ES405_PM4_Concentration':          'PM4_ugm3',
        'PM10':                             'PM10_ugm3',
        'ES405_PM10_Concentration':         'PM10_ugm3',
        'FLOW':                             'Flow_Lpm',
        'ES405_Air_Flow_Rate':              'Flow_Lpm',
        'ITMP':                             'Internal_T_F',
        'ES405_Internal_Air_Temperature':   'Internal_T_C',
        'INRH':                             'Internal_RH_pct',
        'ES405_Internal_Relative_Humidity': 'Internal_RH_pct',
        'PRES':                             'Internal_P_hpa',
        'ES405_Internal_Air_Pressure':      'Internal_P_hPa',
        'ERRR':                             'Status',
        'ES405_Error_Code':                 'Status',
        'PM2.5_Data_Flagged':               'QAQC_Flag'
    },
    'metone_es642': {
        'PM25':                             'PM2.5_ugm3',
        'ES642_PM2.5_Concentration':        'PM2.5_ugm3',
        'FLOW':                             'Flow_Lpm',
        'ES642_Air_Flow_Rate':              'Flow_Lpm',
        'ITMP':                             'Ambient_T_F',
        'ES642_Internal_Air_Temperature':   'Ambient_T_C',
        'INRH':                             'Internal_RH_pct',
        'ES642_Internal_Relative_Humidity': 'Internal_RH_pct',
        'PRES':                             'Ambient_P_hpa',
        'ES642_Internal_Air_Pressure':      'Ambient_P_hPa',
        'ERRR':                             'Status',
        'ES642_Error_Code':                 'Status',
        'PM2.5_Data_Flagged':               'QAQC_Flag'
    }
}


class HorelFile(filesystem.DataFile, metaclass=ABCMeta):
    """
    Abstract base class for Horel data files.

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
    time_col : str
        The time column name.
    instrument : str
        The instrument name.

    Methods
    -------
    usecols(col)
        Check if a column should be used based on the instrument.
    convert_nodata(data, nodata=-9999.0)
        Convert NoData values to NaN.
    coerce_numeric(data, exclude='Time_UTC')
        Coerce columns to numeric.
    """
    date_slicer = slice(6, 13)
    file_freq = 'M'
    time_col: str

    def __init__(self, path: str, instrument: str):
        """
        Initialize a HorelFile subclass object.

        The instrument parameter is used to filter columns based on the instrument name.

        Parameters
        ----------
        path : str
            The file path.
        instrument : str
            The instrument name - used to filter columns.
        """
        super().__init__(path)
        self.instrument = instrument

    def usecols(self, col: str) -> bool:
        """
        Check if a column should be used based on the instrument.

        Parameters
        ----------
        col : str
            The column name.

        Returns
        -------
        bool
            True if the column should be used, False otherwise.
        """
        columns = [self.time_col] + [*column_mapping[self.instrument]]
        return col in columns

    def format_time(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Format the time column in the data DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            The data DataFrame.
        **kwargs : dict
            Additional keyword arguments to pass to pd.to_datetime.

        Returns
        -------
        pd.DataFrame
            The data DataFrame with the time column formatted as Time_UTC.
        """
        data[self.time_col] = pd.to_datetime(data[self.time_col],
                                             errors='coerce', **kwargs)
        return data.rename(columns={self.time_col: 'Time_UTC'})

    @staticmethod
    def convert_nodata(data: pd.DataFrame, nodata: float = -9999.0
                       ) -> pd.DataFrame:
        """
        Convert NoData values to NaN.

        Parameters
        ----------
        data : pd.DataFrame
            The data DataFrame.
        nodata : float
            The NoData value.

        Returns
        -------
        pd.DataFrame
            The data DataFrame with NoData values converted to NaN.
        """
        return data.replace(nodata, np.nan)

    @staticmethod
    def coerce_numeric(data, exclude: str | list[str] = 'Time_UTC'
                       ) -> pd.DataFrame:
        """
        Coerce columns to numeric.

        Parameters
        ----------
        data : pd.DataFrame
            The data DataFrame.
        exclude : str | Sequence[str]
            Columns to exclude from coercion.

        Returns
        -------
        pd.DataFrame
            The data DataFrame with columns coerced to numeric.
        """
        if isinstance(exclude, str):
            exclude = [exclude]
        return data.apply(lambda x: pd.to_numeric(x, errors='coerce')
                          if x.name not in exclude else x)


class HorelH5File(HorelFile):
    """
    Class for parsing H5 files from the Horel group.

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
    time_col : str
        The time column name.
    instrument : str
        The instrument name.

    Methods
    -------
    usecols(col)
        Check if a column should be used based on the instrument.
    convert_nodata(data, nodata=-9999.0)
        Convert NoData values to NaN.
    coerce_numeric(data, exclude='Time_UTC')
        Coerce columns to numeric.
    """
    logger = 'campbellsci'
    ext = 'h5'
    time_col = 'EPOCHTIME'

    def parse(self) -> pd.DataFrame:
        """
        Parse the H5 file and return a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the parsed data.
        """
        vprint(f'Parsing {os.path.relpath(self.path, HOREL_DIR)}')

        with pytbls.open_file(self.path, mode='r') as f:
            table = f.root['obsdata/observations']
            data = pd.DataFrame(table.read())

        # Subset columns by instrument
        data = data.loc[:, data.columns.map(self.usecols).to_list()]

        # Format time
        data = self.format_time(data, unit='s')

        # Convert horel-group NoData to None
        data = HorelFile.convert_nodata(data)

        # Coerce to numeric
        data = HorelFile.coerce_numeric(data)

        return data


class HorelCSVFile(HorelFile):
    """
    Class for parsing CSV files from the Horel group.

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
    time_col : str
        The time column name.
    instrument : str
        The instrument name.

    Methods
    -------
    usecols(col)
        Check if a column should be used based on the instrument.
    convert_nodata(data, nodata=-9999.0)
        Convert NoData values to NaN.
    coerce_numeric(data, exclude='Time_UTC')
        Coerce columns to numeric.
    parse()
        Parse the CSV file and return a DataFrame.
    """
    logger = 'horel-pipeline'
    ext = 'csv.gz'
    time_col = 'Timestamp'

    # TODO This could definitely be made more efficient - especially when reading multiple instruments
    # We should store the csv in memory until the end of the read_data call so we can just grab the columns we need
    def parse(self) -> pd.DataFrame:
        """
        Parse the CSV file and return a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the parsed data.
        """
        vprint(f'Parsing {os.path.relpath(self.path, HOREL_DIR)}')

        data = pd.read_csv(self.path, compression='gzip', skiprows=[1],
                           usecols=self.usecols)

        # Format time
        data = self.format_time(data, format='ISO8601')

        # Convert horel-group NoData to None
        data = HorelFile.convert_nodata(data)

        # Coerce to numeric
        data = HorelFile.coerce_numeric(data)

        # Parse QAQC flags
        # https://github.com/uataq/data-pipeline/blob/main/QAQC_Flags.md
        if self.instrument == 'gps':
            data['QAQC_Flag'] = 0
            if 'GPS_Data_Flagged' in data.columns:
                # GPS_Data_Flagged == 1 indicates if the vehicle is in/near storage
                data.loc[data.GPS_Data_Flagged == 1, 'QAQC_Flag'] = 20
                data.drop(columns=['GPS_Data_Flagged'], inplace=True)
            if 'GPS_RMC_Valid' in data.columns:
                # RMC Status message (valid: ['A', 1]; invalid: ['V', 0])
                data.loc[data.GPS_RMC_Valid == 0, 'QAQC_Flag'] = -23
                data.drop(columns=['GPS_RMC_Valid'], inplace=True)

        for flag in ['Ozone_Data_Flagged', 'PM2.5_Data_Flagged']:
            if flag in data.columns:
                # Replace 1 with -1 (negative flag means bad data)
                data[flag] = data[flag].replace(1, -1)
                data.rename(columns={flag: 'QAQC_Flag'}, inplace=True)

        return data


class HorelCSVFinalizedFile(HorelCSVFile):
    """
    Class for parsing finalized CSV

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
    time_col : str
        The time column name.
    final_patterns : list[str]
        A list of patterns to filter columns.
    instrument : str
        The instrument name.

    Methods
    -------
    usecols(col)
        Check if a column should be used based on the instrument.
    convert_nodata(data, nodata=-9999.0)
        Convert NoData values to NaN.
    coerce_numeric(data, exclude='Time_UTC')
        Coerce columns to numeric.
    parse()
        Parse the CSV file, finalize the data, and return a DataFrame.
    """
    final_patterns = ['UTC', 'Ambient', 'ppb', 'ugm3', 'deg', 'm_s']

    def parse(self) -> pd.DataFrame:
        """
        Parse the CSV file, finalize the data, and return a DataFrame.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the finalized data.
        """
        data = super().parse()
        
        # Need to standardize in order to filter columns
        # TODO this is a bit of a hack, but it works for now
        data = HorelGroup.standardize_data(self.instrument, data)

        if 'QAQC_Flag' in data.columns:
            # Drop rows with QAQC_Flag < 0
            data = data[data.QAQC_Flag >= 0]

        # Use final patterns to filter columns
        data = data.filter(regex='|'.join(self.final_patterns))

        # Drop rows without obs
        data.dropna(how='all', inplace=True)

        return data


class HorelGroup(filesystem.GroupSpace):
    """
    A class representing the Horel group data space in the CHPC UATAQ filesystem.

    Attributes
    ----------
    name : str
        The group name.
    datafiles : dict[str, Type[DataFile]]
        A dictionary mapping datafile keys to DataFile classes.

    Methods
    -------
    get_highest_lvl(SID, instrument)
        Get the highest data level for a given site and instrument.
    get_files(SID, instrument, lvl, logger)
        Get list of file paths for a given site, instrument, and level.
    get_datafile_key(instrument, lvl, logger)
        Get the datafile key based on the instrument, level, and logger.
    get_datafiles(SID, instrument, lvl, logger, time_range, pattern=None)
        Returns a list of data files for a given level and time range.
    """
    name = 'horel'

    datafiles = {
        'raw': HorelH5File,
        'qaqc': HorelCSVFile,
        'final': HorelCSVFinalizedFile
    }

    @staticmethod
    def get_highest_lvl(SID: str, instrument: str) -> str:
        finalized_dir = 'csv_finalized_ebus' if SID.startswith('BUS') else 'csv_finalized'
        # UUTRAX_PILOT_DIR is the pilot phase which does not have qaqc/final data
        site_files = list_files(os.path.join(UUTRAX_DIR, finalized_dir),
                                pattern=f'{SID.upper()}*')
        return 'final' if len(site_files) > 0 else 'raw'

    def get_files(self, SID: str, instrument: str, lvl: str,
                  logger: str = 'campbellsci') -> List[str]:
        # Map UATAQ instrument names to Horel instrument names to find the correct directories
        instrument_mapper = {
            '2b_205': '2b',
            '2b_405': 'nox',
            'gps': 'cr1000',
            'metone_es642': 'esampler',
            'metone_es405': 'esampler'  # both metones are called esampler in horel-group
        }
        instrument = instrument_mapper.get(instrument, instrument)

        data_dirs = lvl_data_dirs.get(lvl)
        if data_dirs is None:
            raise ValueError(f'Invalid data level: {lvl} for group: {self.name}')

        files = []
        for data_dir in data_dirs:
            if lvl == 'raw':
                data_path = os.path.join(data_dir, instrument)
            else:
                finalized_dir = 'csv_finalized_ebus' if SID.startswith('BUS') else 'csv_finalized'
                data_path = os.path.join(data_dir, finalized_dir)
            files.extend(list_files(data_path, pattern=f'*{SID.upper()}*', full_names=True))

        return files

    def get_datafile_key(self, instrument: str, lvl: str, logger: str) -> str:
        # The only logger for Horel data is campbellsci
        # The datafile key is based only on the level
        return lvl

    def get_datafiles(self, SID: str, instrument: str, lvl: str, logger: str,
                      time_range: TimeRange, pattern: str | None = None
                      ) -> list[filesystem.DataFile]:
        """
        Returns a list of data files for a given level and time range.
        Extends DataFile.get_datafiles by supplying the instrument name to the DataFile subclass.

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
            The time range.
        pattern : str | None
            The pattern to match file names.

        Returns
        -------
        list[DataFile]
            A list of data files.
        """
        # Pilot Phase Warning
        if (lvl != 'raw' and SID in PILOT_PHASE
            and (time_range.start is None or time_range.start < PILOT_PHASE[SID].stop)):
            print(f'Warning: No {lvl} data available for {SID} before pilot phase conclusion.')
            print(f'Use raw data for pilot phase: {PILOT_PHASE[SID].start} ~ {PILOT_PHASE[SID].stop}')

        DataFileClass = self.get_datafile_class(instrument, lvl, logger)

        files = self.get_files(SID, instrument, lvl, logger)

        datafiles = []
        for path in files:
            if path.endswith(DataFileClass.ext):
                try:
                    datafiles.append(DataFileClass(path, instrument))
                except errors.DataFileInitializationError as e:
                    vprint(f'Unable to initialize {DataFileClass.__name__} from {path}: {e}')
            continue

        return filesystem.filter_datafiles(datafiles, time_range, pattern)

    @staticmethod
    def standardize_data(instrument: str, data: pd.DataFrame
                         ) -> pd.DataFrame:
        mapping = column_mapping[instrument]

        ### Column specific manipulations ###

        if 'ITMP' in data.columns:
            # Convert from F to C
            data['ITMP'] = (data.ITMP.values * units('degF')).to(units('degC')).magnitude

            # Replace ITMP with instrument-specific column name
            new_col = mapping['ITMP'][:-1] + 'C'
            data.rename(columns={'ITMP': new_col}, inplace=True)

        return data.rename(columns=mapping)

filesystem.groups['horel'] = HorelGroup()
