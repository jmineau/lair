"""
John Lin group - UUCON, TRAX, etc.

This module contains classes and functions for working with the Lin group data in the CHPC UATAQ filesystem.
"""

from copy import deepcopy
import json
import numpy as np
import os
import pandas as pd
import re
import subprocess
from typing import List

from lair.config import HOME, vprint
from lair.uataq.errors import DataFileInitializationError, ParserError
import lair.uataq.filesystem._filesystem as filesystem
from lair.utils.clock import TimeRange
from lair.utils.geo import dms2dd
from lair.utils.records import list_files

# TODO: check for reprocessing

#: Directory for Lin group measurements.
MEASUREMENTS_DIR: str = os.path.join(HOME, 'lin-group20', 'measurements')
#: Directory for Lin group pipeline configuration.
CONFIG_DIR: str = os.path.join(MEASUREMENTS_DIR, 'pipeline', 'config')
#: Directory for Lin group data.
DATA_DIR: str = os.path.join(MEASUREMENTS_DIR, 'data')

with open(os.path.join(CONFIG_DIR, 'data_config.json')) as data_config_file:
    #: Data configuration.
    DATA_CONFIG: dict = json.load(data_config_file)
    DATA_CONFIG['lgr_ugga_manual_cal'] = DATA_CONFIG['lgr_ugga']  # manual cal has same format

site_config_path = os.path.join(CONFIG_DIR, 'site_config.csv')
if not os.path.exists(site_config_path):
    site_config_path = 'https://raw.githubusercontent.com/uataq/data-pipeline/main/config/site_config.csv'
#: Site configuration.
SITE_CONFIG = pd.read_csv(site_config_path, skipinitialspace=True, index_col='stid')

#: Lin to UATAQ column mapping
column_mapping: dict[str, dict[str, str]] = {
    '2b_205': {
        'o3_ppb':       'O3_ppb',
        't_c':          'Internal_T_C',
        'Cavity_T_C':   'Internal_T_C',
        'p_hpa':        'Internal_P_hPa',
        'Cavity_P_hPa': 'Internal_P_hPa',
        'flow_ccpm':    'Flow_mLpm',
        'Flow_CCmin':   'Flow_mLpm'
    },
    'gps': {
        'inst_date':      'Instrument_Date',
        'inst_time':      'Instrument_Time',
        'fix_quality':    'Fix_Quality',
        'n_sat':          'N_Satellites',
        'N_Sat':          'N_Satellites',
        'altitude_amsl':  'Altitude_msl',
        'speed_kt':       'Speed_kt',
        'true_course':    'True_Course',
        'status':         'Status'
    },
    'licor_6262': {
        'RECORD':                  'Record',
        'batt_volt_Min':           'Battery_Voltage_V',
        'PTemp_Avg':               'Logger_T_C',
        'Panel_T_C':               'Logger_T_C',
        'Room_T_Avg':              'Ambient_T_C',
        'IRGA_T_Avg':              'Instrument_T_C',
        'Cavity_T_C_IRGA':         'Instrument_T_C',
        'IRGA_P_Avg':              'Internal_P_kPa',
        'Cavity_P_kPa_IRGA':       'Internal_P_kPa',
        'MF_Controller_mLmin_Avg': 'Flow_mLpm',
        'Flow_mLmin':              'Flow_mLpm',
        'PressureVolt_Avg':        'Internal_P_mV',
        'Cavity_P_mV':             'Internal_P_mV',
        'RH_voltage_Avg':          'Internal_RH_mV',
        'Cavity_RH_mV':            'Internal_RH_mV',
        'Gas_T_Avg':               'Internal_T_C',
        'Cavity_T_C':              'Internal_T_C',
        'rawCO2_Voltage_Avg':      'Analog_CO2_ppm',
        'CO2_Analog_ppm':          'Analog_CO2_ppm',
        'rawCO2_Avg':              'CO2_ppm',
        'rawH2O_Avg':              'Analog_H2O_ppm',
        'H2O_ppth_IRGA':           'Analog_H2O_ppm',  # TODO check this
        'Cavity_RH_pct':           'Internal_RH_pct',
        'Cavity_P_Pa':             'Analog_Internal_P_Pa'
    },
    'lgr_no2': {
        'inst_time':     'Instrument_Time',
        'Pressure_torr': 'Internal_P_torr',
        'Cavity_P_torr': 'Internal_P_torr',
        # do i want to drop the -1
    },
    'lgr_ugga': {
        'GasP_torr':        'Internal_P_torr',
        'Cavity_P_torr':    'Internal_P_torr',
        'GasP_torr_sd':     'Internal_P_torr_sd',
        'Cavity_P_torr_sd': 'Internal_P_torr_sd',
        'GasT_C':           'Internal_T_C',
        'Cavity_T_C':       'Internal_T_C',
        'GasT_C_sd':        'Internal_T_C_sd',
        'Cavity_T_C_sd':    'Internal_T_C_sd',
        'AmbT_C':           'Ambient_T_C',  # FIXME should this actually be called ambient??
        'AmbT_C_sd':        'Ambient_T_C_sd',
        'RD0_us':           'RingDown0_us',
        'RD0_us_sd':        'RingDown0_us_sd',
        'RD1_us':           'RingDown1_us',
        'RD1_us_sd':        'RingDown1_us_sd'
    },
    'magee_ae33': {
        'bc1_ngm3':  'BC1_ngm3',
        'bc2_ngm3':  'BC2_ngm3',
        'bc3_ngm3':  'BC3_ngm3',
        'bc4_ngm3':  'BC4_ngm3',
        'bc5_ngm3':  'BC5_ngm3',
        'bc6_ngm3':  'BC6_ngm3',
        'bc7_ngm3':  'BC7_ngm3',
        'flow_lpm':  'Flow_Lpm',
        'Flow_Lmin': 'Flow_Lpm'
    },
    'metone_es642': {
        'RECORD':        'Record',
        'batt_volt_Min': 'Battery_Voltage_V',
        'PTemp_Avg':     'Logger_T_C',
        'PM_25_Avg':     'PM2.5_ugm3',
        'pm25_mgm3':     'PM2.5_mgm3',
        'Flow_Avg':      'Flow_Lpm',
        'flow_lpm':      'Flow_Lpm',
        'Flow_Lmin':     'Flow_Lpm',
        'Temp_Avg':      'Ambient_T_C',
        't_c':           'Ambient_T_C',
        'RH_Avg':        'Internal_RH_pct',
        'rh_pct':        'Internal_RH_pct',
        'Cavity_RH_pct': 'Internal_RH_pct',
        'BP_Avg':        'Ambient_P_hPa',
        'pres_hpa':      'Ambient_P_hPa',
        'status':        'Status',
        'checksum':      'Checksum'
    },
    'teledyne_t200': {
        'nox_std_ppb':     'NOx_ppb_std',
        'flow_ccm':        'Flow_mLpm',
        'Flow_CCmin':      'Flow_mLpm',
        'o3_flow_ccm':     'O3_Flow_mLpm',
        'O3_Flow_CCmin':   'O3_Flow_mLpm',
        'pmt_mv':          'PMT_mV',
        'pmt_norm_mv':     'PMT_Norm_mV',
        'azero_mv':        'Azero_mV',
        'hvps_v':          'HVPS_V',
        'rcell_t_c':       'RCell_T_C',
        'RCel_T_C':        'RCell_T_C',
        'box_t_c':         'Instrument_T_C',
        'Box_T_C':         'Instrument_T_C',
        'pmt_t_c':         'PMT_T_C',
        'moly_t_c':        'Moly_T_C',
        'rcel_pres_inhga': 'RCell_P_inHgA',
        'RCel_Pres_inHgA': 'RCell_P_inHgA',
        'samp_pres_inhga': 'Internal_P_inHgA',
        'Samp_Pres_inHgA': 'Internal_P_inHgA',
        'nox_slope':       'NOx_Slope',
        'nox_offset_mv':   'NOx_Offset_mV',
        'no_slope':        'NO_Slope',
        'no_offset_mv':    'NO_Offset_mV',
        'no2_ppb':         'NO2_ppb',
        'nox_ppb':         'NOx_ppb',
        'no_ppb':          'NO_ppb',
        'test_mv':         'Test_mV'
    },
    'teledyne_t300': {
        'co_std_ppb':      'CO_ppb_std',
        'CO_std_ppb':      'CO_ppb_std',
        'co_meas_mv':      'CO_Meas_mV',
        'co_ref_mv':       'CO_Ref_mV',
        'mr_ratio':        'MR_Ratio',
        'samp_pres_inhga': 'Internal_P_inHgA',
        'Samp_Pres_inHgA': 'Internal_P_inHgA',
        'flow_ccm':        'Flow_mLpm',
        'Flow_CCmin':      'Flow_mLpm',
        'samp_t_c':        'Internal_T_C',
        'Samp_T_C':        'Internal_T_C',
        'bench_t_c':       'Bench_T_C',
        'wheel_t_c':       'Wheel_T_C',
        'pht_drive_mv':    'PHT_Drive_mV',
        'slope':           'Slope',
        'slope1':          'Slope1',
        'slope2':          'Slope2',
        'offset':          'Offset',
        'offset1':         'Offset1',
        'offset2':         'Offset2',
        'co_ppb':          'CO_ppb',
        'test_mv':         'Test_mV'
    },
    'teledyne_t400': {
        'o3_std_ppb':      'O3_ppb_std',
        'o3_std_ppb':      'O3_ppb_std',
        'o3_meas_mv':      'O3_Meas_mV',
        'o3_ref_mv':       'O3_Ref_mV',
        'o3_gen_mv':       'O3_Gen_mV',
        'o3_drive_mv':     'O3_Drive_mV',
        'photo_power_mv':  'Photo_Power_mV',
        'samp_pres_inhga': 'Internal_P_inHgA',
        'Samp_Pres_inHgA': 'Internal_P_inHgA',
        'flow_ccm':        'Flow_mLpm',
        'Flow_CCmin':      'Flow_mLpm',
        'samp_t_c':        'Internal_T_C',
        'Samp_T_C':        'Internal_T_C',
        'photo_lamp_t_c':  'Photo_Lamp_T_C',
        'o3_scrub_t_c':    'O3_Scrub_T_C',
        'o3_gen_t_c':      'O3_Gen_T_C',
        'box_t_c':         'Instrument_T_C',
        'Box_T_C':         'Instrument_T_C',
        'slope':           'Slope',
        'offset':          'Offset',
        'o3_ppb':          'O3_ppb',
        'test_mv':         'Test_mV'
    },
    'teledyne_t500u': {
        'phase_t_c':       'Phase_T_C',
        'bench_phase_s':   'Bench_Phase_s',
        'meas_l_mm':       'Measure_Loss_Mm',
        'Meas_L_mm':       'Measure_Loss_Mm',
        'aref_l_mm':       'ARef_Loss_Mm',
        'ARef_L_mm':       'ARef_Loss_Mm',
        'samp_pres_inhga': 'Internal_P_inHgA',
        'Samp_Pres_inHgA': 'Internal_P_inHgA',
        'samp_temp_c':     'Internal_T_C',
        'Samp_Temp_C':     'Internal_T_C',
        'bench_t_c':       'Bench_T_C',
        'box_t_c':         'Instrument_T_C',
        'Box_T_C':         'Instrument_T_C',
        'no2_slope':       'NO2_Slope',
        'no2_offset_mv':   'NO2_Offset_mV',
        'no2_ppb':         'NO2_ppb',
        'no2_std_ppb':     'NO2_ppb_std',
        'NO2_std_ppb':     'NO2_ppb_std',
        'test_mv':         'Test_mV'
    },
    'teom_1400ab': {
        'pm25_ugm3':       'PM2.5_ugm3',
        'pm25_ugm3_30min': 'PM2.5_ugm3_30min',
        'pm25_ugm3_1hr':   'PM2.5_ugm3_1hr',
    }
}
column_mapping['licor_7000'] = column_mapping['licor_6262']


class LinDatFile(filesystem.DataFile):
    """
    A class for parsing Lin data files.

    Attributes
    ----------
    path : str
        The file path.
    period : pd.Period
        The period of the data file.
    logger : str
        The logger name.
    date_slicer : slice
        A slice object for extracting the date from the file name.
    file_freq : str
        The file frequency.
    ext : str
        The file extension.

    Methods
    -------
    parse()
        Parse the data file.
    """
    logger = 'campbellsci'
    date_slicer = slice(7)
    file_freq = 'M'
    ext = 'dat'

    def __init__(self, path: str):
        super().__init__(path)

        # Get instrument name and lvl from file path
        self.instrument = self.get_instrument_name()
        self.lvl = self.get_lvl()

    def get_instrument_name(self) -> str:
        """
        Get the instrument name from the file path.

        Returns
        -------
        str
            The instrument name.
        """
        # Instrument name is the name of the directory two levels up from the file path
        return os.path.basename(os.path.dirname(os.path.dirname(self.path)))

    def get_lvl(self) -> str:
        """
        Get the data level from the file path.

        Returns
        -------
        str
            The data level.
        """
        # Data level is the name of the directory one level up from the file path
        return os.path.basename(os.path.dirname(self.path))

    def parse(self) -> pd.DataFrame:
        """
        Parse a Lin data file.

        Returns
        -------
        pd.DataFrame
            The parsed data.
        """
        vprint(f'Parsing {os.path.relpath(self.path, DATA_DIR)}')

        data = pd.read_csv(self.path, on_bad_lines='skip')
        
        if data.columns[0] is not 'TIMESTAMP':
            col_names = DATA_CONFIG[self.instrument][self.lvl]['col_names']
            data = pd.DataFrame(np.vstack([
                    data.columns,
                    data.values
                    ]), columns=col_names)

        # Format time col
        data.rename(columns={'TIMESTAMP': 'Time_UTC'}, inplace=True)
        data['Time_UTC'] = pd.to_datetime(data.Time_UTC, errors='coerce',
                                          format='ISO8601')

        return data


class LGR_UGGA_File(filesystem.DataFile):
    """
    A class for parsing LGR UGGA data files.

    Attributes
    ----------
    path : str
        The file path.
    period : pd.Period
        The period of the data file.
    logger : str
        The logger name.
    date_slicer : slice
        A slice object for extracting the date from the file name.
    file_freq : str
        The file frequency.
    ext : str
        The file extension.

    Methods
    -------
    get_meta(path)
        Get meta data from UGGA file.
    get_serial(path)
        Get serial number of UGGA from file.
    get_files(SID, instrument='lgr_ugga', lvl='raw')
        Get list of UGGA files for a given site.
    parse()
        Parse files transferred directly from LGR software.
    """
    logger = 'lgr_ugga'
    file_freq = 'D'
    ext = 'txt'

    version_date_formats = {
        '904M': {
            'slicer': slice(3, 12),
            'format': '%d%b%Y'
        },
        '2f90039': {
            'slicer': slice(4, 14),
            'format': '%Y-%m-%d'
        },
        'cf32204': {
            'slicer': slice(4, 14),
            'format': '%Y-%m-%d'
        }
    }

    def __init__(self, path: str):
        self.path = path
        self.meta = self.get_meta(path)
        self.serial = self.meta['SN']
        self.version = self.meta['VC']

        # Get date format based on version
        date_format = self.version_date_formats[self.version]
        self.date_slicer = date_format['slicer']
        self.date_format = date_format['format']

        # Get date from file name
        fname = os.path.basename(path)
        date_str = fname[self.date_slicer]
        self.period = pd.Period(date_str, freq=self.file_freq)

    @staticmethod
    def get_meta(path) -> dict[str, str]:
        """
        Get meta data from UGGA file

        Parameters
        ----------
        path : str
            The path to the UGGA file.

        Returns
        -------
        dict[str, str]
            A dictionary containing the meta data.
        """
        pattern = (r'VC:(?P<VC>\w+)\s'
                   r'BD:(?P<BD>[a-zA-Z]{3}\s\d{2}\s\d{4})\s'
                   r'SN:(?P<SN>.+)')

        header = subprocess.getoutput(f'head -n 1 {path}')
        try:
            meta = re.match(pattern, header).groupdict()
        except AttributeError:
            raise DataFileInitializationError(f'Failed to read meta data from {path}')
        return meta

    @staticmethod
    def get_serial(path) -> str:
        """
        Get serial number of UGGA from file.

        Parameters
        ----------
        path : str
            The path to the UGGA file.

        Returns
        -------
        str
            The serial number.
        """
        return LGR_UGGA_File.get_meta(path)['SN']

    @staticmethod
    def get_files(SID: str, instrument: str = 'lgr_ugga', lvl: str = 'raw') -> List[str]:
        """
        Get list of UGGA files for a given site.

        Parameters
        ----------
        SID : str
            The site ID.
        instrument : str
            The instrument name. Defaults to 'lgr_ugga'.
        lvl : str
            The data level. Defaults to 'raw'.

        Returns
        -------
        list[str]
            A list of file paths.
        """
        assert lvl == 'raw', 'Only raw data is stored in UGGA files'
        data_path = LinGroup.data_path(SID, instrument, lvl)
        pattern = '*f????.txt'
        return list_files(data_path, pattern=pattern, full_names=True, recursive=True)

    def parse(self):
        """
        Parse files transferred directly from LGR software.

        Returns
        -------
        pd.DataFrame
            The parsed data.
        """
        vprint(f'Parsing {os.path.relpath(self.path, DATA_DIR)}')

        # Adapt column names depending on LGR software version.
        #  2013-2014 version has 23 columns
        #  2014+ version has 24 columns (MIU split into valve and description)

        data_config = deepcopy(DATA_CONFIG['lgr_ugga']['raw'])

        r2py_types = {'c': str,
                      'd': float,
                      'T': str}

        col_names = data_config['col_names']
        col_types = data_config['col_types']
        col_types = {name: r2py_types[t] for name, t
                    in zip(col_names, col_types)}

        cols = subprocess.getoutput(f'head -n 2 {self.path} | tail -n 1')
        ndelim = cols.count(',')

        if ndelim == 0:
            raise ParserError('No deliminators in header!')
        elif ndelim == 23:
            # col_names config has 23 columns, temporarily add fill column
            col_names.insert(22, 'fillvalue')
            col_types['fillvalue'] = float

        # Check for incomplete final row (probably power issue)
        last_line = subprocess.getoutput(f'tail -n 1 {self.path}')
        footer = 1 if last_line.count(',') < 22 else 0

        # Read file assigning cols and dtypes
        data = pd.read_csv(self.path, header=1, names=col_names, dtype=col_types,
                         on_bad_lines='skip', na_values=['TO', ''],
                         skipinitialspace=True, skipfooter=footer,
                         engine='python')

        # Update columns now that data has been read in
        ncols = len(data.columns)
        if (ncols < 23) | (ncols > 24):
            raise ParserError('Too few or too many columns!')
        elif ncols == 24:
            #  drop the valve (fill) column
            data.drop(columns=data.columns[22], inplace=True)

        # Format time
        data.Time_UTC = pd.to_datetime(data.Time_UTC.str.strip(),
                                     format='%m/%d/%Y %H:%M:%S.%f',
                                     errors='coerce')

        return data


class AirTrendFile(filesystem.DataFile):
    """
    A class for parsing AirTrend data files.

    Attributes
    ----------
    path : str
        The file path.
    period : pd.Period
        The period of the data file.
    logger : str
        The logger name.
    date_slicer : slice
        A slice object for extracting the date from the file name.
    file_freq : str
        The file frequency.
    ext : str
        The file extension.
    config : dict
        The data configuration.

    Methods
    -------
    parse()
        Parse the data file.
    """

    logger = 'air-trend'
    date_slicer = slice(10)
    file_freq = 'D'
    ext = 'csv'

    def __init__(self, path: str):
        # Store data config labels
        self.config = {}

        # instrument name is the name of the directory two levels up
        instrument_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        
        # Some instruments have special configurations indicated by their name
        # but use the same column structure as the default configuration
        # Example: lgr_ugga_manual_cal for trx01
        self.config['instrument'] = re.match('|'.join(DATA_CONFIG.keys()),
                                             instrument_name).group()

        # Extract filename extension from custom air trend file handlers
        # This is necessary for instruments like the gps which have different types of files
        air_trend_ext = re.match(r'\d{4}-\d{2}-\d{2}_?([^\s]+)?\.csv',
                                 os.path.basename(path)).group(1)
        self.config['lvl'] = 'air_trend' if air_trend_ext is None else f'air_trend_{air_trend_ext}'

        super().__init__(path)

    def parse(self) -> pd.DataFrame:
        """
        Parse an AirTrend file.

        Returns
        -------
        pd.DataFrame
            The parsed data.
        """
        vprint(f'Parsing {os.path.relpath(self.path, DATA_DIR)}')

        data_config = DATA_CONFIG[self.config['instrument']][self.config['lvl']]

        col_names = data_config['col_names']
        col_types = data_config['col_types']

        data = pd.read_csv(self.path, names=col_names, header=0,
                           na_values=['XXXX'],
                           on_bad_lines='skip', low_memory=False)

        # Format time
        data['time'] = pd.to_datetime(data.time, errors='coerce')
        data.rename(columns={'time': 'Time_UTC'}, inplace=True)

        # Coerce numeric columns
        numeric_cols = [col for col, dtype in zip(col_names, col_types) if dtype == 'd']
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

        return data


class LinGroup(filesystem.GroupSpace):
    """
    A class representing the Lin group space in the CHPC UATAQ filesystem.

    Attributes
    ----------
    name : str
        The group name.
    datafiles : dict[str, Type[DataFile]]
        A dictionary mapping datafile keys to DataFile classes.
    data_dir : str
        The path to the data directory.
    data_config : dict
        The data configuration.

    Methods
    -------
    get_highest_lvl(SID, instrument)
        Get the highest level of data for a given site and instrument.
    data_path(SID, instrument, lvl)
        Get the path to the data directory for a given site, instrument, and level.
    get_files(SID, instrument, lvl, logger)
        Get list of file paths for a given site, instrument, and level.
    get_datafile_key(instrument, lvl, logger)
        Get the datafile key based on the instrument, level, and logger.
    get_datafiles(SID, instrument, lvl, logger, time_range, pattern)
        Get list of data files for a given level and time range.
    """
    name = 'lin'

    datafiles = {
        'data-pipeline': LinDatFile,
        'campbellsci': LinDatFile,
        'air-trend': AirTrendFile,
        'lgr_ugga': LGR_UGGA_File
    }

    data_dir = DATA_DIR
    data_config = deepcopy(DATA_CONFIG)

    @staticmethod
    def get_highest_lvl(SID: str, instrument: str) -> str:
        path = os.path.join(DATA_DIR, SID.lower(), instrument)
        inst_lvls = [d for d in os.listdir(path) if d in filesystem.lvls]
        lvl = max(inst_lvls, key=lambda d: filesystem.lvls[d])
        return lvl

    @staticmethod
    def data_path(SID, instrument, lvl) -> str:
        """
        Get the path to the data directory for a given site, instrument, and level.

        Parameters
        ----------
        SID : str
            The site ID.
        instrument : str
            The instrument name.
        lvl : str
            The data level.

        Returns
        -------
        str
            The path to the data directory.
        """
        return os.path.join(DATA_DIR, SID.lower(), instrument, lvl)

    def get_files(self, SID: str, instrument: str, lvl: str,
                  logger: str = 'campbellsci') -> List[str]:
        # Raw lgr_ugga files are stored subdirectories with other files
        if lvl == 'raw' and logger == 'lgr_ugga':
            return LGR_UGGA_File.get_files(SID, instrument, lvl)

        data_path = LinGroup.data_path(SID, instrument, lvl)
        return list_files(data_path, full_names=True)

    def get_datafile_key(self, instrument: str, lvl: str, logger: str) -> str:
        key = logger if lvl == 'raw' else 'data-pipeline'
        return key

    def get_datafiles(self, SID: str, instrument: str, lvl: str, logger: str,
                  time_range: TimeRange, pattern: str | None = None
                  ) -> List[filesystem.DataFile]:
        # Custom handling for raw Lin files
        if lvl == 'raw':
            if SID.startswith('TRX'):
                print('Warning: Time_UTC may not be accurate for mobile raw Lin data.')
            if logger == 'lgr_ugga':
                vprint('Adding a day on either side of time range to ensure all data is included.')
                # Raw lgr_ugga files names dont necessarily match the period
                # Add a day on either side to ensure we get all the data
                # Construct new TimeRange to avoid modifing original range
                one_day = pd.Timedelta(days=1)
                start = time_range.start - one_day if time_range.start else None
                stop = time_range.stop + one_day if time_range.stop else None
                time_range = TimeRange(start=start, stop=stop)
            elif instrument == 'gps':
                # Set default file pattern for raw lin gps data
                pattern = pattern or 'gpgga'

        return super().get_datafiles(SID, instrument, lvl, logger, time_range, pattern)

    @staticmethod
    def standardize_data(instrument: str, data: pd.DataFrame
                         ) -> pd.DataFrame:
        mapping = column_mapping.get(instrument, {})

        ### Column specific manipulations ###

        if instrument == 'gps':
            if 'latitude_dm' in data.columns:
                # convert dms to dd
                data['Latitude_deg'] = data.apply(lambda row: 
                                                  dms2dd(d=row.latitude_dm // 100,
                                                         m=row.latitude_dm % 100),
                                                  axis=1)
                data['Longitude_deg'] = data.apply(lambda row: 
                                                   dms2dd(d=row.longitude_dm // 100,
                                                          m=row.longitude_dm % 100),
                                                   axis=1)
                data['Latitude_deg'] *= np.where(data.n_s == 'S', -1, 1)
                data['Longitude_deg'] *= np.where(data.e_w == 'W', -1, 1)
                data.drop(columns=['latitude_dm', 'longitude_dm', 'n_s', 'e_w'],
                        inplace=True)
            for status in ['status', 'Status']:
                # Map status to binary
                if status in data.columns:
                    data[status] = data[status].map({'A': 1, 'V': 0})

        elif instrument == '2b_205':
            for flow in ['flow_ccpm', 'Flow_CCmin']:
                if flow in data.columns:
                    data[flow] = data[flow] / 1000
                    data.rename(columns={flow: 'Flow_Lpm'}, inplace=True)

        if 'pm25_mgm3' in data.columns:
            data['pm25_mgm3'] = data.pm25_mgm3 * 1000
            data.rename(columns={'pm25_mgm3': 'PM2.5_ugm3'}, inplace=True)

        return data.rename(columns=mapping)

filesystem.groups['lin'] = LinGroup()
