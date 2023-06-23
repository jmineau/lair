#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:40:10 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for LGR UGGA instrument
"""

from functools import partial
import os
import pandas as pd
import re
import subprocess

from config import DATA_DIR, data_config, r2py_types
from .. import errors
from ..preprocess import preprocessor
from utils.records import DataFile, filter_files, parallelize_file_parser


INSTRUMENT = 'lgr_ugga'
data_config = data_config[INSTRUMENT]

PI_SITES = ('trx01', 'landfill')


@preprocessor
def get_files(site, lvl, time_range=None):

    data_dir = os.path.join(DATA_DIR, site, INSTRUMENT, lvl)

    if lvl == 'raw' and site not in PI_SITES:
        # raw lgr files are too complicated for filter_files function

        pattern = re.compile(r'f....\.txt$')

        raw_files = []
        for root, dirs, files in os.walk(data_dir):
            raw_files.extend([os.path.join(root, file) for file in files
                              if pattern.search(file)])

        raw_files.sort(key=lambda file: os.path.getmtime(file))

        return raw_files

    files = []

    date_slicer = slice(4, 14) if site in PI_SITES else slice(7)
    date_format = '%Y_%m_%d' if site in PI_SITES else '%Y_%m'

    for file in os.listdir(data_dir):
        if file.endswith('dat'):
            file_path = os.path.join(data_dir, file)
            date = pd.to_datetime(file[date_slicer], format=date_format)

            files.append(DataFile(file_path, date))

    return filter_files(files, time_range)


def count_delim(file, delim=','):
    cols = subprocess.getoutput(f'head -n 2 {file} | tail -n 1')

    return cols.count(',')


def adapt_cols(file):
    # Adapt column names depending on LGR software version.
    #  2013-2014 version has 23 columns
    #  2014+ version has 24 columns (MIU split into valve and description)

    col_names = data_config.raw['col_names'].copy()
    col_types = data_config.raw['col_types']
    col_types = {name: r2py_types[t] for name, t
                 in zip(col_names, col_types)}

    ndelim = count_delim(file)

    if ndelim == 0:
        raise errors.ParsingError('No deliminators in header!')
    elif ndelim == 23:
        # col_names config has 23 columns, temporarily add fill column
        col_names.insert(22, 'fillvalue')
        col_types['fillvalue'] = float

    return col_names, col_types


def get_serial(file):
    # Get serial number of UGGA
    def get_meta(header):
        pattern = (r'VC:(?P<VC>\w+)\s'
                   r'BD:(?P<BD>[a-zA-Z]{3}\s\d{2}\s\d{4})\s'
                   r'SN:(?P<SN>.+)')

        return re.match(pattern, header).groupdict()

    header = subprocess.getoutput(f'head -n 1 {file}')
    meta = get_meta(header)
    serial = meta['SN']

    return serial


def check_footer(file):
    # Check for incomplete final row (probably power issue)
    last_line = subprocess.getoutput(f'tail -n 1 {file}')

    return 1 if last_line.count(',') < 22 else 0


def update_cols(df):
    ncols = len(df.columns)
    if (ncols < 23) | (ncols > 24):
        raise errors.ParsingError('Too few or too many columns!')
    elif ncols == 24:
        # Now that the data has been read in with 24 columns,
        #  drop the valve (fill) column
        df.drop(columns=df.columns[22], inplace=True)

    # Reassign orignal column names (not sure this is nessecary)
    df.columns = data_config.raw['col_names'].copy()

    return df


def drop_specie_col(df, other_specie):

    return df[[col for col in df.columns if other_specie not in col.upper()]]


def _parse_raw(file, verbose):
    if verbose:
        print(f'Loading {os.sep.join(file.split(os.sep)[-2:])}')

    try:
        # Adapt columns due to differences in UGGA format
        col_names, col_types = adapt_cols(file)

        # Check for incomplete final row (probably power issue)
        footer = check_footer(file)

        # Read file assigning cols and dtypes
        df = pd.read_csv(file, header=1, names=col_names, dtype=col_types,
                         on_bad_lines='skip', na_values=['TO', ''],
                         skipinitialspace=True, skipfooter=footer,
                         engine='python')

        # Update columns now that data has been read in
        df = update_cols(df)

    except errors.ParsingError as e:
        if verbose:
            print(f'    {e}')
        return None

    # Format time
    df.Time_UTC = pd.to_datetime(df.Time_UTC.str.strip(),
                                 format='%m/%d/%Y %H:%M:%S.%f',
                                 errors='coerce')

    return df


def _parse_pi_data(file, MIU):
    names = ["Time_UTC", "ID", "Time_UTC_LGR", "CH4_ppm", "CH4_ppm_sd",
             "H2O_ppm", "H2O_ppm_sd", "CO2_ppm", "CO2_ppm_sd", "CH4d_ppm",
             "CH4d_ppm_sd", "CO2d_ppm", "CO2d_ppm_sd", "GasP_torr",
             "GasP_torr_sd", "GasT_C", "GasT_C_sd", "AmbT_C", "AmbT_C_sd",
             "RD0_us", "RD0_us_sd", "RD1_us", "RD1_us_sd", "Fit_Flag",
             "MIU_Valve", "MIU_Desc"]

    df = pd.read_csv(file, names=names, dtype=str,
                     on_bad_lines='skip', skipinitialspace=True)

    # Format datetime, set pi time as index, and filter
    df['Time_UTC'] = pd.to_datetime(df.Time_UTC.str.strip(), 'coerce',
                                    format='ISO8601')
    df['Time_UTC_LGR'] = pd.to_datetime(df.Time_UTC_LGR.str.strip(),
                                        errors='coerce',
                                        format='%d/%m/%Y %H:%M:%S.%f')

    # Convert numeric cols to float
    str_cols = ['Time_UTC', 'Time_UTC_LGR', 'MIU_Desc']
    if MIU:
        str_cols.append('ID')
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def _parse(file):
    df = pd.read_csv(file, parse_dates=['Time_UTC'])
    return df


@preprocessor
def read_obs(site, species=('CO2', 'CH4'), lvl='calibrated',
             time_range=None, num_processes=1, verbose=False, MIU=True):

    assert all(s in ['CO2', 'CH4'] for s in species)

    if verbose:
        print(f'Reading {lvl} observations collected by LGR UGGA')

    # Raw files require special parsing
    if lvl == 'raw':

        files = get_files(site, lvl='raw', time_range=time_range)

        if site in PI_SITES:
            if site.startswith('trx'):
                MIU = False

            read_files = parallelize_file_parser(_parse_pi_data,
                                                 num_processes=num_processes)
            df = pd.concat(read_files(files, MIU=MIU))

        else:
            read_files = parallelize_file_parser(_parse_raw,
                                                 num_processes=num_processes)
            df = pd.concat(read_files(files, verbose=verbose))

    else:
        # Parse files
        files = get_files(site, lvl, time_range)

        read_files = parallelize_file_parser(_parse,
                                             num_processes=num_processes)
        df = pd.concat(read_files(files))

    # Set time as index and filter to time range
    df = df.dropna(subset='Time_UTC').set_index('Time_UTC').sort_index()
    df = df.loc[time_range[0]: time_range[1]]

    # Drop the other specie if only one is specified
    if len(species) == 1:
        other_specie = 'CH4' if species[0] == 'CO2' else 'CO2'

        df = drop_specie_col(df, other_specie)

    return df
