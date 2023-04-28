#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 09:40:10 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for LGR UGGA instrument
"""

from .. import uucon
from .. import pipeline as pipe
import os
import re
import subprocess
import pandas as pd


# %% RAW

def get_raw_files(site):
    # raw_dir = f'~/wkspace/pipeline/local/data/{site}/lgr_ugga/raw'
    # raw_dir = ('/uufs/chpc.utah.edu/common/home/u6036966/wkspace/pipeline/'
    #            f'local/data/{site}/lgr_ugga/raw')
    raw_dir = os.path.join(uucon.UATAQ_DIR, 'data', site, 'lgr_ugga', 'raw')

    pattern = re.compile(r'f....\.txt$')

    raw_files = []
    for root, dirs, files in os.walk(raw_dir):
        raw_files.extend([os.path.join(root, file) for file in files
                          if pattern.search(file)])

    raw_files.sort(key=lambda file: os.path.getmtime(file))

    return raw_files


def count_delim(file, delim=','):
    cols = subprocess.getoutput(f'head -n 2 {file} | tail -n 1')

    return cols.count(',')


def adapt_cols(file):
    # Adapt column names depending on LGR software version.
    #  2013-2014 version has 23 columns
    #  2014+ version has 24 columns (MIU split into valve and description)

    r2py_types = {'c': str,
                  'd': float}

    col_names = uucon.data_config.lgr_ugga.raw['col_names'].copy()
    col_types = uucon.data_config.lgr_ugga.raw['col_types']
    col_types = {name: r2py_types[t] for name, t
                 in zip(col_names, col_types)}

    ndelim = count_delim(file)

    if ndelim == 0:
        raise pipe.ParsingError('No deliminators in header!')
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
        raise pipe.ParsingError('Too few or too many columns!')
    elif ncols == 24:
        # Now that the data has been read in with 24 columns,
        #  drop the valve (fill) column
        df.drop(columns=df.columns[22], inplace=True)

    # Reassign orignal column names (not sure this is nessecary)
    df.columns = uucon.data_config.lgr_ugga.raw['col_names'].copy()

    return df


def RAW(site):
    def parse(file):
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
        df = update_cols(df).assign(filename=file)

        return df

    # Get files
    files = get_raw_files(site)

    dfs = []
    bad_files = []

    # Parse dataframes
    for file in files:
        print(f'Loading {os.sep.join(file.split(os.sep)[-2:])}')
        try:
            df = parse(file)
            dfs.append(df)
        except pipe.ParsingError as e:
            print(f'    {e}')
            bad_files.append((file, str(e)))
            continue

    data = pd.concat(dfs)  # Merge dataframes

    # Format Time_UTC column
    data.Time_UTC = pd.to_datetime(data.Time_UTC.str.strip(),
                                   format='%m/%d/%Y %H:%M:%S.%f',
                                   errors='coerce')

    # Drop rows with invalid time and sort on time
    data.dropna(subset='Time_UTC', inplace=True)
    data.sort_values('Time_UTC', inplace=True)

    return data, bad_files


# %% PROCESSED

def read_processed(site, lvl):
    data_path = os.path.join(uucon.UATAQ_DIR, 'data', site, 'lgr_ugga', lvl)

    files = [os.path.join(data_path, file) for file in os.listdir(data_path)]

    df = pd.concat([pd.read_csv(file, parse_dates=['Time_UTC'])
                    for file in files]).set_index('Time_UTC').sort_index()

    return df


def QAQC(site):
    return read_processed(site, 'qaqc')


def CALIBRATED(site):
    return read_processed(site, 'calibrated')
