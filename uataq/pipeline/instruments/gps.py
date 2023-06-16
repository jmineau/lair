#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 15:11:40 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for TRAX gps
"""

import os
import pandas as pd

from config import DATA_DIR, instrument_config
from . import cr1000
from ..preprocess import preprocessor
from utils.grid import dms2dd
from utils.records import DataFile, filter_files


INSTRUMENT = 'gps'

NMEA_dict = {
    'GPGGA': {
        'usecols': [0, 1, 2, 3, 5, 7, 8, 9, 10],
        'names': ['Time_UTC', 'NMEA', 'GPS_Time_UTC', 'Lati_dm',
                  'Long_dm', 'Fix_Quality', 'NSat',
                  'Location_Uncertainty_m', 'Altitude_m']},
    'GPRMC': {
        'usecols': [0, 1, 2, 3, 4, 6, 8, 9, 10],
        'names': ['Time_UTC', 'NMEA', 'GPS_Time_UTC', 'Status', 'Lati_dm',
                  'Long_dm', 'Speed_kt', 'True_Course', 'GPS_Date']}}

AIR_TREND_USECOLS = [0, 1, 2, 4, 6, 7, 8]
AIR_TREND_COLS = ['Time_UTC', 'GPS_Time_UTC', 'Lati_dm', 'Long_dm',
                  'Fix_Quality', 'NSat', 'Altitude_m']

str_cols = ['Time_UTC', 'NMEA', 'Status', 'GPS_Time_UTC', 'GPS_Date',
            'Time_UTC_GPS']


@preprocessor
def get_files(site, lvl, time_range=None, use_lin_group=False):

    installation_date = instrument_config[site]['GPS'][0]['installation_date']

    files = []

    if use_lin_group:
        data_dir = os.path.join(DATA_DIR, site, INSTRUMENT, lvl)

        if lvl == 'raw':
            date_format = '%Y_%m_%d'
            exts = ['csv', 'dat']

            if site == 'trx01':
                date_slicer = slice(4, 14)
            elif site == 'trx02':
                date_slicer = slice(10)
        else:
            date_format = '%Y_%m'
            date_slicer = slice(7)
            exts = ['dat']

        for file in os.listdir(data_dir):
            if file[-3:] in exts:
                file_path = os.path.join(data_dir, file)
                date = pd.to_datetime(file[date_slicer].replace('-', '_'),
                                      format=date_format)

                if date < pd.to_datetime(installation_date):
                    continue

                files.append(DataFile(file_path, date))

        return filter_files(files, time_range)

    return cr1000.get_files(site, time_range=time_range)


def format_lat(lat):
    lat = str(lat)
    d, m = lat[:2], lat[2:]
    return dms2dd(d, m)


def format_lon(lon):
    lon = str(lon)
    d, m = lon[:3], lon[3:]
    return -1 * dms2dd(d, m)


def format_gps_time(time: str):
    split = time.split('.')

    if len(split) == 1:
        return split[0].zfill(6)

    integer, deci = split
    return integer.zfill(6) + '.' + deci


@preprocessor
def read_raw(site, time_range=None, NMEA='GPGGA'):

    files = get_files(site, 'raw', time_range, True)

    dfs = []
    for file in files:

        # Parse lin-group data
        if site == 'trx01':
            # TRX01 records both GPGGA & GPRMC messages
            assert NMEA in ['GPGGA', 'GPRMC']

            usecols = NMEA_dict[NMEA]['usecols']
            names = NMEA_dict[NMEA]['names']

            df = pd.read_csv(file, header=None, usecols=usecols, dtype=str,
                             names=names, parse_dates=['Time_UTC'],
                             index_col='Time_UTC', on_bad_lines='skip')

            # Filter rows to those matching NMEA code
            df = df[df.NMEA == f'${NMEA}']
            df.drop(columns=['NMEA'], inplace=True)

            if NMEA == 'GPRMC':
                # Combine GPS date and time into datetime
                gps_datetime = df.GPS_Date + df.GPS_Time_UTC
                df['Time_UTC_GPS'] = pd.to_datetime(gps_datetime,
                                                    errors='coerce',
                                                    format='%d%m%y%H%M%S.%f')
                df.drop(columns=['GPS_Time_UTC', 'GPS_Date'], inplace=True)

        elif site == 'trx02':
            # TRX02 uses air-trend code and only saves GPGGA messages
            df = pd.read_csv(file, header=0, names=AIR_TREND_COLS, dtype=str,
                             usecols=AIR_TREND_USECOLS, on_bad_lines='skip',
                             parse_dates=['Time_UTC'], index_col='Time_UTC')

        for col in df.columns:
            if col not in str_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert coordinates to decimal degrees
        df.dropna(subset=['Lati_dm', 'Long_dm'], inplace=True)
        df['Lati_deg'] = df['Lati_dm'].apply(format_lat)
        df['Long_deg'] = df['Long_dm'].apply(format_lon)
        df.drop(columns=['Lati_dm', 'Long_dm'], inplace=True)

        dfs.append(df)

    df = pd.concat(dfs)

    return df


@preprocessor
def read_obs(site, specie='GPS', lvl='raw', time_range=None,
             use_lin_group=False, NMEA='GPGGA'):
    assert specie == 'GPS'

    if not use_lin_group:
        # Parse horel-group data
        df = cr1000.read_obs(site, time_range=time_range)

        if lvl != 'raw':
            # Create QAQC_Flag column
            df['QAQC_Flag'] = 0

    else:
        if lvl == 'raw':
            df = read_raw(site, time_range, NMEA)
        else:
            files = get_files(site, lvl, time_range, use_lin_group)

            dfs = []
            for file in files:
                df = pd.read_csv(file, dtype=str, parse_dates=['Time_UTC'],
                                 index_col='Time_UTC')

                for col in df.columns[1:]:
                    df[col] = pd.to_numeric(df[col], 'coerce')

                df.GPS_Time_UTC = df.GPS_Time_UTC.apply(format_gps_time)

                dfs.append(df)

            df = pd.concat(dfs)

    return df
