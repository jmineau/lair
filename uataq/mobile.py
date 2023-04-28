#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:41:52 2023

@author: James Mineau (James.Mineau@uta.edu)

Module to interact with mobile data including TRAX, NerdMobile, and eventually
the eBUS
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass

from utils.grid import dms2dd
from utils.time import UTC2MTN


class Bus:
    pass


class NerdMobile:
    def __init__(self, dates=['2022-10-28']):
        # Set directories
        user_dir = '/uufs/chpc.utah.edu/common/home/u6036966'
        self.project_dir = os.path.join(user_dir, 'Projects/MethaneAIR')
        self.data_dir = os.path.join(self.project_dir, 'data/nerdmobile')

        # List of datetime dates to subset by
        self.dates = [dt.datetime.strptime(date, '%Y-%m-%d').date()
                      for date in dates]

    def readV2(self):  # Read NerdMobileV2
        # file templates
        raw_template = 'methaneAir_field_campaign_{}.dat'
        geojson_template = 'nerdmobile_{}.geojson'

        def read_raw(date, raw_path, geojson_path):  # Read raw V2 data

            # Read Data
            df = pd.read_table(raw_path, sep=',', header=1, skiprows=[2, 3])

            # Units
            # unit_df = pd.read_table(raw_path, sep=',', header=1, nrows=1)
            # units = {col: unit_df[col].iloc[0] for col in unit_df.columns}

            # Set measurements to np.nan when all are 0
            LGR_cols = ['LGR_CH4', 'LGR_CO2', 'LGR_H2O',
                        'LGR_CH4_dry', 'LGR_CO2_dry']
            valid = ((df.LGR_CH4 == 0) & (df.LGR_CO2 == 0) & (df.LGR_H2O == 0)
                     & (df.LGR_CH4_dry == 0) & (df.LGR_CO2_dry == 0))
            df.loc[valid, LGR_cols] = np.nan

            # Convert DMS to DD
            df['lat'] = df.apply(lambda row: dms2dd(row.Latitude_A,
                                                    row.Latitude_B), axis=1)
            df['lon'] = df.apply(lambda row: dms2dd(row.Longitude_A,
                                                    row.Longitude_B), axis=1)

            # Correct TIMESTAMP (2003-03-14 -> 2022-10-28)
            df.TIMESTAMP = pd.to_datetime(df.TIMESTAMP)

            def fix_timestamp(date):
                return date.replace(year=2022, month=10, day=28)
            df.TIMESTAMP = df.TIMESTAMP.apply(fix_timestamp)
            df = df.set_index('TIMESTAMP')  # Set TIMESTAMP as inde

            df['Local_Time'] = UTC2MTN(df.index)

            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(df, crs='EPSG:4326',
                                   geometry=gpd.points_from_xy(df.lon, df.lat))

            gdf.to_file(geojson_path)  # save processed data to geojson

            return gdf

        gdfs = []  # list of geodataframes for each date

        for date in self.dates:
            # Get file paths from templates
            raw_path = raw_template.format(date.strftime('%m-%d-%Y'))
            raw_path = os.path.join(self.data_dir, raw_path)

            geojson_path = geojson_template.format(date.strftime('%Y_%m_%d'))
            geojson_path = os.path.join(self.data_dir, geojson_path)

            # Check is raw data has been processed
            if not os.path.exists(geojson_path):  # if not, process it
                gdf = read_raw(date, raw_path, geojson_path)

            else:  # read processed geojson
                gdf = gpd.read_file(geojson_path)

            gdfs.append(gdf)

        return pd.concat(gdfs)  # merge dates


@dataclass
class TRAX:
    vehicle: str
    months: list
    species: list

    # TODO add process and read to pipeline

    # def process(self):
    #     # Process TRAX calibrated data as transects
    #     process()

    # def read(self):
    #     # Read processed TRAX transects
    #     read()
