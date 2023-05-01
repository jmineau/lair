#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:16:30 2023

@author: James Mineau (James.Mineau@utah.edu)

Module to download HRRR winds at a point

Documentation:
    https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/python_data_loading.html
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import cartopy.crs as ccrs
import dataclasses
import datetime as dt
import numcodecs as ncd
import numpy as np
import pandas as pd
import s3fs
import xarray as xr
import zarr


PROJECTION = ccrs.LambertConformal(central_longitude=262.5,
                                   central_latitude=38.5,
                                   standard_parallels=(38.5, 38.5),
                                   globe=ccrs.Globe(semimajor_axis=6371229,
                                                    semiminor_axis=6371229))


def create_s3_group_url(zarr_id, prefix=True):
    url = "s3://hrrrzarr/" if prefix else ""  # Skip when using boto3
    url += zarr_id.run_hour.strftime(
        f"{zarr_id.level_type}/%Y%m%d/%Y%m%d_%Hz_{zarr_id.model_type}.zarr/")
    url += f"{zarr_id.var_level}/{zarr_id.var_name}"
    return url


def create_s3_subgroup_url(zarr_id, prefix=True):
    url = create_s3_group_url(zarr_id, prefix)
    url += f"/{zarr_id.var_level}"
    return url


def create_s3_chunk_url(zarr_id, chunk_id, prefix=False):
    url = create_s3_subgroup_url(zarr_id, prefix)
    url += f"/{zarr_id.var_name}/{zarr_id.format_chunk_id(chunk_id)}"
    return url


def get_nearest_point(longitude, latitude, chunk_index):
    x, y = PROJECTION.transform_point(longitude, latitude, ccrs.PlateCarree())
    return chunk_index.sel(x=x, y=y, method="nearest")


def retrieve_object(s3, s3_url):
    obj = s3.Object('hrrrzarr', s3_url)
    return obj.get()['Body'].read()


def generate_zarr_ids(times, level_type, variables: [(str, str)], model_type):
    '''
    Generate ZarrId instance for multiple times and variables

    Parameters
    ----------
    times : [dt.datetime]
        list of datetime objects with hour specified [UTC]
    level_type : str
        veritical level
    variables : [(str, str)]
        list of tuples: (var_name, var_level)
    model_type : str
        'anl' | 'fcst'

    Returns
    -------
    ids : [ZarrId]
        dictionary of zarr ids with var as key

    '''
    ids = {}

    for var in variables:
        ids[var] = []
        for time in times:
            ID = ZarrId(run_hour=time,
                        level_type=level_type,
                        var_name=var[0],
                        var_level=var[1],
                        model_type=model_type)
            ids[var].append(ID)

    return ids


def decompress_chunk(zarr_id, compressed_data):
    buffer = ncd.blosc.decompress(compressed_data)

    dtype = "<f2"
    if zarr_id.var_level == "surface" and zarr_id.var_name == "PRES":
        dtype = "<f4"

    chunk = np.frombuffer(buffer, dtype=dtype)

    if zarr_id.model_type == "anl":
        arr = np.reshape(chunk, (150, 150))
    else:
        entry_size = 22500
        arr = np.reshape(chunk, (len(chunk)//entry_size, 150, 150))

    return arr


def get_value(s3, zarr_id, chunk_id, nearest_point):
    compressed_data = retrieve_object(s3, create_s3_chunk_url(zarr_id, chunk_id))
    chunk_data = decompress_chunk(zarr_id, compressed_data)
    if zarr_id.model_type == "fcst":
        return chunk_data[:, nearest_point.in_chunk_y.values, nearest_point.in_chunk_x.values]
    else:
        return chunk_data[nearest_point.in_chunk_y.values, nearest_point.in_chunk_x.values]


@dataclasses.dataclass
class ZarrId:
    run_hour: dt.datetime
    level_type: str
    var_level: str
    var_name: str
    model_type: str

    def format_chunk_id(self, chunk_id):
        if self.model_type == "fcst":
            # Extra id part since forecasts have an additional (time) dimension
            return "0." + str(chunk_id)
        else:
            return chunk_id


@dataclasses.dataclass
class Winds:
    lat: float
    lon: float
    times: [dt.datetime]

    def __post_init__(self):
        fs = s3fs.S3FileSystem(anon=True)
        s3 = boto3.resource(service_name='s3', region_name='us-west-1',
                            config=Config(signature_version=UNSIGNED))

        # Chunk Index
        chunk_index_url = "s3://hrrrzarr/grid/HRRR_chunk_index.zarr"
        self.chunk_index = xr.open_zarr(s3fs.S3Map(chunk_index_url, s3=fs))

        # Nearest Point
        self.nearest_point = get_nearest_point(self.lon, self.lat,
                                               self.chunk_index)
        self.chunk_id = self.nearest_point.chunk_id.values

        # Create zarr ids
        variables = [('UGRD', '10m_above_ground'),
                     ('VGRD', '10m_above_ground'),
                     ('WIND_max_fcst', '10m_above_ground')]
        self.zarr_ids = generate_zarr_ids(times=self.times,
                                          level_type='surface',
                                          variables=variables,
                                          model_type='anl')

        # Download variables
        var_data = {}
        for (var_name, _), ids in self.zarr_ids.items():
            var_data[var_name] = [get_value(s3, zarr_id, self.chunk_id,
                                            self.nearest_point)
                                  for zarr_id in ids]

        # Create dataframe
        self.data = pd.DataFrame(var_data, index=self.times)
