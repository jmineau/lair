"""
HRRR winds at a point.

Documentation: https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/python_data_loading.html
"""

# Optional dependencies
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
except ImportError:
    print("boto3 not installed. Please install boto3 to use the hrrr module.")
try:
    import s3fs
except ImportError:
    print("s3fs not installed. Please install s3fs to use the hrrr module.")
try:
    import numcodecs as ncd
    import zarr
except ImportError:
    print("zarr not installed. Please install zarr to use the hrrr module.")

import cartopy.crs as ccrs
import dataclasses
import datetime as dt
import numpy as np
import pandas as pd
from typing import Literal, Tuple
import xarray as xr

from lair.air import wind_direction, rotate_winds


#: HRRR Projection
PROJECTION: ccrs.CRS = ccrs.LambertConformal(central_longitude=262.5,
                                             central_latitude=38.5,
                                             standard_parallels=(38.5, 38.5),
                                             globe=ccrs.Globe(semimajor_axis=6371229,
                                                              semiminor_axis=6371229))


@dataclasses.dataclass
class ZarrId:
    """
    Class to store information about a HRRR zarr file.
    
    Attributes
    ----------
    run_hour : dt.datetime
        run hour of the model
    level_type : 'sfc' | 'prs'
        vertical level
    var_level : str
        variable level
    var_name : str
        variable name
    model_type : 'anl' | 'fcst'
        analysis or forecast

    Methods
    -------
    format_chunk_id(chunk_id)
        Format chunk_id for zarr url.
    """
    run_hour: dt.datetime
    level_type: Literal['sfc', 'prs']
    var_level: str
    var_name: str
    model_type: Literal['anl', 'fcst']

    def format_chunk_id(self, chunk_id):
        """
        Format chunk_id for zarr url.

        Parameters
        ----------
        chunk_id : int
            chunk_id

        Returns
        -------
        str
            formatted chunk_id
        """
        if self.model_type == "fcst":
            # Extra id part since forecasts have an additional (time) dimension
            return "0." + str(chunk_id)
        else:
            return chunk_id


@dataclasses.dataclass
class Winds:
    """
    Class to download HRRR winds at a point.

    Attributes
    ----------
    lat : float
        latitude of point
    lon : float
        longitude of point
    times : list[dt.datetime]
        list of times to download
    nearest_point : xr.Dataset
        nearest point to lat, lon
    chunk_id : float
        chunk id of nearest point
    zarr_ids : dict[str, list[ZarrId]]
        dictionary of zarr ids with var as key
    data : pd.DataFrame
        wind data
    """
    lat: float
    lon: float
    times: list[dt.datetime]

    def __post_init__(self):
        # Connect to AWS servers
        fs = s3fs.S3FileSystem(anon=True)
        # Don't recreate this resource in a loop! That caused a 3-4x slowdown for me.
        # Not sure who the above is, but should be listened to.
        s3 = boto3.resource(service_name='s3', region_name='us-west-1',
                            config=Config(signature_version=UNSIGNED))

        # Chunk Index
        chunk_index_url = "s3://hrrrzarr/grid/HRRR_chunk_index.zarr"
        self.chunk_index = xr.open_zarr(s3fs.S3Map(chunk_index_url, s3=fs))

        # Nearest Point to chunk_id
        self.nearest_point = get_nearest_point(self.lon, self.lat,
                                               self.chunk_index)
        self.chunk_id = self.nearest_point.chunk_id.values

        # Generate zarr ids
        variables = [('UGRD', '10m_above_ground'),
                     ('VGRD', '10m_above_ground'),
                     ('WIND_max_fcst', '10m_above_ground')]
        self.zarr_ids = generate_zarr_ids(times=self.times,
                                          level_type='sfc',
                                          variables=variables,
                                          model_type='anl')

        # Download variables
        var_data = {}
        for (var_name, _), ids in self.zarr_ids.items():
            var_data[var_name] = [get_value(s3, zarr_id, self.chunk_id,
                                            self.nearest_point)
                                  for zarr_id in ids]

        u_earth, v_earth = rotate_winds(var_data['UGRD'], var_data['VGRD'],
                                        self.lon)

        angle = wind_direction(u_earth, v_earth)

        self.data = pd.DataFrame({'u': u_earth,
                                  'v': v_earth,
                                  'ws': var_data['WIND_max_fcst'],
                                  'wd': angle}, index=self.times)


def create_s3_group_url(zarr_id: ZarrId, prefix: bool=True) -> str:
    """
    Create the s3 group url for a zarr id.
    
     - Start of the zarr array data format
     - Includes metadata such as the grid

    Parameters
    ----------
    zarr_id : ZarrId
        ZarrId instance
    prefix : bool
        include prefix

    Returns
    -------
    str
        s3 group url
    """
    url = "s3://hrrrzarr/" if prefix else ""  # Skip when using boto3
    url += zarr_id.run_hour.strftime(
        f"{zarr_id.level_type}/%Y%m%d/%Y%m%d_%Hz_{zarr_id.model_type}.zarr/")
    url += f"{zarr_id.var_level}/{zarr_id.var_name}"
    return url


def create_s3_subgroup_url(zarr_id: ZarrId, prefix: bool=True) -> str:
    """
    Create the s3 subgroup url for a zarr id.

     - Where the actual data variable is stored
     - While subgroups are part of the zarr spec, there's not necessarily a
       good reason for the data in this case to have this extra level of nesting

    Parameters
    ----------
    zarr_id : ZarrId
        ZarrId instance
    prefix : bool
        include prefix

    Returns
    -------
    str
        s3 subgroup url
    """
    url = create_s3_group_url(zarr_id, prefix)
    url += f"/{zarr_id.var_level}"
    return url


def create_s3_chunk_url(zarr_id: ZarrId, chunk_id, prefix: bool=False) -> str:
    """
    Create the s3 chunk url for a zarr id.

     - Contains just the data for the chunk (no metadata)
     - Current APIs (zarr, xarray) don't support reading zarr data by chunks,
       so we have to write relatively low-level code to load data on this level

    Parameters
    ----------
    zarr_id : ZarrId
        ZarrId instance
    chunk_id : int  # TODO check this
        chunk id
    prefix : bool
        include prefix

    Returns
    -------
    str
        s3 chunk url
    """
    url = create_s3_subgroup_url(zarr_id, prefix)
    url += f"/{zarr_id.var_name}/{zarr_id.format_chunk_id(chunk_id)}"
    return url


def get_nearest_point(longitude: float, latitude: float, chunk_index: xr.Dataset
                      ) -> xr.Dataset:
    """
    Get the nearest point to a given latitude and longitude.

    Parameters
    ----------
    longitude : float
        longitude of point
    latitude : float
        latitude of point
    chunk_index : xr.Dataset
        chunk index dataset

    Returns
    -------
    xr.Dataset
        nearest point
    """
    x, y = PROJECTION.transform_point(longitude, latitude, ccrs.PlateCarree())
    return chunk_index.sel(x=x, y=y, method="nearest")


def retrieve_object(s3, s3_url: str):
    """
    Retrieve object from s3.

    Parameters
    ----------
    s3 : boto3.resource
        s3 resource
    s3_url : str
        s3 url

    Returns
    -------
    bytes
        compressed object data
    """
    obj = s3.Object('hrrrzarr', s3_url)
    return obj.get()['Body'].read()


def generate_zarr_ids(times: list[dt.datetime], level_type: str, variables: list[Tuple[str, str]],
                      model_type: Literal['anl', 'fcst']):
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


def decompress_chunk(zarr_id: ZarrId, compressed_data: bytes):
    """
    Decompress s3 chunk data.

    Parameters
    ----------
    zarr_id : ZarrId
        ZarrId instance
    compressed_data : bytes
        compressed data

    Returns
    -------
    np.array
        decompressed chunk
    """
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


def get_value(s3, zarr_id: ZarrId, chunk_id, nearest_point: xr.Dataset):
    """
    Get an array of values from am s3 zarr chunk.

    Parameters
    ----------
    s3 : boto3.resource
        s3 resource
    zarr_id : ZarrId
        ZarrId instance
    chunk_id : float
        chunk id
    nearest_point : xr.Dataset
        nearest point

    Returns
    -------
    np.array
        array of values
    """
    compressed_data = retrieve_object(s3, create_s3_chunk_url(zarr_id, chunk_id))
    chunk_data = decompress_chunk(zarr_id, compressed_data)
    if zarr_id.model_type == "fcst":
        return chunk_data[:, nearest_point.in_chunk_y.values, nearest_point.in_chunk_x.values]
    else:
        return chunk_data[nearest_point.in_chunk_y.values, nearest_point.in_chunk_x.values]
