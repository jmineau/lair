"""
HRRR winds at a point.

:class:`Winds` downloads the 10 m winds of the HRRR analyses at the grid point
nearest a latitude and longitude, from the University of Utah's HRRR-Zarr
archive on Amazon S3, and rotates them to earth-relative u and v. The other
functions are the steps it takes: build the S3 keys of a model run
(:class:`ZarrId`), find the grid point, read and decompress its chunk.
:doc:`/examples/hrrr_winds` shows it at one point.

The archive's documentation:
https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/python_data_loading.html
"""

import dataclasses
import datetime as dt
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from lair._optional import import_optional_dependency
from lair.air import rotate_winds, wind_direction
from lair.clock import convert_timezones

# Optional dependencies
boto3 = import_optional_dependency("boto3")
cartopy = import_optional_dependency("cartopy")
s3fs = import_optional_dependency("s3fs")
zarr = import_optional_dependency("zarr")  # needed by xr.open_zarr below
ncd = import_optional_dependency("numcodecs")

import cartopy.crs as ccrs  # noqa: E402
from botocore import UNSIGNED  # noqa: E402
from botocore.config import Config  # noqa: E402

#: HRRR's Lambert conformal projection, a cartopy CRS. :func:`get_nearest_point`
#: uses it to find a longitude and latitude on the HRRR grid.
PROJECTION: ccrs.CRS = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)


@dataclasses.dataclass
class ZarrId:
    """
    Class to store information about a HRRR zarr file.

    Each parameter is also an attribute of the same name.

    Parameters
    ----------
    run_hour : dt.datetime
        run hour of the model. A naive time follows :ref:`the UTC rule
        <utc-rule>`; a tz-aware time is converted to UTC when the S3 key is
        built.
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

    See Also
    --------
    generate_zarr_ids : Makes one for each time and variable.
    create_s3_group_url : The S3 address of its group.
    create_s3_chunk_url : The S3 address of one chunk.

    Examples
    --------
    The analysis of 12 UTC on 2024-01-15, the 10 m u wind:

    >>> import datetime as dt
    >>> from lair.hrrr import ZarrId, create_s3_group_url
    >>> zarr_id = ZarrId(
    ...     run_hour=dt.datetime(2024, 1, 15, 12),
    ...     level_type="sfc",
    ...     var_level="10m_above_ground",
    ...     var_name="UGRD",
    ...     model_type="anl",
    ... )
    >>> create_s3_group_url(zarr_id)
    's3://hrrrzarr/sfc/20240115/20240115_12z_anl.zarr/10m_above_ground/UGRD'
    """

    run_hour: dt.datetime
    level_type: Literal["sfc", "prs"]
    var_level: str
    var_name: str
    model_type: Literal["anl", "fcst"]

    def format_chunk_id(self, chunk_id):
        """
        Format chunk_id for zarr url.

        Parameters
        ----------
        chunk_id : str
            Chunk id, for example ``"4.5"``.

        Returns
        -------
        str
            formatted chunk_id

        See Also
        --------
        lair.hrrr.create_s3_chunk_url : Uses it to name a chunk.

        Examples
        --------
        A forecast has a leading time index; an analysis does not:

        >>> import datetime as dt
        >>> from lair.hrrr import ZarrId
        >>> kwargs = dict(
        ...     run_hour=dt.datetime(2024, 1, 15, 12),
        ...     level_type="sfc",
        ...     var_level="10m_above_ground",
        ...     var_name="UGRD",
        ... )
        >>> ZarrId(model_type="anl", **kwargs).format_chunk_id("4.5")
        '4.5'
        >>> ZarrId(model_type="fcst", **kwargs).format_chunk_id("4.5")
        '0.4.5'
        """
        if self.model_type == "fcst":
            # Extra id part since forecasts have an additional (time) dimension
            return "0." + str(chunk_id)
        else:
            return chunk_id


def winds_frame(u_grid, v_grid, lon: float, times) -> pd.DataFrame:
    """
    Earth-relative 10 m winds at a point from HRRR grid-relative components.

    Parameters
    ----------
    u_grid, v_grid : array-like
        Grid-relative u and v components (m/s), one per time.
    lon : float
        Longitude of the point.
    times : list[dt.datetime]
        Valid times, used as the index.

    Returns
    -------
    pd.DataFrame
        Columns ``u``, ``v`` (earth-relative, m/s), ``ws`` (speed from u and v,
        m/s) and ``wd`` (direction, degrees).

    See Also
    --------
    Winds : Downloads the components and calls this.
    lair.air.rotate_winds : Turns grid-relative components to earth-relative.
    lair.air.wind_direction : The direction of a u and v.

    Examples
    --------
    At 97.5 W, the middle of the HRRR domain, the grid points north, so the
    components do not change. A wind blowing toward the south (v = -4) comes
    from the north (0 degrees); one blowing toward the east (u = 3), from the
    west (270):

    >>> import datetime as dt
    >>> from lair.hrrr import winds_frame
    >>> times = [dt.datetime(2024, 1, 15, 12), dt.datetime(2024, 1, 15, 13)]
    >>> winds_frame([0.0, 3.0], [-4.0, 0.0], lon=-97.5, times=times).round(2)
                           u    v   ws     wd
    2024-01-15 12:00:00  0.0 -4.0  4.0    0.0
    2024-01-15 13:00:00  3.0  0.0  3.0  270.0
    """
    u, v = rotate_winds(
        np.asarray(u_grid, dtype=float), np.asarray(v_grid, dtype=float), lon
    )
    return pd.DataFrame(
        {"u": u, "v": v, "ws": np.hypot(u, v), "wd": wind_direction(u, v)}, index=times
    )


@dataclasses.dataclass
class Winds:
    """
    Class to download HRRR winds at a point.

    Making one downloads the winds: the 10 m u and v analyses of each hour,
    at the HRRR grid point nearest to ``lat`` and ``lon``, from the public
    HRRR-Zarr archive on Amazon S3. :doc:`/examples/hrrr_winds` shows it.
    Each parameter is also an attribute of the same name.

    Parameters
    ----------
    lat : float
        latitude of point
    lon : float
        longitude of point
    times : list[dt.datetime]
        Hours of the analyses to download. A naive time follows :ref:`the UTC
        rule <utc-rule>`. A tz-aware time is converted to UTC to choose the
        analysis, so 05:00 in ``America/Denver`` in January reads the 12 UTC
        analysis.

    Attributes
    ----------
    nearest_point : xr.Dataset
        nearest point to lat, lon
    chunk_id : numpy.ndarray
        chunk id of nearest point (a 0-d array holding a string such as
        ``"4.5"``)
    zarr_ids : dict[str, list[ZarrId]]
        dictionary of zarr ids with var as key
    data : pd.DataFrame
        10 m winds: ``u``, ``v`` (earth-relative), ``ws`` (speed from u and v)
        and ``wd`` (see :func:`winds_frame`). The index is ``times`` as given,
        so tz-aware times keep their time zone.

    See Also
    --------
    winds_frame : Builds ``data`` from the downloaded components.
    ZarrId : Names one hour of one variable in the archive.
    lair.clock.UTC2MTN : Moves the index to Mountain time.

    Examples
    --------
    Needs the internet (the ``requests``, ``formats`` and ``geo`` extras):

    >>> import datetime as dt
    >>> from lair.hrrr import Winds
    >>> times = [dt.datetime(2024, 1, 15, 12), dt.datetime(2024, 1, 15, 13)]  # UTC
    >>> w = Winds(lat=40.766, lon=-111.848, times=times)  # doctest: +SKIP
    >>> w.data  # doctest: +SKIP
    """

    lat: float
    lon: float
    times: list[dt.datetime]

    def __post_init__(self):
        # Connect to AWS servers
        fs = s3fs.S3FileSystem(anon=True)
        # Don't recreate this resource in a loop! That caused a 3-4x slowdown for me.
        # Not sure who the above is, but should be listened to.
        s3 = boto3.resource(
            service_name="s3",
            region_name="us-west-1",
            config=Config(signature_version=UNSIGNED),
        )

        # Chunk Index
        chunk_index_url = "s3://hrrrzarr/grid/HRRR_chunk_index.zarr"
        self.chunk_index = xr.open_zarr(s3fs.S3Map(chunk_index_url, s3=fs))

        # Nearest Point to chunk_id
        self.nearest_point = get_nearest_point(self.lon, self.lat, self.chunk_index)
        self.chunk_id = self.nearest_point.chunk_id.values

        # Generate zarr ids
        variables = [("UGRD", "10m_above_ground"), ("VGRD", "10m_above_ground")]
        self.zarr_ids = generate_zarr_ids(
            times=self.times, level_type="sfc", variables=variables, model_type="anl"
        )

        # Download variables
        var_data = {}
        for (var_name, _), ids in self.zarr_ids.items():
            var_data[var_name] = [
                get_value(s3, zarr_id, self.chunk_id, self.nearest_point)
                for zarr_id in ids
            ]

        self.data = winds_frame(
            var_data["UGRD"], var_data["VGRD"], self.lon, self.times
        )


def create_s3_group_url(zarr_id: ZarrId, prefix: bool = True) -> str:
    """
    Create the s3 group url for a zarr id.

     - Start of the zarr array data format
     - Includes metadata such as the grid

    The archive is keyed by the UTC run hour: a naive ``run_hour`` follows
    :ref:`the UTC rule <utc-rule>`, and a tz-aware one is converted to UTC.

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

    See Also
    --------
    create_s3_subgroup_url : The same, one level down.
    create_s3_chunk_url : The same, down to one chunk.
    ZarrId : What it takes.

    Examples
    --------
    The archive is keyed by the UTC hour, so 05:00 in Utah in January is the
    12 UTC run:

    >>> import datetime as dt
    >>> from zoneinfo import ZoneInfo
    >>> from lair.hrrr import ZarrId, create_s3_group_url
    >>> local = dt.datetime(2024, 1, 15, 5, tzinfo=ZoneInfo("America/Denver"))
    >>> zarr_id = ZarrId(local, "sfc", "10m_above_ground", "UGRD", "anl")
    >>> create_s3_group_url(zarr_id, prefix=False)
    'sfc/20240115/20240115_12z_anl.zarr/10m_above_ground/UGRD'
    """
    url = "s3://hrrrzarr/" if prefix else ""  # Skip when using boto3
    # strftime formats a tz-aware time in its own zone, so convert it first
    (run_hour,) = convert_timezones(
        [zarr_id.run_hour], totz="UTC", fromtz="UTC", localize=True
    )
    url += run_hour.strftime(
        f"{zarr_id.level_type}/%Y%m%d/%Y%m%d_%Hz_{zarr_id.model_type}.zarr/"
    )
    url += f"{zarr_id.var_level}/{zarr_id.var_name}"
    return url


def create_s3_subgroup_url(zarr_id: ZarrId, prefix: bool = True) -> str:
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

    See Also
    --------
    create_s3_group_url : The level above.
    create_s3_chunk_url : The level below.

    Examples
    --------
    >>> import datetime as dt
    >>> from lair.hrrr import ZarrId, create_s3_subgroup_url
    >>> zarr_id = ZarrId(
    ...     dt.datetime(2024, 1, 15, 12), "sfc", "10m_above_ground", "UGRD", "anl"
    ... )
    >>> create_s3_subgroup_url(zarr_id, prefix=False)
    'sfc/20240115/20240115_12z_anl.zarr/10m_above_ground/UGRD/10m_above_ground'
    """
    url = create_s3_group_url(zarr_id, prefix)
    url += f"/{zarr_id.var_level}"
    return url


def create_s3_chunk_url(zarr_id: ZarrId, chunk_id, prefix: bool = False) -> str:
    """
    Create the s3 chunk url for a zarr id.

     - Contains just the data for the chunk (no metadata)
     - Current APIs (zarr, xarray) don't support reading zarr data by chunks,
       so we have to write relatively low-level code to load data on this level

    Parameters
    ----------
    zarr_id : ZarrId
        ZarrId instance
    chunk_id : str
        Name of the chunk's object in the zarr group, for example ``"4.5"``
        (forecasts get a leading time index, see ``ZarrId.format_chunk_id``).
        It is the ``chunk_id`` of a point in the HRRR chunk index (see
        ``get_nearest_point``).
    prefix : bool
        include prefix

    Returns
    -------
    str
        s3 chunk url

    See Also
    --------
    create_s3_subgroup_url : The level above.
    get_value : Reads the chunk this names.
    get_nearest_point : Gives the ``chunk_id`` of a point.

    Examples
    --------
    >>> import datetime as dt
    >>> from lair.hrrr import ZarrId, create_s3_chunk_url
    >>> zarr_id = ZarrId(
    ...     dt.datetime(2024, 1, 15, 12), "sfc", "10m_above_ground", "UGRD", "anl"
    ... )
    >>> create_s3_chunk_url(zarr_id, "4.5")
    'sfc/20240115/20240115_12z_anl.zarr/10m_above_ground/UGRD/10m_above_ground/UGRD/4.5'
    """
    url = create_s3_subgroup_url(zarr_id, prefix)
    url += f"/{zarr_id.var_name}/{zarr_id.format_chunk_id(chunk_id)}"
    return url


def get_nearest_point(
    longitude: float, latitude: float, chunk_index: xr.Dataset
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

    See Also
    --------
    create_s3_chunk_url : Takes the ``chunk_id`` of the point it returns.
    Winds : Calls it for the point you give.
    PROJECTION : The projection it uses.

    Examples
    --------
    The chunk index is a Zarr dataset in the HRRR archive (needs the internet):

    >>> import s3fs
    >>> import xarray as xr
    >>> from lair.hrrr import get_nearest_point
    >>> fs = s3fs.S3FileSystem(anon=True)
    >>> index = xr.open_zarr(
    ...     s3fs.S3Map("s3://hrrrzarr/grid/HRRR_chunk_index.zarr", s3=fs)
    ... )  # doctest: +SKIP
    >>> point = get_nearest_point(-111.848, 40.766, index)  # doctest: +SKIP
    >>> str(point.chunk_id.values)  # doctest: +SKIP
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

    See Also
    --------
    decompress_chunk : Turns the bytes into an array.
    get_value : Does both, and picks the point out of the chunk.

    Examples
    --------
    Needs the internet:

    >>> import boto3
    >>> from botocore import UNSIGNED
    >>> from botocore.config import Config
    >>> from lair.hrrr import retrieve_object
    >>> s3 = boto3.resource(
    ...     "s3", region_name="us-west-1", config=Config(signature_version=UNSIGNED)
    ... )  # doctest: +SKIP
    >>> key = "sfc/20240115/20240115_12z_anl.zarr/10m_above_ground/UGRD/10m_above_ground/UGRD/4.5"
    >>> data = retrieve_object(s3, key)  # doctest: +SKIP
    """
    obj = s3.Object("hrrrzarr", s3_url)
    return obj.get()["Body"].read()


def generate_zarr_ids(
    times: list[dt.datetime],
    level_type: Literal["sfc", "prs"],
    variables: list[tuple[str, str]],
    model_type: Literal["anl", "fcst"],
):
    """
    Generate ZarrId instance for multiple times and variables

    Parameters
    ----------
    times : [dt.datetime]
        list of datetime objects with hour specified. Naive times follow
        :ref:`the UTC rule <utc-rule>`; tz-aware times are converted to UTC when
        the S3 keys are built.
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

    See Also
    --------
    ZarrId : What it makes.
    Winds : Calls it for the 10 m u and v.

    Examples
    --------
    Two hours of the 10 m u wind: one key per variable, with a list of one
    ``ZarrId`` per time:

    >>> import datetime as dt
    >>> from lair.hrrr import generate_zarr_ids
    >>> times = [dt.datetime(2024, 1, 15, 12), dt.datetime(2024, 1, 15, 13)]
    >>> ids = generate_zarr_ids(
    ...     times, "sfc", [("UGRD", "10m_above_ground")], model_type="anl"
    ... )
    >>> list(ids)
    [('UGRD', '10m_above_ground')]
    >>> [z.run_hour.hour for z in ids[("UGRD", "10m_above_ground")]]
    [12, 13]
    """
    ids = {}

    for var in variables:
        ids[var] = []
        for time in times:
            ID = ZarrId(
                run_hour=time,
                level_type=level_type,
                var_name=var[0],
                var_level=var[1],
                model_type=model_type,
            )
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

    See Also
    --------
    retrieve_object : Gets the bytes it takes.
    get_value : Does both, and picks the point out of the chunk.

    Examples
    --------
    An analysis chunk is 150 by 150 half-precision numbers; here, a made-up one:

    >>> import datetime as dt
    >>> import numcodecs
    >>> import numpy as np
    >>> from lair.hrrr import ZarrId, decompress_chunk
    >>> zarr_id = ZarrId(
    ...     dt.datetime(2024, 1, 15, 12), "sfc", "10m_above_ground", "UGRD", "anl"
    ... )
    >>> raw = np.arange(150 * 150, dtype="<f2").tobytes()
    >>> chunk = decompress_chunk(
    ...     zarr_id, bytes(numcodecs.blosc.compress(raw, b"zstd", 5))
    ... )
    >>> chunk.shape, chunk[0, :3]
    ((150, 150), array([0., 1., 2.], dtype=float16))
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
        arr = np.reshape(chunk, (len(chunk) // entry_size, 150, 150))

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
    chunk_id : str
        Chunk id, for example ``"4.5"``.
    nearest_point : xr.Dataset
        nearest point

    Returns
    -------
    np.array
        array of values

    See Also
    --------
    retrieve_object : Gets the chunk's bytes.
    decompress_chunk : Turns them into an array.
    get_nearest_point : Gives the ``chunk_id`` and ``nearest_point``.

    Examples
    --------
    ``Winds`` does this for the 10 m u and v. For one hour of u at the nearest
    point (needs the internet; ``s3`` and ``point`` are as in
    :func:`retrieve_object` and :func:`get_nearest_point`):

    >>> import datetime as dt
    >>> from lair.hrrr import ZarrId, get_value
    >>> zarr_id = ZarrId(
    ...     dt.datetime(2024, 1, 15, 12), "sfc", "10m_above_ground", "UGRD", "anl"
    ... )
    >>> u = get_value(s3, zarr_id, point.chunk_id.values, point)  # doctest: +SKIP
    """
    compressed_data = retrieve_object(s3, create_s3_chunk_url(zarr_id, chunk_id))
    chunk_data = decompress_chunk(zarr_id, compressed_data)
    if zarr_id.model_type == "fcst":
        return chunk_data[
            :, nearest_point.in_chunk_y.values, nearest_point.in_chunk_x.values
        ]
    else:
        return chunk_data[
            nearest_point.in_chunk_y.values, nearest_point.in_chunk_x.values
        ]
