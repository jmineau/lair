"""
Geo-spatial utilities.
"""

import copy
from typing import Any, Literal, Sequence

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import rasterio.crs
import xesmf as xe
from cartopy.mpl.ticker import (LatitudeFormatter, LatitudeLocator,
                                LongitudeFormatter, LongitudeLocator)
from numpy.typing import ArrayLike
from shapely import Polygon
from typing_extensions import \
    Self  # requires python 3.11 to import from typing
from xarray import DataArray, Dataset

# ----- BOUNDS ----- #

def bbox2extent(bbox: list[float]) -> list[float]:
    """
    Bounding box to extent.

    Parameters
    ----------
    bbox : list[minx, miny, maxx, maxy]
        Bounding box

    Returns
    -------
    list[minx, maxx, miny, maxy]
        Extent
    """
    minx, miny, maxx, maxy = bbox
    extent = [minx, maxx, miny, maxy]
    return extent


def extent2bbox(extent: list[float] | tuple[float, float, float, float]
                ) -> list[float]:
    """
    Extent to bounding box.

    Parameters
    ----------
    extent : list[minx, maxx, miny, maxy]
        Extent

    Returns
    -------
    list[minx, miny, maxx, maxy]
        Bounding box
    """
    minx, maxx, miny, maxy = extent
    bbox = [minx, miny, maxx, maxy]
    return bbox


# ----- COORDINATES ----- #

PC = ccrs.PlateCarree()  # Plate Carree projection


class CRS:
    """
    Coordinate Reference System (CRS) class.

    This class is a wrapper around the pyproj.CRS class, with additional methods
    for converting to other CRS classes.

    See https://pyproj4.github.io/pyproj/stable/crs_compatibility.html#cartopy for more information.

    .. note::
        `osgeo`, `fiona`, and `pycrs` conversions have not been implemented.

    Attributes
    ----------
    crs : pyproj.CRS
        Pyproj CRS object
    epsg : int | None
        EPSG code of the CRS
    proj4 : str
        PROJ4 string of the CRS
    wkt : str
        WKT string of the CRS
    """

    def __init__(self, crs: Any):
        # Convert input to pyproj.CRS
        if isinstance(crs, int):
            self.crs = pyproj.CRS.from_epsg(crs)
        elif isinstance(crs, str) and crs.startswith('EPSG:'):
            epsg = int(crs.split(':')[1])
            self.crs = pyproj.CRS.from_epsg(epsg)
        elif isinstance(crs, ccrs.CRS):
            self.crs = pyproj.CRS.from_user_input(crs)
        elif isinstance(crs, pyproj.CRS):
            self.crs = crs
        elif isinstance(crs, rasterio.CRS):
            with rasterio.Env(OSR_WKT_FORMAT="WKT2_2018"):
                self.crs = pyproj.CRS.from_wkt(crs.wkt)
        else:
            # If not any of the above, try to convert to pyproj.CRS
            # using the from_user_input method
            self.crs = pyproj.CRS.from_user_input(crs)

    def __repr__(self):
        return repr(self.crs)

    def __str__(self):
        return str(self.crs)

    @property
    def epsg(self) -> int | None:
        """Get the EPSG code of the CRS."""
        return self.crs.to_epsg()

    @property
    def proj4(self) -> str:
        """Get the PROJ4 string of the CRS."""
        return self.crs.to_proj4()

    @property
    def units(self) -> str:
        """Get the units of the CRS."""
        return self.to_rasterio().linear_units

    @property
    def wkt(self) -> str:
        """Get the WKT string of the CRS."""
        return self.crs.to_wkt()

    def to_cartopy(self) -> ccrs.CRS:
        """Convert to cartopy CRS."""
        return ccrs.CRS(self.crs)

    def to_rasterio(self) -> rasterio.CRS:
        """Convert to rasterio CRS."""
        return rasterio.CRS.from_user_input(self.crs)

    def to_pyproj(self) -> pyproj.CRS:
        """Convert to pyproj CRS."""
        return self.crs


def dms2dd(d: float=0.0, m: float=0.0, s: float=0.0) -> float:
    """
    Degree-minute-second to decimal degree

    Parameters
    ----------
    d : float, optional
        Degrees, by default 0.0
    m : float, optional
        Minutes, by default 0.0
    s : float, optional
        Seconds, by default 0.0

    Returns
    -------
    float
        Decimal degrees

    Raises
    ------
    ValueError
        If any of the inputs are not floats
    """
    try:
        dd = float(d) + float(m) / 60 + float(s) / 3600
        return dd
    except ValueError:
        return np.nan


def wrap_lons(lons: np.ndarray) -> ArrayLike:
    '''
    Wrap longitudes ranging from 0~360 to -180~180

    Parameters
    ----------
    lons : np.ndarray
        Longitudes

    Returns
    -------
    array-like
        Wrapped longitudes
    '''
    return (lons + 180) % 360 - 180


# ----- PLOTTING UTILITIES ----- #

def add_lat_ticks(ax: plt.Axes, ylims: list[float], labelsize: int | None=None, more_ticks: int=0) -> None:
    """
    Add latitude ticks to the map.

    Parameters
    ----------
    ax : plt.Axes
        Axes object
    ylims : list[float]
        Latitude limits
    labelsize : int, optional
        Font size of the labels, by default None
    more_ticks : int, optional
        Number of additional ticks, by default 0

    Returns
    -------
    None
    """
    fig = ax.figure
    bins = (fig.get_size_inches()[1] * fig.dpi / 100).astype(int) + 1

    y_ticks = LatitudeLocator(nbins=bins + more_ticks, prune='both')\
        .tick_values(ylims[0], ylims[1])

    ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
    ax.yaxis.tick_left()
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    if labelsize is not None:
        ax.tick_params(axis='y', labelsize=labelsize)

    return None


def add_lon_ticks(ax: plt.Axes, xlims: list[float], rotation: int=0, labelsize: int | None=None, more_ticks: int=0) -> None:
    """
    Add longitude ticks to the map.

    Parameters
    ----------
    ax : plt.Axes
        Axes object
    xlims : list[float]
        Longitude limits
    rotation : int, optional
        Rotation of the labels, by default 0
    labelsize : int, optional
        Font size of the labels, by default None
    more_ticks : int, optional
        Number of additional ticks, by default 0

    Returns
    -------
    None
    """
    fig = ax.figure
    bins = (fig.get_size_inches()[0] * fig.dpi / 100).astype(int) + 1

    x_ticks = LongitudeLocator(nbins=bins + more_ticks, prune='both')\
        .tick_values(xlims[0], xlims[1])

    ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.tick_bottom()
    ax.xaxis.set_major_formatter(LongitudeFormatter())

    if rotation != 0:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation,
                           ha='right', rotation_mode='anchor')

    if labelsize is not None:
        ax.tick_params(axis='x', labelsize=labelsize)

    return None


def add_latlon_ticks(ax: plt.Axes, extent: list[float], x_rotation: int=0, labelsize: int | None=None,
                     more_lon_ticks: int=0, more_lat_ticks: int=0) -> None:
    """
    Add latitude and longitude ticks to the map.

    Parameters
    ----------
    ax : plt.Axes
        Axes object
    extent : list[float]
        Extent of the map. [minx, maxx, miny, maxy]
    x_rotation : int, optional
        Rotation of the longitude labels, by default 0
    labelsize : int, optional
        Font size of the labels, by default None
    more_lon_ticks : int, optional
        Number of additional longitude ticks, by default 0
    more_lat_ticks : int, optional
        Number of additional latitude ticks, by default 0

    Returns
    -------
    None
    """
    xlims, ylims = extent[:2], extent[2:]

    add_lat_ticks(ax, ylims, labelsize=labelsize, more_ticks=more_lat_ticks)

    add_lon_ticks(ax, xlims, rotation=x_rotation, labelsize=labelsize,
                  more_ticks=more_lon_ticks)

    return None

def add_extent_map(fig: 'matplotlib.figure.Figure', main_extent: list[float], main_extent_crs: ccrs.CRS,
                   extent_map_rect: list[float], extent_map_extent: list[float], extent_map_crs: ccrs.CRS,
                   color: str, linewidth: int, zorder: int | None=None) -> plt.Axes:
    """
    Add an extent map to the figure.
    
    TODO This needs better naming and documentation.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object
    main_extent : list[float]
        Extent of the main map
    main_extent_crs : ccrs.CRS
        CRS of the main extent
    extent_map_rect : list[float]
        Rectangle of the extent map
    extent_map_extent : list[float]
        Extent of the extent map
    extent_map_crs : ccrs.CRS
        CRS of the extent map
    color : str
        Color of the extent map
    linewidth : int
        Line width of the extent map
    zorder : int, optional
        Zorder of the extent map, by default None

    Returns
    -------
    plt.Axes
        Axes object of the extent map
    """
    import cartopy.feature as cfeature
    from shapely.geometry import box

    extent_map_ax = fig.add_axes(extent_map_rect,
                                 projection=extent_map_crs, zorder=zorder)
    extent_map_ax.set_extent(extent_map_extent)

    extent_map_ax.add_feature(cfeature.LAND)
    extent_map_ax.add_feature(cfeature.OCEAN)
    extent_map_ax.add_feature(cfeature.STATES)

    # # plot the extent of the main map on the extent map
    # x_coords = main_extent[0:2] + [main_extent[1],
    #                                main_extent[0],
    #                                main_extent[0]]
    # y_coords = main_extent[2:4] + [main_extent[2],
    #                                main_extent[2],
    #                                main_extent[3]]
    # extent_map_ax.plot(x_coords, y_coords,
    #                    transform=main_extent_crs,
    #                    color=color, linewidth=linewidth)

    main_poly = box(*extent2bbox(main_extent))

    extent_map_ax.add_geometries([main_poly], crs=main_extent_crs,
                                 color=color, linewidth=linewidth)

    return extent_map_ax


# ----- XARRAY UTILITIES ----- #

XESMF_Regrid_Methods = Literal[
        "bilinear",
        "conservative",
        "conservative_normed",
        "nearest_s2d",
        "nearest_d2s",
        "patch",
    ]


class BaseGrid:
    """
    Base class for working with gridded data.
    
    This class is a wrapper around xarray DataArray and Dataset objects, with additional methods
    for clipping, regridding, resampling, and reprojection. All operations are performed inplace,
    but return the grid object for chaining.
    """

    def __init__(self, data, crs):
        self.crs = CRS(crs)
        self.data = write_rio_crs(data, self.crs)

    def copy(self) -> Self:
        """
        Create a copy of the grid.

        Returns
        -------
        BaseGrid
            The copied grid.
        """
        return copy.deepcopy(self)

    @property
    def gridcell_area(self) -> DataArray:
        """
        Calculate the grid cell area in km^2.

        Returns
        -------
        xr.DataArray
            The grid cell area.
        """
        return gridcell_area(self.data)

    def clip(self,
             bbox: tuple[float, float, float, float] | None = None,
             extent: tuple[float, float, float, float] | None = None,
             geom: Polygon | None = None,
             crs: Any = None,
             **kwargs: Any) -> Self:
        """
        Clip the data to the given bounds. Mopdifies the data in place.

        Input bounds must be in the same CRS as the data.

        .. note::
            The result can be slightly different between supplying a geom and a bbox/extent.
            Clipping with a geom seems to be exclusive of the bounds,
            while clipping with a bbox/extent seems to be inclusive of the bounds.

        Parameters
        ----------
        bbox : tuple[minx, miny, maxx, maxy]
            The bounding box to clip the data to.
        extent : tuple[minx, maxx, miny, maxy]
            The extent to clip the data to.
        geom : shapely.Polygon
            The geometry to clip the data to.
        crs : Any
            The CRS of the input geometries. If not provided, the CRS of the data is used.
        inplace : bool, optional
            Whether to modify the data in place, by default False.
        kwargs : Any
            Additional keyword arguments to pass to the rioxarray clip method.

        Returns
        -------
        BaseGrid
            The clipped grid
        """
        crs = crs or self.crs.to_rasterio()
        self.data = clip(self.data, bbox=bbox, extent=extent, geom=geom, crs=crs, **kwargs)
        return self

    def regrid(self, out_grid: Dataset,
               method: XESMF_Regrid_Methods = 'bilinear') -> Self:
        """
        Regrid the data to a new grid. Uses `xesmf` for regridding. Modifies the data in place.

        .. note::
            At present, `xesmf` only supports regridding lat-lon grids. self.data must be on a lat-lon grid.
            Possibly could use `xesmf.frontend.BaseRegridder` to regrid to a generic grid.

        .. warning::
            `xarray.Dataset.cf.add_bounds` is known to have issues, including near the 180th meridian.
            Care should be taken when using this method, especially with global datasets.

        Parameters
        ----------
        out_grid : xr.DataArray
            The new grid to resample to. Must be a lat-lon grid.
        method : str, optional
            The regridding method, by default 'bilinear'.

        Returns
        -------
        BaseGrid
            The regridded grid
        """
        self.data = regrid(self.data, out_grid=out_grid, method=method)
        return self

    def resample(self, resolution: float | tuple[float, float],
                 regrid_method: XESMF_Regrid_Methods = 'bilinear') -> Self:
        """
        Resample the data to a new resolution. Modifies the data in place.

        Parameters
        ----------
        resolution : float | tuple[x_res, y_res]
            The new resolution in degrees. If a single value is provided, the resolution
            is assumed to be the same in both dimensions.
        regrid_method : str, optional
            The regridding method, by default 'bilinear'.

        Returns
        -------
        BaseGrid
            The resampled grid
        """
        self.data = resample(self.data, resolution=resolution, regrid_method=regrid_method)
        return self

    def reproject(self, resolution: float | tuple[float, float],
                  regrid_method: XESMF_Regrid_Methods = 'bilinear') -> Self:
        """
        Reproject the data to a lat lon rectilinear grid.

        Parameters
        ----------
        resolution : float | tuple[x_res, y_res]
            The new resolution in degrees. If a single value is provided, the resolution
            is assumed to be the same in both dimensions.
        regrid_method : str, optional
            The regridding method, by default 'bilinear'.

        Returns
        -------
        BaseGrid
            The reprojected grid
        """
        assert self.crs.epsg != 4326, 'Data is already in lat lon'

        resampled_data = resample(self.data, resolution=resolution,
                                  regrid_method=regrid_method)

        self.crs = CRS('EPSG:4326')
        self.data = write_rio_crs(resampled_data, self.crs)
        return self


def clip(data: DataArray | Dataset,
         bbox: list[float] | tuple[float, float, float, float] | None = None,
         extent: list[float] | tuple[float, float, float, float] | None = None,
         geom: Polygon | list[Polygon] | None = None,
         crs: Any='EPSG:4326',
         **kwargs: Any
         ) -> DataArray | Dataset:
    """
    Clip the data to the given bounds.

    Input bounds must be in the same CRS as the data.

    .. note::
        The result can be slightly different between supplying a geom and a bbox/extent.
        Clipping with a geom seems to be exclusive of the outer bounds,
        while clipping with a bbox/extent seems to be inclusive of the outer bounds.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to clip.
    bbox : tuple[minx, miny, maxx, maxy]
        The bounding box to clip the data to.
    extent : tuple[minx, maxx, miny, maxy]
        The extent to clip the data to.
    geom : shapely.Polygon
        The geometry or geometries to clip the data to.
    crs : Any, optional
        The CRS of the input geometries. Default is 'EPSG:4326'.
    kwargs : Any
        Additional keyword arguments to pass to the rioxarray clip method.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The clipped data.
    """
    assert (bbox is not None) + (extent is not None) + (geom is not None) == 1, 'Only one of bbox, extent, or geom must be provided.'

    if extent is not None:
        # Convert extent to bbox
        bbox = extent2bbox(extent)
    if bbox is not None:
        data = data.rio.clip_box(*bbox, crs=crs, **kwargs)
    elif geom is not None:
        if not isinstance(geom, Sequence):
            geom = [geom]

        data = data.rio.clip(geom, crs=crs, **kwargs)

    return data


def gridcell_area(grid: DataArray | Dataset, R: float | ArrayLike | None = None
                  ) -> DataArray:
    """
    Calculate the area of each grid cell in a grid.

    .. note::
        For lat-lon grids, `xesmf.utils.grid_area` is used to calculate the area,
        which requires the radius of the earth in kilometers. If the radius of the
        earth is not provided, it will be calculated based on the latitude.

    Parameters
    ----------
    grid : xr.DataArray | xr.Dataset
        Grid data. `rioxarray` coords must be set.
    R : float, optional
        Radius of earth in kilometers, by default calculated based on the latitude.

    Returns
    -------
    np.ndarray | xr.DataArray
        grid-cell area in square-kilometers
    """

    if grid.rio.crs == 'EPSG:4326':
        R = R or earth_radius(grid['lat'])
        area = xe.util.cell_area(grid, earth_radius=R)
    elif grid.rio.crs.linear_units == 'metre':
        bounds = grid.cf.add_bounds(['x', 'y'])
        dx = bounds.x_bounds.diff('bounds').squeeze()
        dy = bounds.y_bounds.diff('bounds').squeeze()
        cell_area_m2 = dx * dy
        area = cell_area_m2.pint.quantify('m2')\
            .pint.to('km2')\
            .pint.dequantify()
    else:
        raise ValueError('Only lat-lon and meter grids are supported.')
    return area


def regrid(data: DataArray | Dataset,
           out_grid: Dataset,
           method: XESMF_Regrid_Methods = 'bilinear') -> DataArray | Dataset:
    """
    Regrid data to a new grid. Uses `xesmf` for regridding.

    .. note::
        At present, `xesmf` only supports regridding lat-lon grids. self.data must be on a lat-lon grid.
        Possibly could use `xesmf.frontend.BaseRegridder` to regrid to a generic grid.

    .. warning::
        `xarray.Dataset.cf.add_bounds` is known to have issues, including near the 180th meridian.
        Care should be taken when using this method, especially with global datasets.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to regrid.
    out_grid : xr.DataArray
        The new grid to resample to. Must be a lat-lon grid.
    method : str, optional
        The regridding method, by default 'bilinear'.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The regridded data.
    """
    out_crs = 'EPSG:4326'

    # Use cf-xarray to calculate the bounds of the grid cells
    data = data.cf.add_bounds(['lat', 'lon'])

    # Regrid the data using `xesmf`
    regridder = xe.Regridder(ds_in=data, ds_out=out_grid, method=method)
    regridded = regridder(data, keep_attrs=True)

    if len(regridded.lon.dims) == 2:
        # New grid has 2D lat/lon, but lat is constant over x axis,
        # and lon is constant over y axis.
        # We need to convert to 1D lat/lon
        lats = regridded.lat.isel(x=0).values
        lons = regridded.lon.isel(y=0).values
        regridded = regridded.drop_vars(['lat', 'lon'])\
            .rename_dims({'x': 'lon', 'y': 'lat'})\
            .assign_coords(lat=lats, lon=lons)

    # Regridding drops rioxarray info - reattach
    regridded.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    regridded = write_rio_crs(regridded, out_crs)

    return regridded


def resample(data: DataArray | Dataset,
             resolution: float | tuple[float, float],
             regrid_method: XESMF_Regrid_Methods = 'bilinear') -> DataArray | Dataset:
    """
    Resample the data to a new resolution. Modifies the data in place.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data to resample.
    resolution : float | tuple[x_res, y_res]
        The new resolution in degrees. If a single value is provided, the resolution
        is assumed to be the same in both dimensions.
    regrid_method : str, optional
        The regridding method, by default 'bilinear'. 

    Returns
    -------
    xr.DataArray | xr.Dataset
        The resampled data.
    """
    if isinstance(resolution, float):
        resolution: tuple = (resolution, resolution)

    # Calculate the new grid
    bounds = data.cf.add_bounds(['lat', 'lon'])
    xmin = bounds.lon_bounds.min()
    xmax = bounds.lon_bounds.max()
    ymin = bounds.lat_bounds.min()
    ymax = bounds.lat_bounds.max()
    dx = resolution[0]
    dy = resolution[1]
    if len(data.lon.dims) == 2:
        out_grid = xe.util.grid_2d(xmin, xmax, dx,
                                    ymin, ymax, dy)
    else:
        out_grid = Dataset({
            "lat": (["lat"], np.arange(ymin + dy/2, ymax - dy/2, dy), {"units": "degrees_north"}),
            "lon": (["lon"], np.arange(xmin + dx/2, xmax - dx/2, dx), {"units": "degrees_east"}),
            })

    return regrid(data, out_grid=out_grid, method=regrid_method)


def round_latlon(data: DataArray | Dataset,
                 lat_deci: int, lon_deci: int,
                 lat_dim: str = 'lat', lon_dim: str = 'lon') -> Dataset:
    """
    Round latitude and longitude values to a specified number of decimal places.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The data with lat/lon coordinates to round.
    lat_deci : int
        Number of decimal places to round latitude values to.
    lon_deci : int
        Number of decimal places to round longitude values to.
    lat_dim : str, optional
        Name of the latitude dimension, by default 'lat'.
    lon_dim : str, optional
        Name of the longitude dimension, by default 'lon'.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The data with rounded lat/lon coordinates.
    """
    return data.assign_coords({
        lat_dim: np.round(data[lat_dim], lat_deci),
        lon_dim: np.round(data[lon_dim], lon_deci)
    })


def write_rio_crs(data: DataArray | Dataset, crs: Any) -> DataArray | Dataset:
    """
    Write the CRS and coordinate system to the rioxarray accessor.

    Parameters
    ----------
    data : DataArray | Dataset
        The data to write the CRS to.
    crs : Any
        The CRS to write to the data.

    Returns
    -------
    DataArray | Dataset
        The data with the CRS written to the rioxarray accessor.
    """
    if isinstance(crs, CRS):
        crs = crs.to_rasterio()

    data = data.rio.write_crs(crs)\
        .rio.write_coordinate_system(inplace=True)

    return data


# ----- MISCELLANEOUS ----- #

def bearing(lat1, lon1, lat2, lon2, deg=True, final=False):
    # http://www.movable-type.co.uk/scripts/latlong.html
    # TODO I dont think this works correctly

    if deg:
        lat1, lon1, lat2, lon2 = np.deg2rad([lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) \
        - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    # Inital bearing in degrees
    brng = (np.rad2deg(np.arctan2(y, x))
            + 360) % 360  # in [0,360)

    if final:
        # Final bearing in degrees
        brng = (brng + 180) % 360

    return brng


def cosine_weights(lats: np.ndarray) -> np.ndarray:
    """
    Calculate cosine weights from latitude.

    Parameters
    ----------
    lats : np.ndarray
        Latitude values

    Returns
    -------
    np.ndarray
        Cosine weighting

    Examples
    --------
    >>> ds: xr.Dataset
    >>> weights = cosine_weighting(ds.lat)
    >>> ds_weighted = ds.weighted(weights)
    """
    return np.cos(np.deg2rad(lats))


def earth_radius(lat: ArrayLike) -> ArrayLike:
    '''
    Calculate radius of Earth assuming oblate spheroid defined by WGS84

    Parameters
    ----------
    lat : array-like
        latitudes in degrees

    Returns
    -------
    array-like
        vector of radius in kilometers

    Notes
    -----
     - Originally copied from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7
     - WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2 / a**2)

    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = np.deg2rad(lat)
    lat_gc = np.arctan((1-e2) * np.tan(lat))

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5)
        / (1 - (e2 * np.cos(lat_gc)**2))**0.5
        )

    r /= 1000  # convert to km
    return r


def gridcell_area_from_latlon(lat: ArrayLike, lon: ArrayLike,
                              R: float | None = None) -> np.ndarray:
    """
    Calculate the area of each grid cell in a lat-lon grid.

    Parameters
    ----------
    lat : ArrayLike
        Latitude array
    lon : ArrayLike
        Longitude array

    Returns
    -------
    np.ndarray
        Grid-cell area in square-kilometers
    """
    lat, lon = np.array(lat), np.array(lon)
    grid = DataArray(coords={'lat': lat, 'lon': lon},
                     dims=['lat', 'lon'])
    grid.rio.set_spatial_dims('lon', 'lat', inplace=True)
    grid = write_rio_crs(grid, crs='EPSG:4326')

    area = gridcell_area(grid, R=R)
    return area.values


def haversine(lat1, lon1, lat2, lon2, R=6371, deg=True):
    # http://www.movable-type.co.uk/scripts/latlong.html
    # TODO not sure this is completely correct

    if deg:
        lat1, lon1, lat2, lon2 = np.deg2rad([lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin((dlat) / 2)**2 + np.cos(lat1) \
        * np.cos(lat2) * np.sin((dlon) / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d
