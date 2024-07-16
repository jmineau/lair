"""
Functions for working with spatial grid data.
"""

import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LatitudeFormatter, LatitudeLocator,
                                LongitudeFormatter, LongitudeLocator)
import matplotlib.pyplot as plt
import numpy as np
from xarray import DataArray
from xarray import DataArray


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


def extent2bbox(extent: list[float]) -> list[float]:
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


def wrap_lons(lons):
    '''
    Wrap longitudes ranging from 0~360 to -180~180

    Parameters
    ----------
    lons : array-like  # TODO
        Longitudes

    Returns
    -------
    array-like
        Wrapped longitudes
    '''
    return (lons.round(2) + 180) % 360 - 180


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


def area_grid(lat, lon):
    """
    Calculate the area of each grid cell [m2]

    Parameters
    ----------
    lat : array-like
        vector of latitude in degrees
    lon : array-like
        vector of longitude in degrees

    Returns
    -------
    array-like
        grid-cell area in square-meters with dimensions, [lat,lon]

    Notes
    -----
     - Originally copied from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7
     - Based on the function in https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos

    xlon, ylat = meshgrid(lon, lat)
    R = earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    return area


def area_DataArray(da: DataArray) -> DataArray:
    """
    Calculate the area of each grid cell in xarray dataarray [m2]

    Parameters
    ----------
    da : xr.DataArray
        DataArray with dimensions [lat, lon]

    Returns
    -------
    xr.DataArray
        grid-cell area in square-meters

    Notes
    -----
    Originally copied from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7
    """

    from xarray import DataArray

    lat_name, lon_name = da.rio.y_dim, da.rio.x_dim

    da = da.sortby(lat_name)  # sort lats so areas aren't negative
    lat, lon = da[lat_name], da[lon_name]

    area = area_grid(lat, lon)
    ada = DataArray(area,
                    dims=[lat_name, lon_name],
                    coords={lat_name: lat, lon_name: lon},
                    attrs={"long_name": "area_per_pixel",
                           "description": "area per pixel",
                           "units": "m^2"})

    return ada


def earth_radius(lat: float | np.ndarray) -> float | np.ndarray:
    '''
    Calculate radius of Earth assuming oblate spheroid defined by WGS84

    Parameters
    ----------
    lat : array-like
        latitudes in degrees

    Returns
    -------
    array-like
        vector of radius in meters

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

    return r


def haversine(lat1, lon1, lat2, lon2, R=6371, deg=True):
    # http://www.movable-type.co.uk/scripts/latlong.html

    if deg:
        lat1, lon1, lat2, lon2 = np.deg2rad([lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin((dlat) / 2)**2 + np.cos(lat1) \
        * np.cos(lat2) * np.sin((dlon) / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d


def bearing(lat1, lon1, lat2, lon2, deg=True, final=False):
    # http://www.movable-type.co.uk/scripts/latlong.html

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


class CRS_Converter:
    # TODO
    def __init__(self, in_crs):
        self.crs = in_crs

        self._type = self._which_type()

    def _which_type(self):
        'Determine type of in_crs'

    def to_cartopy(self):
        pass

    def to_pyproj(self):
        pass

    def to_rasterio(self):
        pass
