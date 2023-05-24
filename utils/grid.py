#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 22:53:55 2023

@author: James Mineau
"""


def dms2dd(d=0.0, m=0.0, s=0.0):  # degree-minute-second to decimal degree
    dd = float(d) + float(m) / 60 + float(s) / 3600
    return dd


def bbox2extent(bbox):
    minx, miny, maxx, maxy = bbox
    extent = [minx, maxx, miny, maxy]
    return extent


def extent2bbox(extent):
    minx, maxx, miny, maxy = extent
    bbox = [minx, miny, maxx, maxy]
    return bbox


def wrap_lons(lons):
    '''Wrap longitudes ranging from 0~360 to -180~180'''

    return (lons.round(2) + 180) % 360 - 180


def add_lat_ticks(ax, ylims, labelsize=None, more_ticks=0):
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LatitudeFormatter, LatitudeLocator

    fig = ax.figure
    bins = (fig.get_size_inches()[1] * fig.dpi / 100).astype(int) + 1

    y_ticks = LatitudeLocator(nbins=bins + more_ticks, prune='both')\
        .tick_values(ylims[0], ylims[1])

    ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
    ax.yaxis.tick_left()
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    if labelsize is not None:
        ax.tick_params(axis='y', labelsize=labelsize)


def add_lon_ticks(ax, xlims, rotation=0, labelsize=None, more_ticks=0):
    import cartopy.crs as ccrs
    from cartopy.mpl.ticker import LongitudeFormatter, LongitudeLocator

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


def add_latlon_ticks(ax, extent, x_rotation=0, labelsize=None,
                     more_lon_ticks=0, more_lat_ticks=0):
    xlims, ylims = extent[:2], extent[2:]

    add_lat_ticks(ax, ylims, labelsize=labelsize, more_ticks=more_lat_ticks)

    add_lon_ticks(ax, xlims, rotation=x_rotation, labelsize=labelsize,
                  more_ticks=more_lon_ticks)


def area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters

    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees

    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]

    Notes
    -----------
    Originally copied from
        https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7

    Based on the function in
        https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
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


def area_DataArray(da):
    """
    Calculate the area of each grid cell in xarray dataarray
    Area is in square meters

    Input
    -----------
    da: rioxarray DataArray

    Output
    -----------
    ada: rioxarray DataArray with grid-cell area in square-meters

    Notes
    -----------
    Originally copied from
        https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7
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


def earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84

    Input
    ---------
    lat: vector or latitudes in degrees

    Output
    ----------
    r: vector of radius in meters

    Notes
    -----------
    Originally copied from
        https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7

    WGS84:
        https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    import numpy as np

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


def add_extent_map(fig, main_extent, main_extent_crs,
                   extent_map_rect, extent_map_extent, extent_map_proj,
                   color, linewidth, zorder=None):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from shapely.geometry import box

    extent_map_ax = fig.add_axes(extent_map_rect,
                                 projection=extent_map_proj, zorder=zorder)
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
