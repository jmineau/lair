"""
Geo-spatial utilities.
"""

from __future__ import (
    annotations,
)  # keep optional-dep annotations (e.g. shapely Polygon) lazy

import copy
import math
from collections import deque
from typing import Any, Literal, TypeVar, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike
from typing import Iterable
from typing_extensions import Self  # requires python 3.11 to import from typing
from xarray import DataArray, Dataset

from lair._optional import import_optional_dependency

#: An xarray object; functions annotated with it return the type they're given
_XarrayT = TypeVar("_XarrayT", bound=DataArray | Dataset)

cartopy = import_optional_dependency("cartopy")
pyproj = import_optional_dependency("pyproj")
rasterio = import_optional_dependency("rasterio")
rioxarray = import_optional_dependency("rioxarray")
shapely = import_optional_dependency("shapely")

import cartopy.crs as ccrs  # noqa: E402
from cartopy.mpl.geoaxes import GeoAxes  # noqa: E402
import rasterio.crs  # noqa: E402 F811
import rioxarray as rxr  # noqa: E402 F401
from cartopy.mpl.ticker import (  # noqa: E402
    LatitudeFormatter,
    LatitudeLocator,
    LongitudeFormatter,
    LongitudeLocator,
)
from shapely import LineString, Point, Polygon, MultiLineString  # noqa: E402


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


def extent2bbox(extent: list[float] | tuple[float, float, float, float]) -> list[float]:
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
        if isinstance(crs, CRS):
            self.crs = crs.crs
        elif isinstance(crs, int):
            self.crs = pyproj.CRS.from_epsg(crs)
        elif isinstance(crs, str) and crs.startswith("EPSG:"):
            epsg = int(crs.split(":")[1])
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


def dms2dd(d: float = 0.0, m: float = 0.0, s: float = 0.0) -> float:
    """
    Degree-minute-second to decimal degree

    The sign of ``d`` applies to the whole angle (``m`` and ``s`` are
    magnitudes), so ``dms2dd(-40, 30)`` is -40.5. For angles between 0 and
    -1 degree pass ``d=-0.0``.

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
        Decimal degrees, or NaN if any input cannot be converted to a float
        (e.g. ``None`` or a non-numeric string).
    """
    try:
        d, m, s = float(d), float(m), float(s)
    except (TypeError, ValueError):
        return np.nan
    # copysign keeps the sign of -0.0
    return float(np.copysign(abs(d) + m / 60 + s / 3600, d))


def wrap_lons(
    longitudes: npt.ArrayLike, base: float = -180.0, period: float = 360.0
) -> np.ndarray:
    """
    Transform the longitude values to be within the half-open interval
    [base, base + period).

    Parameters
    ----------
    longitudes : ArrayLike
        One or more longitude values (degrees) to be wrapped.
    base : float, default=-180.0
        The start limit (degrees) of the interval (included).
    period : float, default=360.0
        The length of the interval (degrees); ``base + period`` itself wraps
        to ``base``.

    Returns
    -------
    ndarray
        The transformed longitude values.

    Notes
    -----
    .. copied from https://github.com/jamesp/geovista/blob/4850c519c7a37c4765befa06fbab933350637c93/lib/geovista/common.py#L274

    """
    #
    # TODO: support radians
    #
    lons = np.asanyarray(
        longitudes if isinstance(longitudes, Iterable) else [longitudes]
    )
    result = ((lons.astype(np.float64) - base + period * 2) % period) + base

    return result


# ----- PLOTTING UTILITIES ----- #


def add_lat_ticks(
    ax: plt.Axes, ylims: list[float], labelsize: int | None = None, more_ticks: int = 0
) -> None:
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
    fig = cast(Figure, ax.figure)
    # One tick per inch of axis height (independent of dpi)
    bins = int(fig.get_size_inches()[1]) + 1

    y_ticks = LatitudeLocator(nbins=bins + more_ticks, prune="both").tick_values(
        ylims[0], ylims[1]
    )

    ax.set_yticks(y_ticks, crs=ccrs.PlateCarree())
    ax.yaxis.tick_left()
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    if labelsize is not None:
        ax.tick_params(axis="y", labelsize=labelsize)

    return None


def add_lon_ticks(
    ax: plt.Axes,
    xlims: list[float],
    rotation: int = 0,
    labelsize: int | None = None,
    more_ticks: int = 0,
) -> None:
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
    fig = cast(Figure, ax.figure)
    # One tick per inch of axis width (independent of dpi)
    bins = int(fig.get_size_inches()[0]) + 1

    x_ticks = LongitudeLocator(nbins=bins + more_ticks, prune="both").tick_values(
        xlims[0], xlims[1]
    )

    ax.set_xticks(x_ticks, crs=ccrs.PlateCarree())
    ax.xaxis.tick_bottom()
    ax.xaxis.set_major_formatter(LongitudeFormatter())

    if rotation != 0:
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=rotation, ha="right", rotation_mode="anchor"
        )

    if labelsize is not None:
        ax.tick_params(axis="x", labelsize=labelsize)

    return None


def add_latlon_ticks(
    ax: plt.Axes,
    extent: list[float],
    x_rotation: int = 0,
    labelsize: int | None = None,
    more_lon_ticks: int = 0,
    more_lat_ticks: int = 0,
) -> None:
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

    add_lon_ticks(
        ax, xlims, rotation=x_rotation, labelsize=labelsize, more_ticks=more_lon_ticks
    )

    return None


def add_extent_map(
    fig: "plt.Figure",
    main_extent: list[float],
    main_extent_crs: ccrs.CRS,
    extent_map_rect: tuple[float, float, float, float],
    extent_map_extent: list[float],
    extent_map_crs: ccrs.CRS,
    color: str,
    linewidth: int,
    zorder: int | None = None,
) -> plt.Axes:
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
    extent_map_rect : tuple[left, bottom, width, height]
        Rectangle of the extent map
    extent_map_extent : list[float]
        Extent of the extent map
    extent_map_crs : ccrs.CRS
        CRS of the extent map
    color : str
        Color of the main-extent outline drawn on the extent map
    linewidth : int
        Line width of the main-extent outline
    zorder : int, optional
        Zorder of the extent map, by default None

    Returns
    -------
    plt.Axes
        Axes object of the extent map
    """
    import cartopy.feature as cfeature
    from shapely.geometry import box

    extent_map_ax: GeoAxes = fig.add_axes(
        extent_map_rect, projection=extent_map_crs, zorder=zorder
    )
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

    # Outline only: `color=` would also fill the box
    extent_map_ax.add_geometries(
        [main_poly],
        crs=main_extent_crs,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
    )

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
    for clipping, regridding, resampling, and reprojection. Operations return a new
    grid by default; pass ``inplace=True`` to modify this one instead. Either way the
    grid object is returned, for chaining.
    """

    def __init__(self, data, crs, **kwargs):
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

    def clip(
        self,
        bbox: tuple[float, float, float, float] | None = None,
        extent: tuple[float, float, float, float] | None = None,
        geom: Polygon | None = None,
        crs: Any = None,
        inplace: bool = False,
        **kwargs: Any,
    ) -> Self:
        """
        Clip the data to the given bounds.

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
            Whether to modify the data in place. Default is False (returns a
            new copy with the clipped data).
        kwargs : Any
            Additional keyword arguments to pass to the rioxarray clip method.

        Returns
        -------
        BaseGrid
            The clipped grid
        """
        crs = crs or self.crs.to_rasterio()
        data = clip(self.data, bbox=bbox, extent=extent, geom=geom, crs=crs, **kwargs)
        if inplace:
            self.data = data
            return self
        else:
            new = self.copy()
            new.data = data
            return new

    def regrid(
        self,
        out_grid: Dataset,
        method: XESMF_Regrid_Methods = "bilinear",
        inplace: bool = False,
    ) -> Self:
        """
        Regrid the data to a new grid. Uses `xesmf` for regridding.

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
        inplace : bool, optional
            Whether to modify the object in-place. Default is False.

        Returns
        -------
        BaseGrid
            The regridded grid
        """
        data = regrid(self.data, out_grid=out_grid, method=method)
        if inplace:
            self.data = data
            return self
        else:
            new = self.copy()
            new.data = data
            return new

    def resample(
        self,
        resolution: float | tuple[float, float],
        regrid_method: XESMF_Regrid_Methods = "bilinear",
        inplace: bool = False,
    ) -> Self:
        """
        Resample the data to a new resolution.

        Parameters
        ----------
        resolution : float | tuple[x_res, y_res]
            The new resolution in degrees. If a single value is provided, the resolution
            is assumed to be the same in both dimensions.
        regrid_method : str, optional
            The regridding method, by default 'bilinear'.
        inplace : bool, optional
            Whether to modify the object in-place. Default is False.

        Returns
        -------
        BaseGrid
            The resampled grid
        """
        data = resample(self.data, resolution=resolution, regrid_method=regrid_method)
        if inplace:
            self.data = data
            return self
        else:
            new = self.copy()
            new.data = data
            return new

    def reproject(
        self,
        resolution: float | tuple[float, float],
        regrid_method: XESMF_Regrid_Methods = "bilinear",
        inplace: bool = False,
    ) -> Self:
        """
        Reproject the data to a lat lon rectilinear grid.

        Parameters
        ----------
        resolution : float | tuple[x_res, y_res]
            The new resolution in degrees. If a single value is provided, the resolution
            is assumed to be the same in both dimensions.
        regrid_method : str, optional
            The regridding method, by default 'bilinear'.
        inplace : bool, optional
            Whether to modify the object in-place. Default is False.

        Returns
        -------
        BaseGrid
            The reprojected grid
        """
        assert self.crs.epsg != 4326, "Data is already in lat lon"

        resampled_data = resample(
            self.data, resolution=resolution, regrid_method=regrid_method
        )

        if inplace:
            self.crs = CRS("EPSG:4326")
            self.data = write_rio_crs(resampled_data, self.crs)
            return self
        else:
            new = self.copy()
            new.crs = CRS("EPSG:4326")
            new.data = write_rio_crs(resampled_data, new.crs)
            return new


def clip(
    data: DataArray | Dataset,
    bbox: list[float] | tuple[float, float, float, float] | None = None,
    extent: list[float] | tuple[float, float, float, float] | None = None,
    geom: Polygon | Iterable[Polygon] | None = None,
    crs: Any = "EPSG:4326",
    **kwargs: Any,
) -> DataArray | Dataset:
    """
    Clip the data to the given bounds.

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
    geom : shapely.Polygon | Iterable[shapely.Polygon]
        The geometry, or a collection of geometries (list, array, GeoSeries),
        to clip the data to.
    crs : Any, optional
        The CRS of the input geometries. Default is 'EPSG:4326'.
    kwargs : Any
        Additional keyword arguments to pass to the rioxarray clip method.

    Returns
    -------
    xr.DataArray | xr.Dataset
        The clipped data.
    """
    assert (bbox is not None) + (extent is not None) + (geom is not None) == 1, (
        "Only one of bbox, extent, or geom must be provided."
    )

    if extent is not None:
        # Convert extent to bbox
        bbox = extent2bbox(extent)
    if bbox is not None:
        data = data.rio.clip_box(*bbox, crs=crs, **kwargs)
    elif geom is not None:
        # rio.clip takes a collection of geometries; wrap a single one
        if isinstance(geom, shapely.Geometry):
            geom = [geom]

        data = data.rio.clip(geom, crs=crs, **kwargs)

    return data


def gridcell_area(
    grid: DataArray | Dataset, R: float | ArrayLike | None = None
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
    R : float | array-like, optional
        Radius of earth in kilometers (lat-lon grids only), by default
        calculated based on the latitude. An array must broadcast against the
        grid, e.g. a DataArray along ``lat``.

    Returns
    -------
    xr.DataArray
        grid-cell area in square-kilometers, with the grid's (y, x) dims
    """
    # Optional dependency for advanced regridding
    xe = import_optional_dependency("xesmf")

    if grid.rio.crs == "EPSG:4326":
        if R is None:
            R = earth_radius(grid["lat"])
        area = xe.util.cell_area(grid, earth_radius=R)
    elif grid.rio.crs.linear_units == "metre":
        bounds = grid.cf.add_bounds(["x", "y"])
        dx = bounds.x_bounds.diff("bounds").squeeze("bounds", drop=True)
        dy = bounds.y_bounds.diff("bounds").squeeze("bounds", drop=True)
        # abs: north-up rasters store y descending (negative dy).
        # dy first so the result is (y, x) like the data.
        cell_area_m2 = abs(dy * dx)
        area = cell_area_m2.pint.quantify("m2").pint.to("km2").pint.dequantify()
    else:
        raise ValueError("Only lat-lon and meter grids are supported.")
    return area


def plot_grid(
    grid: DataArray | Dataset,
    lw: float = 1,
    ax: plt.Axes | None = None,
    extent: list[float] | None = None,
    crs: Any = None,
    **kwargs: Any,
) -> plt.Axes:
    """
    Plot a grid.

    Parameters
    ----------
    grid : xr.DataArray | xr.Dataset
        The grid to plot. Must have 1D ``lat`` and ``lon`` coordinates.
    lw : float, optional
        Line width of the cell edges, by default 1.
    ax : plt.Axes, optional
        Axes object to plot to, by default None (a new map in ``crs``).
    extent : list[minx, maxx, miny, maxy], optional
        Extent of the plot in longitude/latitude degrees, whatever the map
        projection, by default None.
    crs : Any, optional
        Map projection for a new axes: a cartopy projection, or anything
        :class:`CRS` accepts (EPSG code, "EPSG:..." string, pyproj CRS, ...).
        By default PlateCarree. Ignored when ``ax`` is given.
    kwargs : Any
        Additional keyword arguments to pass to the pcol
        or pcolormesh method.

    Returns
    -------
    plt.Axes
        Axes object
    """
    if crs is None:
        crs = PC
    elif not isinstance(crs, ccrs.Projection):
        # A GeoAxes needs a cartopy Projection (CRS.to_cartopy gives a plain CRS)
        crs = ccrs.Projection(CRS(crs).to_pyproj())

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": crs})
    ax = cast(GeoAxes, ax)

    if extent is not None:
        ax.set_extent(extent, crs=PC)

    lat, lon = grid["lat"], grid["lon"]

    grid = DataArray(
        np.zeros((len(lat), len(lon)), dtype=float), coords={"lat": lat, "lon": lon}
    )

    grid.plot(
        ax=ax, transform=PC, facecolor="none", edgecolor="black", linewidth=lw, **kwargs
    )

    return ax


def generate_regular_grid(
    xmin: float,
    xmax: float,
    dx: float,
    ymin: float,
    ymax: float,
    dy: float,
    x_label="x",
    y_label="y",
    chunks: dict | None = None,
) -> DataArray:
    """
    Generate a regular grid. Grid points are cell centers.

    Cells start at ``xmin``/``ymin`` and continue while their centre is below
    ``xmax``/``ymax`` (a centre landing exactly on the max edge may or may not
    be included, depending on float error). Centres are
    ``min + d * (i + 0.5)``, rounded to 10 decimals only to drop float noise
    (``40.35``, not ``40.349999999999994``), which matches PYSTILT's grid axes.

    Parameters
    ----------
    xmin : float
        x value of the left edge of the grid
    xmax : float
        x value of the right edge of the grid
    dx : float
        x resolution
    ymin : float
        y value of the bottom edge of the grid
    ymax : float
        y value of the top edge of the grid
    dy : float
        y resolution
    x_label : str, optional
        Name of the x coordinate, by default 'x'
    y_label : str, optional
        Name of the y coordinate, by default 'y'
    chunks : dict, optional
        If given, chunk the grid with dask (``DataArray.chunk(chunks)``).

    Returns
    -------
    xr.DataArray
        The generated grid, dims (y_label, x_label)
    """
    x = _cell_centers(xmin, xmax, dx)
    y = _cell_centers(ymin, ymax, dy)
    zeros = np.zeros((len(y), len(x)))
    grid = DataArray(zeros, coords={y_label: y, x_label: x})
    if chunks is not None:
        grid = grid.chunk(chunks)
    return grid


def _cell_centers(vmin: float, vmax: float, d: float) -> np.ndarray:
    """Centres of the cells of size ``d`` from ``vmin`` whose centre is below ``vmax``."""
    vmin, vmax, d = float(vmin), float(vmax), float(d)
    # Cell count exactly as before (np.arange). Where the last centre lands on
    # vmax, float error decides whether that cell is included.
    n = len(np.arange(vmin + d / 2, vmax, d))
    # Each centre from vmin directly (no accumulated error), then drop float noise
    return np.round(vmin + d * (np.arange(n) + 0.5), 10)


def regrid(
    data: DataArray | Dataset,
    out_grid: DataArray | Dataset,
    method: XESMF_Regrid_Methods = "bilinear",
) -> DataArray | Dataset:
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
    # Optional dependency for advanced regridding
    xe = import_optional_dependency("xesmf")

    out_crs = "EPSG:4326"

    # Use cf-xarray to calculate the bounds of the grid cells
    data = data.cf.add_bounds(["lat", "lon"])

    # Regrid the data using `xesmf`
    regridder = xe.Regridder(ds_in=data, ds_out=out_grid, method=method)
    regridded = regridder(data, keep_attrs=True)

    if len(regridded.lon.dims) == 2:
        # New grid has 2D lat/lon, but lat is constant over x axis,
        # and lon is constant over y axis.
        # We need to convert to 1D lat/lon
        lats = regridded.lat.isel(x=0).values
        lons = regridded.lon.isel(y=0).values
        regridded = (
            regridded.drop_vars(["lat", "lon"])
            .rename_dims({"x": "lon", "y": "lat"})
            .assign_coords(lat=lats, lon=lons)
        )

    # Regridding drops rioxarray info - reattach
    regridded.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    regridded = write_rio_crs(regridded, out_crs)

    return regridded


def resample(
    data: DataArray | Dataset,
    resolution: float | tuple[float, float],
    regrid_method: XESMF_Regrid_Methods = "bilinear",
) -> DataArray | Dataset:
    """
    Resample the data to a new resolution. Returns new data; the input is
    not modified.

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
    # Optional dependency for advanced regridding
    xe = import_optional_dependency("xesmf")

    if isinstance(resolution, (int, float)):
        resolution = (resolution, resolution)

    # Calculate the new grid
    bounds = data.cf.add_bounds(["lat", "lon"])
    xmin = bounds.lon_bounds.min()
    xmax = bounds.lon_bounds.max()
    ymin = bounds.lat_bounds.min()
    ymax = bounds.lat_bounds.max()
    dx = resolution[0]
    dy = resolution[1]
    if len(data.lon.dims) == 2:
        out_grid = xe.util.grid_2d(xmin, xmax, dx, ymin, ymax, dy)
    else:
        out_grid = generate_regular_grid(
            xmin=xmin,
            xmax=xmax,
            dx=dx,
            ymin=ymin,
            ymax=ymax,
            dy=dy,
            x_label="lon",
            y_label="lat",
        )
        out_grid.lat.attrs["units"] = "degrees_north"
        out_grid.lon.attrs["units"] = "degrees_east"

    return regrid(data, out_grid=out_grid, method=regrid_method)


def round_latlon(
    data: _XarrayT,
    lat_deci: int,
    lon_deci: int,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> _XarrayT:
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
    return data.assign_coords(
        {
            lat_dim: np.round(data[lat_dim], lat_deci),
            lon_dim: np.round(data[lon_dim], lon_deci),
        }
    )


def write_rio_crs(data: _XarrayT, crs: Any) -> _XarrayT:
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

    data = data.rio.write_crs(crs).rio.write_coordinate_system(inplace=True)

    return data


# ----- MISCELLANEOUS ----- #


def bearing(lat1, lon1, lat2, lon2, deg=True, final=False):
    """
    Great-circle bearing from point 1 to point 2 on a sphere.

    Formulas from https://www.movable-type.co.uk/scripts/latlong.html.
    Vectorized: any argument can be an array (or pandas Series), broadcast
    together with numpy rules.

    Parameters
    ----------
    lat1, lon1 : float or array-like
        Start point.
    lat2, lon2 : float or array-like
        End point.
    deg : bool, default True
        Whether the *inputs* are in degrees (``False``: radians). The output
        is always in degrees.
    final : bool, default False
        Return the final bearing (the course on arrival at point 2) instead of
        the initial bearing (the course on leaving point 1). The two differ
        along a great circle unless the path follows a meridian or the equator.

    Returns
    -------
    float or np.ndarray
        Bearing in degrees clockwise from true north, in ``[0, 360)``.
        Coincident points return 0.

    Examples
    --------
    >>> round(float(bearing(40, -111, 41, -111)), 6)  # due north
    0.0
    >>> round(float(bearing(35, 45, 35, 135)), 2)  # Baghdad -> Osaka
    60.16
    >>> round(float(bearing(35, 45, 35, 135, final=True)), 2)
    119.84
    """
    if final:
        # Final bearing = reverse of the initial bearing from point 2 back to 1.
        # (Not initial + 180: that is just the reverse course at point 1.)
        return (bearing(lat2, lon2, lat1, lon1, deg=deg) + 180) % 360

    if deg:
        lat1, lon1, lat2, lon2 = (np.deg2rad(v) for v in (lat1, lon1, lat2, lon2))

    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    # arctan2 is in (-180, 180]; shift to [0, 360)
    return (np.rad2deg(np.arctan2(y, x)) + 360) % 360


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
    >>> weights = cosine_weights(ds.lat)
    >>> ds_weighted = ds.weighted(weights)
    """
    return np.cos(np.deg2rad(lats))


def earth_radius(lat: ArrayLike) -> ArrayLike:
    """
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
    """

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2 / a**2)

    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = np.deg2rad(lat)
    lat_gc = np.arctan((1 - e2) * np.tan(lat))

    # radius equation
    # see equation 3-107 in WGS84
    r = (a * (1 - e2) ** 0.5) / (1 - (e2 * np.cos(lat_gc) ** 2)) ** 0.5

    r /= 1000  # convert to km
    return r


def gridcell_area_from_latlon(
    lat: ArrayLike, lon: ArrayLike, R: float | None = None
) -> np.ndarray:
    """
    Calculate the area of each grid cell in a lat-lon grid.

    Parameters
    ----------
    lat : ArrayLike
        Latitude array
    lon : ArrayLike
        Longitude array
    R : float, optional
        Radius of earth in kilometers, by default calculated based on the latitude.

    Returns
    -------
    np.ndarray
        Grid-cell area in square-kilometers, shape (len(lat), len(lon))
    """
    lat, lon = np.asarray(lat), np.asarray(lon)
    # gridcell_area needs a Dataset (cf bounds) with cf-recognisable lat/lon
    grid = Dataset(coords={"lat": lat, "lon": lon})
    grid.lat.attrs["units"] = "degrees_north"
    grid.lon.attrs["units"] = "degrees_east"
    grid = grid.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    grid = write_rio_crs(grid, crs="EPSG:4326")

    area = gridcell_area(grid, R=R)
    return area.values


def haversine(lat1, lon1, lat2, lon2, R=6371, deg=True):
    """
    Great-circle distance between two points on a sphere (haversine formula).

    Formula from https://www.movable-type.co.uk/scripts/latlong.html.
    Vectorized: any argument can be an array (or pandas Series), broadcast
    together with numpy rules, so a fixed point against many points works.

    Parameters
    ----------
    lat1, lon1 : float or array-like
        First point.
    lat2, lon2 : float or array-like
        Second point.
    R : float, default 6371
        Sphere radius. The result is in the same units (default: mean Earth
        radius in km, so the result is in km).
    deg : bool, default True
        Whether the inputs are in degrees (``False``: radians).

    Returns
    -------
    float or np.ndarray
        Distance in the units of ``R``.
    """
    if deg:
        lat1, lon1, lat2, lon2 = (np.deg2rad(v) for v in (lat1, lon1, lat2, lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin((dlat) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((dlon) / 2) ** 2
    # Rounding can push a just outside [0, 1] for (near-)antipodal points
    a = np.clip(a, 0, 1)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = R * c
    return d


def points_along_line(
    multiline: LineString | MultiLineString,
    spacing: float,
    resolution_factor: float | None = None,
    decimals: int | None = None,
) -> list[Point]:
    """
    Generate points spaced along a line or a network of lines.

    Every pair of points is at least ``spacing`` apart (Euclidean distance, in
    the units of the coordinates), across the whole network, including between
    disconnected lines. Distances within float rounding of ``spacing`` count
    as ``spacing``, so points on a straight line come out evenly spaced.

    The algorithm works as follows:

    1. **Topology fixing**: ``union_all`` splits the lines where they cross, so
       every intersection becomes a node, then ``line_merge`` stitches simple
       paths back together.
    2. **Graph construction**: each line is segmentized into steps of at most
       ``spacing * resolution_factor``, and the vertices become the nodes of a
       graph whose edges follow the lines. Node coordinates are rounded to
       ``decimals`` places, which snaps microscopic gaps between lines together.
    3. **Point placement**: each connected component is walked breadth-first
       from an endpoint (any node for a closed loop), visiting each node once.
       The walk carries the last point placed behind it as its "origin". A node
       at least ``spacing`` from its origin gets a point, which becomes the new
       origin. A node closer than ``spacing`` to any other placed point is
       skipped, and the walk keeps going, still measuring from the old origin.
    4. **Global spacing**: placed points are kept in a grid of square cells, so
       checking a node against every placed point only looks at nearby cells.
       Runtime is close to linear in the number of graph nodes.

    Parameters
    ----------
    multiline : shapely.LineString | shapely.MultiLineString
        The line, or network of lines.
    spacing : float
        The minimum Euclidean distance between generated points.
    resolution_factor : float, optional
        Graph step size as a fraction of ``spacing``, by default 0.1. Smaller
        values make a denser graph, which follows curves more closely and
        places points closer to exactly ``spacing`` apart, but is slower.
    decimals : int, optional
        Number of decimal places node coordinates are rounded to. By default
        two places finer than the step size (``spacing * resolution_factor``),
        and at least 5.

    Returns
    -------
    list[shapely.Point]
        Points along the network, in the order they were placed, each at least
        ``spacing`` from every other.
    """
    import_optional_dependency("networkx")
    import networkx as nx
    from shapely import line_merge, segmentize, union_all

    if spacing <= 0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    if resolution_factor is None:
        resolution_factor = 0.1
    step_size = spacing * resolution_factor
    if decimals is None:
        # Rounding error at most step_size / 200, so neighbouring nodes never
        # collapse together. Never coarser than the original fixed 5 places.
        decimals = max(5, math.ceil(-math.log10(step_size)) + 2)
    # Distances within `tol` of `spacing` count as `spacing`: float error in
    # differences of rounded coordinates (e.g. 1.5e-4 - 1e-4 < 5e-5) must not
    # push a node that is exactly `spacing` away on to the next node. A
    # thousandth of the rounding precision: far above that float error, far
    # below the step between nodes.
    tol = 1e-3 * 10.0**-decimals

    # --- Topology Fixing ---
    # unary_union splits lines at intersections, creating nodes where lines cross.
    # linemerge then stitches simple paths back together where possible.
    cleaned_geom = line_merge(union_all(multiline))

    if hasattr(cleaned_geom, "geoms"):
        lines = list(cleaned_geom.geoms)
    else:
        lines = [cleaned_geom]

    # --- Build High-Res Graph ---
    G = nx.Graph()

    # Round coordinates to "snap" microscopic gaps
    def round_coord(c):
        return (round(c[0], decimals), round(c[1], decimals))

    def distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    for line in lines:
        # Segmentize to ensure we can measure distance around curves
        dense_line = segmentize(line, step_size)
        coords = list(dense_line.coords)

        for i in range(len(coords) - 1):
            u = round_coord(coords[i])
            v = round_coord(coords[i + 1])

            # Add edge with Euclidean weight
            G.add_edge(u, v, weight=distance(u, v))

    # --- Global State ---
    # Coordinates of the placed points, in order
    placed: list[tuple[float, float]] = []
    # Spatial index: grid cell -> indices into `placed`. Two points closer than
    # `spacing` are less than half a cell apart, so they are in the same or
    # adjacent cells (cells of exactly `spacing` could miss one by float error).
    cell_size = 2 * spacing
    cells: dict[tuple[int, int], list[int]] = {}

    def cell_of(c):
        return (math.floor(c[0] / cell_size), math.floor(c[1] / cell_size))

    def place(c) -> int:
        """Place a point at c; returns its index in `placed`."""
        placed.append(c)
        cells.setdefault(cell_of(c), []).append(len(placed) - 1)
        return len(placed) - 1

    def too_close(c, skip: int = -1) -> bool:
        """Whether c is closer than `spacing` to a placed point (other than `skip`)."""
        cx, cy = cell_of(c)
        for i in (cx - 1, cx, cx + 1):
            for j in (cy - 1, cy, cy + 1):
                for k in cells.get((i, j), ()):
                    if k != skip and distance(c, placed[k]) < spacing - tol:
                        return True
        return False

    # --- Process Every Component ---
    # This loop ensures we jump to the top line even if it's disconnected
    for component_nodes in nx.connected_components(G):
        subgraph = G.subgraph(component_nodes)

        # Pick a start node for this component (preferably an endpoint)
        degrees = dict(subgraph.degree())
        start_node = next(
            (n for n, d in degrees.items() if d == 1), next(iter(subgraph.nodes))
        )

        # The start might be too close to a point on another component.
        # Origin -1 means nothing placed behind us yet: place at the first
        # node that is clear of all placed points.
        start_origin = -1 if too_close(start_node) else place(start_node)

        # BFS walker over (node, index of its origin in `placed`)
        queue = deque([(start_node, start_origin)])
        visited = set()
        while queue:
            node, origin = queue.popleft()
            if node in visited:
                continue
            visited.add(node)

            # Attempt to place a point, checking against ALL placed points
            # (except the origin, which we know is far enough)
            if origin == -1 or distance(node, placed[origin]) >= spacing - tol:
                if not too_close(node, skip=origin):
                    origin = place(node)
                # Otherwise we are blocked by a neighbour: KEEP WALKING, still
                # measuring from the old origin

            # Propagate
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, origin))

    return [Point(c) for c in placed]
