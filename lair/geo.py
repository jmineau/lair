"""
Grids, maps and distances.

:class:`BaseGrid` and the functions it wraps work on gridded xarray data:
clip to a region (:func:`clip`), grid-cell areas on the WGS84 ellipsoid
(:func:`gridcell_area`), and regridding with xESMF (:func:`regrid`,
:func:`resample`; :doc:`/examples/inventory` says which method a flux needs).
:func:`plot_grid` and the tick and inset-map helpers draw maps with cartopy
(:doc:`/examples/maps`), and :func:`haversine` and :func:`bearing` measure on
the sphere. Importing the module needs the ``geo`` extra, and regridding needs
ESMF as well (see :ref:`optional-dependencies`).
"""

from __future__ import (
    annotations,
)  # keep optional-dep annotations (e.g. shapely Polygon) lazy

import copy
import math
from collections import deque
from collections.abc import Iterable
from typing import Any, Literal, Self, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from numpy.typing import ArrayLike
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
import rasterio.crs  # noqa: E402 F811
import rioxarray as rxr  # noqa: E402 F401
from cartopy.mpl.geoaxes import GeoAxes  # noqa: E402
from cartopy.mpl.ticker import (  # noqa: E402
    LatitudeFormatter,
    LatitudeLocator,
    LongitudeFormatter,
    LongitudeLocator,
)
from shapely import LineString, MultiLineString, Point, Polygon  # noqa: E402

__all__ = [
    "bbox2extent",
    "extent2bbox",
    "PC",
    "CRS",
    "dms2dd",
    "wrap_lons",
    "add_lat_ticks",
    "add_lon_ticks",
    "add_latlon_ticks",
    "add_locator_map",
    "XESMF_Regrid_Methods",
    "BaseGrid",
    "clip",
    "gridcell_area",
    "plot_grid",
    "generate_regular_grid",
    "regrid",
    "resample",
    "round_latlon",
    "write_rio_crs",
    "bearing",
    "cosine_weights",
    "earth_radius",
    "gridcell_area_from_latlon",
    "haversine",
    "points_along_line",
]

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

    See Also
    --------
    extent2bbox : The other way.

    Examples
    --------
    >>> from lair.geo import bbox2extent
    >>> bbox2extent([-112.3, 40.4, -111.7, 41.0])
    [-112.3, -111.7, 40.4, 41.0]
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

    See Also
    --------
    bbox2extent : The other way.

    Examples
    --------
    >>> from lair.geo import extent2bbox
    >>> extent2bbox([-112.3, -111.7, 40.4, 41.0])
    [-112.3, 40.4, -111.7, 41.0]
    """
    minx, maxx, miny, maxy = extent
    bbox = [minx, miny, maxx, maxy]
    return bbox


# ----- COORDINATES ----- #

#: cartopy's PlateCarree projection: plain longitude and latitude. Pass it as
#: ``transform=`` to draw lat/lon data on a map in any projection.
#:
#: .. seealso:: :doc:`/examples/maps`, which draws with it.
PC = ccrs.PlateCarree()


class CRS:
    """
    Coordinate Reference System (CRS) class.

    This class is a wrapper around the pyproj.CRS class, with additional methods
    for converting to other CRS classes.

    See https://pyproj4.github.io/pyproj/stable/crs_compatibility.html#cartopy for more information.

    .. note::
        `osgeo`, `fiona`, and `pycrs` conversions have not been implemented.

    Parameters
    ----------
    crs : Any
        The CRS: another ``CRS``, an EPSG code (``4326``), an ``"EPSG:..."``
        string, a cartopy, pyproj or rasterio CRS, or anything
        ``pyproj.CRS.from_user_input`` reads (a PROJ string, WKT).

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

    See Also
    --------
    write_rio_crs : Writes a CRS to xarray data.

    Examples
    --------
    >>> from lair.geo import CRS
    >>> CRS("EPSG:4326").epsg
    4326
    >>> CRS(4326).to_pyproj().name
    'WGS 84'
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
        """
        Get the EPSG code of the CRS.

        None for a CRS without one, such as Vulcan's Lambert conformal grid.

        See Also
        --------
        CRS.wkt : The full description of the CRS.

        Examples
        --------
        >>> from lair.geo import CRS
        >>> CRS("EPSG:4326").epsg
        4326
        """
        return self.crs.to_epsg()

    @property
    def proj4(self) -> str:
        """
        Get the PROJ4 string of the CRS.

        pyproj warns that a PROJ string can lose some of a CRS's information;
        prefer :attr:`wkt` to store one.

        See Also
        --------
        CRS.wkt : The CRS as WKT, which loses nothing.
        """
        return self.crs.to_proj4()

    @property
    def units(self) -> str:
        """
        Get the units of the CRS.

        The linear units, as rasterio names them: ``'metre'`` for a projected
        CRS in meters, ``'unknown'`` for a geographic one (degrees).

        See Also
        --------
        lair.geo.gridcell_area : Needs a geographic CRS or one in meters.

        Examples
        --------
        >>> from lair.geo import CRS
        >>> CRS("EPSG:3857").units
        'metre'
        """
        return self.to_rasterio().linear_units

    @property
    def wkt(self) -> str:
        """
        Get the WKT string of the CRS.

        See Also
        --------
        CRS.proj4 : The shorter PROJ string.

        Examples
        --------
        >>> from lair.geo import CRS
        >>> CRS("EPSG:4326").wkt[:20]
        'GEOGCRS["WGS 84",ENS'
        """
        return self.crs.to_wkt()

    def to_cartopy(self) -> ccrs.CRS:
        """
        Convert to cartopy CRS.

        A plain ``cartopy.crs.CRS``, not a ``Projection``, so it can transform
        data but cannot be a map's projection.

        See Also
        --------
        CRS.to_pyproj : The pyproj CRS it is made from.
        """
        return ccrs.CRS(self.crs)

    def to_rasterio(self) -> rasterio.CRS:
        """
        Convert to rasterio CRS.

        See Also
        --------
        CRS.to_pyproj : The pyproj CRS it is made from.
        """
        return rasterio.CRS.from_user_input(self.crs)

    def to_pyproj(self) -> pyproj.CRS:
        """
        Convert to pyproj CRS.

        See Also
        --------
        CRS.to_cartopy : For cartopy.
        CRS.to_rasterio : For rasterio.

        Examples
        --------
        >>> from lair.geo import CRS
        >>> CRS(4326).to_pyproj().is_geographic
        True
        """
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

    See Also
    --------
    wrap_lons : Puts longitudes in one range.

    Examples
    --------
    >>> from lair.geo import dms2dd
    >>> dms2dd(40, 45, 36)
    40.76
    >>> dms2dd(-111, 53, 24)
    -111.89
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
    Transform the longitude values into the half-open interval [base, base + period).

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

    See Also
    --------
    dms2dd : Degrees, minutes and seconds to degrees.

    Examples
    --------
    >>> from lair.geo import wrap_lons
    >>> wrap_lons([190.0, -181.0, 360.0])
    array([-170.,  179.,    0.])
    >>> wrap_lons(350.0, base=0)
    array([350.])
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

    See Also
    --------
    add_latlon_ticks : Both axes at once.
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

    See Also
    --------
    add_latlon_ticks : Both axes at once.
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

    See Also
    --------
    add_lat_ticks : The latitude ticks alone.
    add_lon_ticks : The longitude ticks alone.

    Examples
    --------
    Ticks about one per inch, on a map of the Salt Lake Valley
    (:doc:`/examples/maps` draws one)::

        extent = [-112.3, -111.7, 40.4, 41.0]
        ax.set_extent(extent, crs=PC)
        add_latlon_ticks(ax, extent, x_rotation=45)
    """
    xlims, ylims = extent[:2], extent[2:]

    add_lat_ticks(ax, ylims, labelsize=labelsize, more_ticks=more_lat_ticks)

    add_lon_ticks(
        ax, xlims, rotation=x_rotation, labelsize=labelsize, more_ticks=more_lon_ticks
    )

    return None


def add_locator_map(
    fig: plt.Figure,
    main_extent: list[float],
    main_extent_crs: ccrs.CRS,
    locator_rect: tuple[float, float, float, float],
    locator_extent: list[float],
    locator_crs: ccrs.CRS,
    color: str,
    linewidth: int,
    zorder: int | None = None,
) -> plt.Axes:
    """
    Add a small locator map to the figure that shows where the main map is.

    The locator map is a new axes with land, ocean and state boundaries, and
    the extent of the main map drawn on it as an outline (no fill).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to add the locator map to
    main_extent : list[float]
        Extent of the main map, ``[minx, maxx, miny, maxy]``, in
        ``main_extent_crs``
    main_extent_crs : ccrs.CRS
        CRS of the main extent
    locator_rect : tuple[left, bottom, width, height]
        Position of the locator map, in figure fractions (0 to 1) as for
        ``Figure.add_axes``
    locator_extent : list[float]
        Extent of the locator map, ``[west, east, south, north]`` in degrees
        of longitude and latitude
    locator_crs : ccrs.CRS
        Projection of the locator map
    color : str
        Color of the main-extent outline drawn on the locator map
    linewidth : int
        Line width of the main-extent outline
    zorder : int, optional
        Zorder of the locator map, by default None

    Returns
    -------
    plt.Axes
        Axes object of the locator map

    See Also
    --------
    plot_grid : Draws a grid's cells on the main map.

    Examples
    --------
    A locator map of Utah and its neighbors in the top left of ``fig``, with
    the main map's extent outlined in red, as :doc:`/examples/maps` draws it::

        add_locator_map(
            fig,
            main_extent=[-112.2, -111.5, 40.4, 41.0],
            main_extent_crs=PC,
            locator_rect=(0.14, 0.58, 0.24, 0.26),
            locator_extent=[-117, -106, 35, 43],
            locator_crs=PC,
            color="red",
            linewidth=1.5,
        )
    """
    import cartopy.feature as cfeature
    from shapely.geometry import box

    locator_ax: GeoAxes = fig.add_axes(
        locator_rect, projection=locator_crs, zorder=zorder
    )
    locator_ax.set_extent(locator_extent)

    locator_ax.add_feature(cfeature.LAND)
    locator_ax.add_feature(cfeature.OCEAN)
    locator_ax.add_feature(cfeature.STATES)

    main_poly = box(*extent2bbox(main_extent))

    # Outline only: `color=` would also fill the box
    locator_ax.add_geometries(
        [main_poly],
        crs=main_extent_crs,
        facecolor="none",
        edgecolor=color,
        linewidth=linewidth,
    )

    return locator_ax


# ----- XARRAY UTILITIES ----- #

XESMF_Regrid_Methods = Literal[
    "bilinear",
    "conservative",
    "conservative_normed",
    "nearest_s2d",
    "nearest_d2s",
    "patch",
]

#: Name of the variable a DataArray is stored under while it is a temporary Dataset
_TMP_VAR = "__lair_data__"


def _as_dataset(data: DataArray | Dataset) -> Dataset:
    """
    Wrap a DataArray in a one-variable Dataset; a Dataset is returned as is.

    cf_xarray's bounds helpers (``cf.add_bounds``/``cf.get_bounds``, also used
    by xesmf) only exist on Datasets. Undo with :func:`_like_input`.
    """
    if isinstance(data, DataArray):
        return data.to_dataset(name=_TMP_VAR)
    return data


def _like_input(result: Dataset, data: _XarrayT) -> _XarrayT:
    """Undo :func:`_as_dataset`: a DataArray input gets a DataArray back, with its name."""
    if isinstance(data, DataArray):
        out = result[_TMP_VAR]
        out.name = data.name
        return cast(_XarrayT, out)
    return cast(_XarrayT, result)


class BaseGrid:
    """
    Base class for working with gridded data.

    This class is a wrapper around xarray DataArray and Dataset objects, with additional methods
    for clipping, regridding, resampling, and reprojection. Operations return a new
    grid by default; pass ``inplace=True`` to modify this one instead. Either way the
    grid object is returned, for chaining.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The gridded data, with its spatial dimensions set for rioxarray
        (``lat``/``lon``, or ``x``/``y``).
    crs : Any
        The CRS of the data, anything :class:`CRS` takes. It is written to the
        data and kept as the ``crs`` attribute.

    See Also
    --------
    lair.inventories.Inventory : The grid of an emission inventory.

    Examples
    --------
    >>> from lair.geo import BaseGrid, generate_regular_grid
    >>> data = generate_regular_grid(
    ...     -112.2, -111.8, 0.1, 40.6, 41.0, 0.1, x_label="lon", y_label="lat"
    ... ).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    >>> grid = BaseGrid(data, "EPSG:4326")
    >>> grid.crs.epsg
    4326
    >>> box = grid.clip(bbox=(-112.1, 40.7, -111.9, 40.9))
    >>> dict(box.data.sizes)
    {'lat': 2, 'lon': 2}
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
            The copied grid: a deep copy, data included.

        See Also
        --------
        BaseGrid.clip : Returns a changed copy unless ``inplace=True``.
        """
        return copy.deepcopy(self)

    @property
    def gridcell_area(self) -> DataArray:
        """
        Calculate the grid cell area in km^2.

        Returns
        -------
        xr.DataArray
            The grid cell area, on the WGS84 ellipsoid for a lat-lon grid.

        See Also
        --------
        gridcell_area : How each cell's area is computed.

        Examples
        --------
        Two rows of 0.1 degree cells near Salt Lake City:

        >>> from lair.geo import BaseGrid, generate_regular_grid
        >>> data = generate_regular_grid(
        ...     -112.0, -111.8, 0.1, 40.6, 40.8, 0.1, x_label="lon", y_label="lat"
        ... ).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        >>> BaseGrid(data, "EPSG:4326").gridcell_area.values.round(3)
        array([[93.922, 93.922],
               [93.783, 93.783]])
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

        See Also
        --------
        clip : The same for xarray data.
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

        See Also
        --------
        regrid : The same for xarray data.
        BaseGrid.resample : To a new resolution over the same area.
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

        A resolution that does not divide the grid's width leaves a strip at the
        far edge; :func:`lair.geo.resample` says what happens to it.

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

        See Also
        --------
        resample : The same for xarray data, and the methods.
        lair.inventories.Inventory.resample : Conservative by default, for a flux.
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
            The reprojected grid, on EPSG:4326 lat/lon.

        Raises
        ------
        ValueError
            If the data is already on a geographic (lat/lon) CRS.

        See Also
        --------
        BaseGrid.resample : For data already on lat/lon.
        """
        # Any geographic CRS (EPSG:4326, OGC:CRS84, NAD83, ...) is already lat/lon
        if self.crs.to_pyproj().is_geographic:
            raise ValueError("Data is already in lat lon")

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

    Raises
    ------
    ValueError
        Unless exactly one of ``bbox``, ``extent`` and ``geom`` is given.

    See Also
    --------
    BaseGrid.clip : The same for a grid or an inventory.
    bbox2extent : Converts between a bbox and an extent.

    Examples
    --------
    A box of 2 by 2 cells out of 4 by 4, by bbox or by extent:

    >>> from lair.geo import clip, generate_regular_grid, write_rio_crs
    >>> grid = generate_regular_grid(
    ...     -112.2, -111.8, 0.1, 40.6, 41.0, 0.1, x_label="lon", y_label="lat"
    ... )
    >>> grid = write_rio_crs(grid.rio.set_spatial_dims(x_dim="lon", y_dim="lat"), 4326)
    >>> dict(clip(grid, bbox=(-112.1, 40.7, -111.9, 40.9)).sizes)
    {'lat': 2, 'lon': 2}
    >>> dict(clip(grid, extent=(-112.1, -111.9, 40.7, 40.9)).sizes)
    {'lat': 2, 'lon': 2}
    """
    if (bbox is not None) + (extent is not None) + (geom is not None) != 1:
        raise ValueError("Exactly one of bbox, extent, or geom must be provided.")

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


#: WGS84 semi-major axis [km] and flattening (NGA TR8350.2, table 3.1)
_WGS84_A = 6378.137
_WGS84_F = 1 / 298.257223563


def _authalic_q(lat: ArrayLike) -> np.ndarray:
    """
    Return ``q`` of the authalic latitude on the WGS84 ellipsoid.

    The area of the ellipsoid between the equator and latitude ``lat`` (degrees)
    over a longitude span ``dlon`` (radians) is ``a**2 / 2 * dlon * q(lat)``
    (Snyder 1987, *Map Projections: A Working Manual*, eq. 3-12).
    """
    e2 = _WGS84_F * (2 - _WGS84_F)
    e = np.sqrt(e2)
    s = np.sin(np.deg2rad(lat))
    return (1 - e2) * (s / (1 - e2 * s**2) + np.arctanh(e * s) / e)


def gridcell_area(
    grid: DataArray | Dataset, R: float | np.ndarray | DataArray | None = None
) -> DataArray:
    """
    Calculate the area of each grid cell in a grid.

    On a lat-lon grid each cell is bounded by two meridians and two parallels,
    halfway between the cell centers, and its area is the exact area of that
    cell on the WGS84 ellipsoid: ``a**2 / 2 * dlon * (q(lat2) - q(lat1))``,
    with ``q`` the function of the authalic latitude (Snyder 1987, eq. 3-12).
    With ``R``, it is the area on a sphere of that radius instead,
    ``R**2 * dlon * (sin(lat2) - sin(lat1))``. On a projected grid in meters
    it is ``dx * dy``.

    Parameters
    ----------
    grid : xr.DataArray | xr.Dataset
        Grid data with a `rioxarray` CRS: geographic (any datum, e.g.
        EPSG:4326, OGC:CRS84, EPSG:4269) with 1-D ``lat``/``lon``
        coordinates, or projected in meters.
    R : float | np.ndarray | xr.DataArray, optional
        Radius of a sphere in kilometers to use instead of the WGS84
        ellipsoid (lat-lon grids only), by default None. An array must
        broadcast against the grid, e.g. a DataArray along ``lat``.

    Returns
    -------
    xr.DataArray
        grid-cell area in square-kilometers, with the grid's (y, x) dims. Its
        only attribute is ``units`` (``'km2'``), so an array multiplied by it
        (``Inventory.absolute_emissions``) takes no description of the grid.

    Raises
    ------
    ValueError
        If the grid has no CRS, its CRS is neither geographic nor in meters, or
        a lat-lon grid's ``lat`` or ``lon`` is not 1-D.

    Notes
    -----
    A sphere is close but not exact. With ``R=6371`` a 1-degree cell is 0.45%
    too large at the equator, 0.12% too small at 40.5 N and 0.9% too small
    next to a pole; a 0.1-degree cell centered at 40.75 N is 93.668 km2
    against 93.783 km2 on the ellipsoid. A sphere whose radius is the
    ellipsoid's radius at the cell's latitude (:func:`earth_radius`) is worse
    at 40.75 N (93.611 km2, 0.18% too small).

    See Also
    --------
    gridcell_area_from_latlon : The same from latitudes and longitudes.
    BaseGrid.gridcell_area : The areas of a grid's cells.

    Examples
    --------
    >>> from lair import geo
    >>> area = geo.gridcell_area_from_latlon([40.65, 40.75, 40.85], [-111.95, -111.85])
    >>> round(float(area[1, 1]), 3)  # the 0.1-degree cell at 40.75 N, 111.85 W
    93.783
    """
    crs = grid.rio.crs
    if crs is None:
        raise ValueError("grid has no CRS; set one with write_rio_crs(grid, crs).")

    # Only the coordinates matter, and cf bounds need a Dataset
    grid = _as_dataset(grid)

    if crs.is_geographic:
        if grid["lat"].ndim != 1 or grid["lon"].ndim != 1:
            raise ValueError(
                "gridcell_area needs 1-D lat and lon coordinates on a lat-lon grid."
            )
        bounds = grid.cf.add_bounds(["lat", "lon"])
        # Each cell's edges, halfway between the centers (a pole stays a pole)
        lat_b = bounds.lat_bounds.clip(-90, 90)
        lat_lo = lat_b.isel(bounds=0, drop=True)
        lat_hi = lat_b.isel(bounds=1, drop=True)
        dlon = np.deg2rad(
            abs(bounds.lon_bounds.diff("bounds").squeeze("bounds", drop=True))
        )
        # abs: north-up rasters store lat descending. The latitude term comes
        # first, so the result is (y, x) like the data.
        if R is None:
            dq = abs(_authalic_q(lat_hi) - _authalic_q(lat_lo))
            area = _WGS84_A**2 / 2 * dq * dlon
        else:
            dsin = abs(np.sin(np.deg2rad(lat_hi)) - np.sin(np.deg2rad(lat_lo)))
            area = R**2 * dsin * dlon
        area = area.drop_vars(["lat_bounds", "lon_bounds"], errors="ignore")
    elif crs.linear_units == "metre":
        bounds = grid.cf.add_bounds(["x", "y"])
        dx = bounds.x_bounds.diff("bounds").squeeze("bounds", drop=True)
        dy = bounds.y_bounds.diff("bounds").squeeze("bounds", drop=True)
        # abs: north-up rasters store y descending (negative dy).
        # dy first so the result is (y, x) like the data.
        cell_area_m2 = abs(dy * dx)
        area = cell_area_m2.pint.quantify("m2").pint.to("km2").pint.dequantify()
    else:
        raise ValueError("Only lat-lon and meter grids are supported.")
    # The area is multiplied into data variables (Inventory.absolute_emissions),
    # which keep its attributes: keep only the units (issue #136).
    area.attrs = {"units": "km2"}
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

    See Also
    --------
    add_latlon_ticks : Tick labels in degrees for the same map.

    Examples
    --------
    The cells of an inventory's grid, outlined over its map, as
    :doc:`/examples/maps` draws them::

        ax = slv.plot(time="2022-01-01")
        plot_grid(slv.data, ax=ax, add_colorbar=False)
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

    Cells start at ``xmin``/``ymin`` and continue while their center is below
    ``xmax``/``ymax`` (a center landing exactly on the max edge may or may not
    be included, depending on float error). Centers are
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

    See Also
    --------
    regrid : Regrids data to such a grid.

    Examples
    --------
    Cells of 0.1 degree from 112.2 W to 111.8 W and 40.6 N to 41.0 N:

    >>> from lair.geo import generate_regular_grid
    >>> grid = generate_regular_grid(
    ...     -112.2, -111.8, 0.1, 40.6, 41.0, 0.1, x_label="lon", y_label="lat"
    ... )
    >>> grid.lon.values
    array([-112.15, -112.05, -111.95, -111.85])
    >>> grid.lat.values
    array([40.65, 40.75, 40.85, 40.95])
    """
    x = _cell_centers(xmin, xmax, dx)
    y = _cell_centers(ymin, ymax, dy)
    zeros = np.zeros((len(y), len(x)))
    grid = DataArray(zeros, coords={y_label: y, x_label: x})
    if chunks is not None:
        grid = grid.chunk(chunks)
    return grid


def _cell_centers(vmin: float, vmax: float, d: float) -> np.ndarray:
    """Centers of the cells of size ``d`` from ``vmin`` whose center is below ``vmax``."""
    vmin, vmax, d = float(vmin), float(vmax), float(d)
    # Cell count exactly as before (np.arange). Where the last center lands on
    # vmax, float error decides whether that cell is included.
    n = len(np.arange(vmin + d / 2, vmax, d))
    # Each center from vmin directly (no accumulated error), then drop float noise
    return np.round(vmin + d * (np.arange(n) + 0.5), 10)


def _end_of_last_cell(vmin: float, vmax: float, d: float) -> float:
    """
    End of the last cell of size ``d`` that starts below ``vmax``.

    Cells start at ``vmin``. When ``d`` does not divide ``vmax - vmin`` the last
    cell reaches past ``vmax``, so the cells cover everything up to ``vmax``.
    """
    vmin, vmax, d = float(vmin), float(vmax), float(d)
    # A strip under a millionth of a cell is float noise in the bounds, not a cell
    n = int(np.ceil((vmax - vmin) / d - 1e-6))
    return vmin + n * d


def regrid(
    data: _XarrayT,
    out_grid: DataArray | Dataset,
    method: XESMF_Regrid_Methods = "bilinear",
) -> _XarrayT:
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
        The regridded data, the same type as ``data``.

    See Also
    --------
    resample : To a new resolution over the same area.
    generate_regular_grid : Makes a grid to regrid to.

    Examples
    --------
    A flux of 1 on four cells of 0.1 degree, to cells of 0.05 degree; a
    conservative regrid keeps the flux:

    >>> import numpy as np
    >>> import xarray as xr
    >>> from lair.geo import generate_regular_grid, regrid
    >>> flux = xr.DataArray(
    ...     np.ones((2, 2)),
    ...     coords={"lat": [40.65, 40.75], "lon": [-111.95, -111.85]},
    ...     dims=("lat", "lon"),
    ... )
    >>> fine = generate_regular_grid(
    ...     -112.0, -111.8, 0.05, 40.6, 40.8, 0.05, x_label="lon", y_label="lat"
    ... )
    >>> out = regrid(flux, fine, method="conservative")
    >>> out.sizes["lat"], out.sizes["lon"]
    (4, 4)
    >>> float(out.min().round(3)), float(out.max().round(3))
    (1.0, 1.0)
    """
    # Optional dependency for advanced regridding
    xe = import_optional_dependency("xesmf")

    out_crs = "EPSG:4326"

    # Use cf-xarray to calculate the bounds of the grid cells (Dataset only;
    # a DataArray is turned back into one at the end)
    ds = _as_dataset(data).cf.add_bounds(["lat", "lon"])

    # Regrid the data using `xesmf`
    regridder = xe.Regridder(ds_in=ds, ds_out=out_grid, method=method)
    regridded = regridder(ds, keep_attrs=True)

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

    return _like_input(regridded, data)


def resample(
    data: _XarrayT,
    resolution: float | tuple[float, float],
    regrid_method: XESMF_Regrid_Methods = "bilinear",
) -> _XarrayT:
    """
    Resample the data to a new resolution.

    Returns new data; the input is not modified.

    The new grid starts at the source's outer edge. When the resolution does not
    divide the source's width (0.3 degrees over 4.0 degrees of longitude), the
    last cell is partial, and what becomes of it depends on the method:

    - ``"conservative"`` keeps it whole, so the new grid covers the whole source
      and its last cell reaches past the source's edge. That cell holds the flux of
      the part inside the source averaged over the whole cell, the part outside
      counting as zero, so flux times area still adds up to the source's total.
    - Every other method (``"bilinear"``, the default, and ``"conservative_normed"``
      included) keeps it only when its center is inside the source, which takes
      more than half of it, and otherwise drops the strip. A cell centered outside
      the source's outermost cell centers gets zero from bilinear interpolation,
      not data, and ``"conservative_normed"`` would give it the flux of the part
      inside but count its whole area.

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
        The resampled data, the same type as ``data``.

    See Also
    --------
    regrid : To a grid you give.
    lair.inventories.Inventory.resample : Conservative by default, for a flux.

    Examples
    --------
    A flux of 1 on four cells of 0.1 degree, to cells of 0.05 degree. The
    default, bilinear, leaves the outer cells at zero (they are outside the
    outermost centers); a flux needs ``"conservative"``:

    >>> import numpy as np
    >>> import xarray as xr
    >>> from lair.geo import resample, write_rio_crs
    >>> flux = xr.DataArray(
    ...     np.ones((2, 2)),
    ...     coords={"lat": [40.65, 40.75], "lon": [-111.95, -111.85]},
    ...     dims=("lat", "lon"),
    ... )
    >>> flux = write_rio_crs(flux.rio.set_spatial_dims(x_dim="lon", y_dim="lat"), 4326)
    >>> resample(flux, 0.05).values
    array([[0., 0., 0., 0.],
           [0., 1., 1., 0.],
           [0., 1., 1., 0.],
           [0., 0., 0., 0.]])
    >>> conservative = resample(flux, 0.05, regrid_method="conservative")
    >>> float(conservative.min().round(3)), float(conservative.max().round(3))
    (1.0, 1.0)
    """
    # Optional dependency for advanced regridding
    xe = import_optional_dependency("xesmf")

    if isinstance(resolution, (int, float)):
        resolution = (resolution, resolution)

    # Calculate the new grid. A conservative regrid must cover the whole source or
    # the strip left over by a resolution that does not divide it is lost
    cover = regrid_method == "conservative"
    bounds = _as_dataset(data).cf.add_bounds(["lat", "lon"])
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
            xmax=_end_of_last_cell(xmin, xmax, dx) if cover else xmax,
            dx=dx,
            ymin=ymin,
            ymax=_end_of_last_cell(ymin, ymax, dy) if cover else ymax,
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

    See Also
    --------
    wrap_lons : Puts longitudes in one range.

    Examples
    --------
    Float noise in a file's coordinates, rounded away so two grids align:

    >>> import xarray as xr
    >>> from lair.geo import round_latlon
    >>> data = xr.DataArray(
    ...     [[1.0]],
    ...     coords={"lat": [40.749999999], "lon": [-111.850000001]},
    ...     dims=("lat", "lon"),
    ... )
    >>> rounded = round_latlon(data, 2, 2)
    >>> float(rounded.lat[0]), float(rounded.lon[0])
    (40.75, -111.85)
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

    See Also
    --------
    CRS : The CRS it takes.
    gridcell_area : Needs the CRS it writes.

    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> from lair.geo import write_rio_crs
    >>> data = xr.DataArray(
    ...     np.zeros((2, 2)),
    ...     coords={"lat": [40.65, 40.75], "lon": [-111.95, -111.85]},
    ...     dims=("lat", "lon"),
    ... ).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    >>> write_rio_crs(data, "EPSG:4326").rio.crs.to_epsg()
    4326
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

    See Also
    --------
    haversine : The distance between the same points.

    Examples
    --------
    >>> from lair.geo import bearing
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

    See Also
    --------
    gridcell_area : The exact area of each cell, for a grid.

    Examples
    --------
    The mean of a field weighted by the area of each latitude band:

    >>> import xarray as xr
    >>> from lair.geo import cosine_weights
    >>> field = xr.DataArray([1.0, 3.0], coords={"lat": [0.0, 60.0]}, dims="lat")
    >>> weights = cosine_weights(field.lat)
    >>> weights.values.round(3)
    array([1. , 0.5])
    >>> round(float(field.weighted(weights).mean()), 3)
    1.667
    """
    return np.cos(np.deg2rad(lats))


def earth_radius(lat: ArrayLike) -> ArrayLike:
    """
    Calculate radius of Earth assuming oblate spheroid defined by WGS84

    The radius is the distance from the Earth's center to the ellipsoid at each
    latitude. It is not the radius of a sphere with the ellipsoid's areas:
    :func:`gridcell_area` computes those on the ellipsoid itself.

    Parameters
    ----------
    lat : array-like
        latitudes in degrees

    Returns
    -------
    array-like
        vector of radius in kilometers. For a ``DataArray`` it is named
        ``earth_radius`` and its only attributes are its own, ``long_name``
        and ``units`` (``km``), not the latitude's.

    Notes
    -----
     - Originally copied from https://towardsdatascience.com/the-correct-way-to-average-the-globe-92ceecd172b7
     - WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf

    See Also
    --------
    gridcell_area : Cell areas on the ellipsoid, not on a sphere of this radius.

    Examples
    --------
    >>> import xarray as xr
    >>> from lair import geo
    >>> lat = xr.DataArray([0.0, 90.0], dims="lat", attrs={"units": "degrees_north"})
    >>> radius = geo.earth_radius(lat)
    >>> radius.values.round(3)
    array([6378.137, 6356.752])
    >>> radius.attrs
    {'long_name': 'Earth radius', 'units': 'km'}
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
    if isinstance(r, DataArray):
        # The ufuncs above kept the latitude's attributes (units: degrees_north,
        # axis: 'Y', ...) and name: label the radius as what it is (issue #139).
        r.attrs = {"long_name": "Earth radius", "units": "km"}
        r.name = "earth_radius"
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
        Radius of a sphere in kilometers to use instead of the WGS84
        ellipsoid, by default None (see :func:`gridcell_area`).

    Returns
    -------
    np.ndarray
        Grid-cell area in square-kilometers, shape (len(lat), len(lon))

    See Also
    --------
    gridcell_area : The same for gridded xarray data.

    Examples
    --------
    >>> from lair.geo import gridcell_area_from_latlon
    >>> gridcell_area_from_latlon([40.65, 40.75, 40.85], [-111.95, -111.85]).round(3)
    array([[93.922, 93.922],
           [93.783, 93.783],
           [93.644, 93.644]])
    """
    lat, lon = np.asarray(lat), np.asarray(lon)
    # A coordinate-only grid with cf-recognizable lat/lon
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

    See Also
    --------
    bearing : The direction from one point to the other.

    Examples
    --------
    From Salt Lake City to Provo and to Ogden, in km:

    >>> import numpy as np
    >>> from lair.geo import haversine
    >>> round(float(haversine(40.7608, -111.8910, 40.2338, -111.6585)), 1)
    61.8
    >>> haversine(
    ...     40.7608,
    ...     -111.8910,
    ...     np.array([40.2338, 41.2230]),
    ...     np.array([-111.6585, -111.9738]),
    ... ).round(1)
    array([61.8, 51.9])
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

    Raises
    ------
    ValueError
        If ``spacing`` is not positive.

    See Also
    --------
    haversine : A distance in km, where this one is in the coordinates' units.

    Examples
    --------
    >>> from shapely import LineString
    >>> from lair.geo import points_along_line
    >>> points = points_along_line(LineString([(0, 0), (10, 0)]), 2.5)
    >>> [(p.x, p.y) for p in points]
    [(0.0, 0.0), (2.5, 0.0), (5.0, 0.0), (7.5, 0.0), (10.0, 0.0)]
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
        # Rounding error at most step_size / 200, so neighboring nodes never
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
            far_enough = origin == -1 or distance(node, placed[origin]) >= spacing - tol
            if far_enough and not too_close(node, skip=origin):
                origin = place(node)
            # Otherwise we are too near the origin or blocked by a neighbor:
            # KEEP WALKING, still measuring from the old origin

            # Propagate
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, origin))

    return [Point(c) for c in placed]
