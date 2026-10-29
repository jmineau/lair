"""
Winds and polar binning.

:func:`wind_components` and :func:`wind_direction` convert between a wind's
speed and direction (the direction it blows from, in degrees clockwise from
north) and its u and v components; :func:`rotate_winds` turns HRRR's
grid-relative components to earth-relative ones. :func:`bin_polar` and its
helpers bin data by wind direction into the 16 compass sectors of
:data:`WIND_SECTORS`, for the polar plots of :mod:`lair.plotter`.
"""

from typing import Any

import numpy as np
import pandas as pd

# %% Polar

#: The 16 compass sectors used by :func:`bin_polar`, in degrees clockwise from N:
#: ``WIND_SECTORS["ESE"]`` is 112.5. :func:`sector_radians` gives them in radians.
WIND_SECTORS = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}


def bin_polar(
    data: pd.DataFrame, x: str = "ws", wd: str = "wd", xbins: int = 30
) -> pd.DataFrame:
    """
    Bin data into polar coordinates.

    Parameters
    ----------
    data : pd.DataFrame
        Data to bin.
    x : str, optional
        Variable to bin. Default is 'ws'.
    wd : str, optional
        Wind direction column name. Default is 'wd'.
    xbins : int | list, optional
        Bin edges for x. An int is the number of evenly spaced edges from the
        minimum to the maximum of x (so ``xbins - 1`` bins); a list gives the
        edges explicitly. Default is 30.

    Returns
    -------
    pd.DataFrame
        A copy of ``data`` with binned wind direction and speed added as
        ``wd_bin`` (compass sector), ``radian_bin`` and ``x_bin``. The
        caller's frame is not modified.

    See Also
    --------
    WIND_SECTORS : The sectors.
    sector_radians : The sectors in radians, the ``radian_bin`` values.
    circularize_radial_data : Closes the circle of the aggregated bins for a plot.
    lair.plotter.polarPlot : Draws the binned data.

    Examples
    --------
    A wind at 350 and one at 10 degrees are both in the N sector; the speed is
    put in the bin whose upper edge is given:

    >>> import pandas as pd
    >>> from lair.air import bin_polar
    >>> df = pd.DataFrame({"ws": [1.0, 2.5, 4.0], "wd": [350.0, 10.0, 95.0]})
    >>> binned = bin_polar(df, xbins=[0, 2, 4])
    >>> binned[["wd_bin", "x_bin"]].astype(str)
      wd_bin x_bin
    0      N     2
    1      N     4
    2      E     4
    """
    data = data.copy()
    directions = WIND_SECTORS

    wd_bins = np.linspace(0, 360, 17) + 11.25
    wd_bins = np.insert(wd_bins, 0, -0.1)
    # The last interval (348.75-360) wraps round to N. pd.cut only accepts a
    # repeated label for an unordered result, so restore the compass order after.
    labels = [*directions, "N"]
    # pandas-stubs has no pd.cut overload for these argument types
    wd_bin = pd.cut(data[wd], wd_bins, labels=labels, ordered=False)  # pyrefly: ignore[no-matching-overload]
    data["wd_bin"] = wd_bin.cat.reorder_categories(list(directions), ordered=True)

    # Missing (or out of range) directions have no sector and map to NaN
    degrees = data["wd_bin"].astype(object).map(directions).astype(float)
    data["radian_bin"] = np.deg2rad(degrees)

    # Bin x (speed); Series.min/max skip NaN, the builtins do not
    if isinstance(xbins, int):
        x_bins = np.linspace(data[x].min(), data[x].max(), xbins)
    elif isinstance(xbins, (list, tuple, range)):
        x_bins = xbins
    else:
        raise ValueError("Invalid xbins")
    # pyrefly: ignore[no-matching-overload]
    data["x_bin"] = pd.cut(
        data[x],
        x_bins,
        labels=x_bins[1:],
        include_lowest=True,
    )

    return data


def sector_radians() -> np.ndarray:
    """
    Radian values of the 16 direction sectors (``radian_bin`` of :func:`bin_polar`).

    Returns
    -------
    np.ndarray
        N, NNE, ..., NNW in radians clockwise from N.

    See Also
    --------
    WIND_SECTORS : The same sectors in degrees.
    bin_polar : Puts each wind in one of them.

    Examples
    --------
    >>> from lair.air import sector_radians
    >>> sector_radians().round(3)[:4]
    array([0.   , 0.393, 0.785, 1.178])
    """
    return np.deg2rad(np.array(list(WIND_SECTORS.values())))


def circularize_radial_data(agg: pd.DataFrame):
    """
    Circularize radial data for polar plots.

    The rows must be evenly spaced in theta and cover the full circle (e.g.
    all 16 sectors of :func:`sector_radians`); the first row is repeated one
    step past the last to close the circle.

    Parameters
    ----------
    agg : pd.DataFrame
        Aggregated data. Index is theta, columns are r.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Theta, r, and c for polar

    See Also
    --------
    bin_polar : Bins the data the aggregate comes from.
    sector_radians : The theta values of the 16 sectors.
    lair.plotter.polarPlot : Does this for a contour plot.

    Examples
    --------
    One value per sector (rows) and per speed (columns): the result has one more
    row, a copy of the first, so the plotted contour closes:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from lair.air import circularize_radial_data, sector_radians
    >>> agg = pd.DataFrame(
    ...     np.arange(32).reshape(16, 2), index=sector_radians(), columns=[2.0, 4.0]
    ... )
    >>> theta, r, c = circularize_radial_data(agg)
    >>> c.shape
    (17, 2)
    >>> bool((c[-1] == c[0]).all())
    True
    """
    theta = agg.index.values
    dtheta = np.diff(theta).mean()
    theta = np.concatenate((theta, theta[-1:] + dtheta))

    r = agg.columns.values
    r, theta = np.meshgrid(r, theta)

    c = agg.values
    c = np.concatenate((c, c[0:1, :]), axis=0)

    return theta, r, c


# %% Wind


def wind_components(speed, wind_direction) -> tuple[Any, Any]:
    """
    Calculate the u and v components of the wind.

    Parameters
    ----------
    speed : float
        Wind speed in m/s.

    wind_direction : float
        Wind direction in degrees clockwise from North (specified as blowing from).

    Returns
    -------
    tuple[float, float]
        u and v components of the wind.

    See Also
    --------
    wind_direction : The direction from the components.

    Examples
    --------
    A west wind (blowing from 270 degrees) at 10 m/s blows toward the east,
    so u is positive:

    >>> from lair.air import wind_components
    >>> u, v = wind_components(10.0, 270.0)
    >>> round(float(u), 6), round(float(v), 6)
    (10.0, 0.0)
    """
    u = -speed * np.sin(np.deg2rad(wind_direction))
    v = -speed * np.cos(np.deg2rad(wind_direction))

    return u, v


def wind_direction(u, v):
    """
    Convert u and v components to wind direction.

    Parameters
    ----------
    u : np.array
        u component of wind
    v : np.array
        v component of wind

    Returns
    -------
    np.array
        wind direction in degrees (blowing from), NaN where u = v = 0: a zero
        vector has no direction, and arctan2(0, 0) = 0 would report 270 (west).
        This matters for vector-mean winds, where a dead-calm period averages
        to exactly zero.

    See Also
    --------
    wind_components : The components from a speed and a direction.

    Examples
    --------
    >>> from lair.air import wind_direction
    >>> float(wind_direction(10.0, 0.0))  # blowing toward the east
    270.0
    >>> float(wind_direction(0.0, 0.0))  # calm
    nan
    """
    direction = (270 - np.rad2deg(np.arctan2(v, u))) % 360
    calm = (np.asarray(u) == 0) & (np.asarray(v) == 0)
    # multiply rather than index, so pandas/xarray inputs keep their labels
    return direction * np.where(calm, np.nan, 1.0)


def rotate_winds(u, v, lon) -> tuple[Any, Any]:
    """
    Rotate HRRR winds from grid to earth coordinates.

    .. note::
        Based on https://rapidrefresh.noaa.gov/faq/HRRR.faq.html
        Modified from https://gist.github.com/fischcheng/411d0bafe7762e6b5d7b1233b625a2bb

    Parameters
    ----------
    u : np.array
        u component of wind
    v : np.array
        v component of wind
    lon : float | np.array
        longitude of point, in -180..180 or 0..360

    Returns
    -------
    tuple[np.array, np.array]
        u and v components of wind in earth coordinates

    See Also
    --------
    lair.hrrr.winds_frame : HRRR winds at a point, rotated with this.
    """
    # Parameters
    rotcon_p = 0.622515
    lon_xx_p = -97.5
    # lat_tan_p  =  25.0 (np.sin(lat_tan_p/180*np.pi)) to get rotcon_p

    # Calc right grid_angle. The offset from the reference longitude must be
    # in -180..180, whichever convention lon uses (e.g. 248 vs -112).
    dlon = (lon - lon_xx_p + 180) % 360 - 180
    angle2 = rotcon_p * dlon * 0.017453  # convert to radian
    sinx2 = np.sin(angle2)
    cosx2 = np.cos(angle2)

    # Wind rotation
    u_out = cosx2 * u + sinx2 * v
    v_out = -sinx2 * u + cosx2 * v

    return u_out, v_out
