"""
Miscellaneous functions for atmospheric data.
"""

import numpy as np
import pandas as pd
from typing import Any

# %% Polar

def bin_polar(data: pd.DataFrame, x: str='ws', wd: str='wd', xbins: int=30
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
    xbins : int, optional
        Number of bins for x. Default is 30.

    Returns
    -------
    pd.DataFrame
        Data with binned wind direction and speed.
    """
    directions = {
        "N":   0,
        "NNE": 22.5,
        "NE":  45,
        "ENE": 67.5,
        "E":   90,
        "ESE": 112.5,
        "SE":  135,
        "SSE": 157.5,
        "S":   180,
        "SSW": 202.5,
        "SW":  225,
        "WSW": 247.5,
        "W":   270,
        "WNW": 292.5,
        "NW":  315,
        "NNW": 337.5,
        "N2":  0}

    wd_bins = np.linspace(0, 360, 17) + 11.25
    wd_bins = np.insert(wd_bins, 0, -0.1)
    data['wd_bin'] = pd.cut(data[wd], wd_bins, labels=directions.keys())
    data['wd_bin'] = data['wd_bin'].replace('N2', 'N')

    def direction_to_radians(direction):
        degrees = directions.get(direction.upper())
        if degrees is None:
            raise ValueError("Invalid direction")
        radians = np.deg2rad(degrees)
        return float(radians)

    data['radian_bin'] = data.wd_bin.apply(direction_to_radians).astype(float)

    # Bin x (speed)
    if isinstance(xbins, int):
        x_bins = np.linspace(min(data[x]), max(data[x]), xbins)
    elif isinstance(xbins, (list, tuple, range)):
        x_bins = xbins
    else:
        raise ValueError('Invalid xbins')
    data['x_bin'] = pd.cut(data[x], x_bins, labels=x_bins[1:],
                           include_lowest=True)

    return data


def circularize_radial_data(agg: pd.DataFrame):
    """
    Circularize radial data for polar plots.

    Parameters
    ----------
    agg : pd.DataFrame
        Aggregated data. Index is theta, columns are r.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Theta, r, and c for polar
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

def wind_components(speed, wind_direction)-> tuple[Any, Any]:
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
    """
    u = - speed * np.sin(np.deg2rad(wind_direction))
    v = - speed * np.cos(np.deg2rad(wind_direction))
    
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
        wind direction in degrees
    """
    return (270 - np.rad2deg(np.arctan2(v, u))) % 360


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
    lon : float
        longitude of point

    Returns
    -------
    tuple[np.array, np.array]
        u and v components of wind in earth coordinates
    """

    # Parameters
    rotcon_p = 0.622515
    lon_xx_p = -97.5
    # lat_tan_p  =  25.0 (np.sin(lat_tan_p/180*np.pi)) to get rotcon_p

    # Calc right grid_angle
    angle2 = rotcon_p * (lon - lon_xx_p) * 0.017453  # convert to radian
    sinx2 = np.sin(angle2)
    cosx2 = np.cos(angle2)

    # Wind rotation
    u_out = cosx2 * u + sinx2 * v
    v_out = -sinx2 * u + cosx2 * v

    return u_out, v_out
