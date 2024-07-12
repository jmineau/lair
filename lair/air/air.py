"""
lair.air.air
~~~~~~~~~~~~

Module of miscellaneous functions for air quality data.
"""

import numpy as np
import pandas as pd

from lair.utils.clock import AFTERNOON


# %% Time Series

def get_well_mixed(data: pd.Series | pd.DataFrame, hours: list[int]=AFTERNOON):
    """
    Subset the data to the well-mixed hours of the day.

    Parameters
    ----------
    data : pd.Series | pd.DataFrame
        Time series data to subset. Must have a datetime index.
    hours : list[int], optional
        Hours of the day to subset. Default is LST afternoon hours.

    Returns
    -------
    pd.Series | pd.DataFrame
        Subset of the data for the well-mixed hours.
    """
    return data[data.index.hour.isin(hours)].resample('1d').mean()


# %% Polar

def bin_polar(data, wd='wd', x='ws', xbins=30):
    # Bin directions
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


def circularize_contour_data(agg):
    theta = agg.index.values
    dtheta = np.diff(theta).mean()
    theta = np.concatenate((theta, theta[-1:] + dtheta))

    r = agg.columns.values
    r, theta = np.meshgrid(r, theta)

    c = agg.values
    c = np.concatenate((c, c[0:1, :]), axis=0)

    return theta, r, c


# %% Wind

def wind_components(speed, wind_direction):
    # Wind direction is in degrees clockwise from North
    #   specified as blowing from
        
    u = - speed * np.sin(np.deg2rad(wind_direction))
    v = - speed * np.cos(np.deg2rad(wind_direction))
    
    return u, v
