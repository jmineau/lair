"""
lair.met.soundings
~~~~~~~~~~~~~~~~~~

Module for working with upper air sounding data.
"""

from collections import deque
import datetime as dt
import numpy as np
import os
import pandas as pd
import requests
from siphon.simplewebservice.wyoming import WyomingUpperAir
from time import sleep
import xarray as xr

from lair.config import vprint, GROUP_DIR
from lair.constants import Rd
from lair.air.meteorology import ideal_gas_law, hypsometric, poisson
from lair.units import C2K, hPa, J, kg, K, m

SOUNDING_DIR = os.path.join(GROUP_DIR, 'soundings')


class Sounding:
    _attrs = ['station', 'time', 'station_number', 'latitude', 'longitude', 'elevation', 'pw']

    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)

        self.station = self.filename.split('_')[0]
        self.time = dt.datetime.strptime(self.filename.split('_')[1].split('.')[0],
                                         '%Y%m%d%H')

        self.data = pd.read_csv(path, parse_dates=['time'])
        self.data.drop(columns=['station', 'time'], inplace=True)

        for attr in self._attrs:
            if attr in self.data.columns:
                setattr(self, attr, self.data[attr].iloc[0])
                self.data.drop(columns=attr, inplace=True)

    def interpolate(self, start=1289, stop=5000, interval=10):
        """Interpolate sounding data to a specified height.

        Parameters
        ----------
        start : int
            The starting height.
        stop : int
            The stopping height.
        step : int
            The height step.
        
        Returns
        -------
        xr.Dataset
            The interpolated sounding data.
        """
        height = range(start, stop, interval)
        df = pd.DataFrame(index=height)
        data_height = self.data.dropna(subset='height').set_index('height')
        data_height['raw'] = True
        merged = pd.concat([df, data_height])
        merged.index.name = 'height'
        merged = merged.sort_values(['height', 'raw'])
        data = merged.interpolate(method='index')
        
        data = data.loc[data.raw.isna()
                        & (data.index >= start)
                        & (data.index < stop)]
        data.drop(columns='raw', inplace=True)

        ds = data.to_xarray().expand_dims(time=[self.time])
        ds['pw'] = (('time', ), [self.pw])

        attrs = {attr: getattr(self, attr) for attr in self._attrs
                 if hasattr(self, attr) and attr not in ['time', 'pw']}
        ds.attrs.update(attrs)
        ds.attrs['interpolation_interval'] = interval
        ds.attrs['notes'] = f'Interpolated to {start}-{stop}m at {interval}m intervals.'

        return ds

    def plot(self):
        # TODO
        raise NotImplementedError


def merge(soundings: list):
    """Merge a list of sounding data.

    Parameters
    ----------
    soundings : list
        A list of Sounding objects.

    Returns
    -------
    pd.DataFrame
        The merged sounding data.
    """
    dfs = []
    for sounding in soundings:
        df = sounding.data.copy()
        df['time'] = sounding.time
        df['pw'] = sounding.pw
        dfs.append(df)

    return pd.concat(dfs)


def download_sounding(station, date, dst=None):
    """Download an upper air sounding from the Wyoming archive.

    Parameters
    ----------
    station : str
        The 4-letter station identifier.  # FIXME: 3-letter?
    date : datetime
        The date and time.
    dst : str
        The destination directory.
    """
    if dst is None:
        dst = os.path.join(SOUNDING_DIR, station)
    os.makedirs(dst, exist_ok=True)

    path = os.path.join(dst, f'{station}_{date:%Y%m%d%H}.csv')
    if not os.path.exists(path):
        print(f'Downloading {station} on {date:%Y-%m-%d %H:%M}...')
        df = WyomingUpperAir.request_data(date, station)
        df.to_csv(path, index=False)
    else:
        print(f'{station} on {date:%Y-%m-%d %H:%M} already exists.')

    return path

def download_soundings(station, start, end, dst=None, months=None):
    print('Downloading soundings...')

    dates = pd.date_range(start, end, freq='12h')

    if months:
        dates = dates[dates.month.isin(months)]

    to_download = deque(dates)
    while len(to_download) > 0:
        date = to_download.popleft()
        try:
            download_sounding(station, date, dst)
        except IndexError as e:
            print(f'Error downloading {station} on {date:%Y-%m-%d %H:%M}: {e}')
            continue
        except ValueError as e:
            print(f'Error downloading {station} on {date:%Y-%m-%d %H:%M}: {e}')
            if 'No data available' in str(e):
                continue
            else:
                raise e
        except requests.exceptions.HTTPError as e:
            print(f'Error downloading {station} on {date:%Y-%m-%d %H:%M}: {e}')
            if 'Please try again later' in str(e):
                print('Trying again in 2 seconds...')
                to_download.appendleft(date)
                sleep(2)
            else:
                raise e


def get_soundings(station='SLC', start=None, end=None, sounding_dir=None, months=None,
                  driver='xarray', **kwargs):
    """
    Get upper air soundings from the Wyoming archive.

    Parameters
    ----------
    station : str
        The 4-letter station identifier.
    start : datetime
        The start date and time.
    end : datetime
        The end date and time.
    sounding_dir : str
        The directory containing the sounding data.

    Returns
    -------
    pd.DataFrame
        The sounding data.
    """
    if sounding_dir is None:
        sounding_dir = os.path.join(SOUNDING_DIR, station)

    files = os.listdir(sounding_dir)
    if len(files) == 0:
        print('No soundings found. Downloading...')

        if not all([start, end]):
            raise ValueError('start and end must be specified if no soundings are found.')
        download_soundings(station, start, end, sounding_dir, months)

    soundings = []
    # TODO filter by start/end
    for file in files:
        path = os.path.join(sounding_dir, file)
        try:
            soundings.append(Sounding(path))
        except Exception as e:
            print(f'Error reading file {path}: {e}')
            continue
        
    if driver in ['xarray', 'nc']:
        data = xr.concat([sounding.interpolate() for sounding in soundings],
                         dim='time').sortby('time')
    elif driver in ['pandas', 'csv']:
        data = merge(soundings)
    else:
        raise ValueError(f'Invalid driver: {driver}')

    return data


def valleyheatdeficit(data, integration_height=2200):
    """
    Calculate the valley heat deficit.

    Whiteman, C. David, et al. “Relationship between Particulate Air Pollution
    and Meteorological Variables in Utah's Salt Lake Valley.”
    Atmospheric Environment, vol. 94, Sept. 2014, pp. 742-53.
    DOI.org (Crossref), https://doi.org/10.1016/j.atmosenv.2014.06.012.

    Parameters
    ----------
    data : xr.DataSet
        The sounding data.
    integration_height : int
        The height to integrate to.
    
    Returns
    -------
    xr.DataArray
        The valley heat deficit.
    """
    h0 = data.elevation * m
    cp = 1005 * J/kg/K
    
    # Subset to the heights between the surface and the integration height
    data = data.sel(height=slice(h0, integration_height))
    
    T = C2K(data.temperature)
    p = data.pressure * hPa

    # Calculate potential temperature using poisson's equation
    theta = poisson(T, p)
    theta_h = theta.sel(height=integration_height, method='nearest')

    # Calculate virtual temperature to account for water vapor
    Tv = hypsometric(p1=p.isel(height=slice(0, -1)).values,
                     p2=p.isel(height=slice(1, None)).values,
                     deltaz=data.interpolation_interval)
    layer_heights = T.height.values[:-1] + data.interpolation_interval / 2
    Tv = xr.DataArray(Tv, coords=[data.time, layer_heights],
                      dims=['time', 'height']).interp_like(T, method='linear')

    # Calculate the density using the ideal gas law
    rho = ideal_gas_law(solve_for='rho', p=p, T=Tv, R=Rd)

    # Calculate the heat deficit by integrating using the trapezoid method
    heat_deficit = (cp * rho * (theta_h - theta)).dropna('height', how='all').integrate('height')

    return heat_deficit  # J/m2
