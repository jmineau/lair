"""
Upper air sounding data.
"""

try:
    from siphon.simplewebservice.wyoming import WyomingUpperAir
except ImportError:
    print('siphon not installed. Please install siphon to use the soundings module.')

from collections import deque
import datetime as dt
import os
import pandas as pd
import requests
from time import sleep
import xarray as xr

from lair.config import vprint, GROUP_DIR
from lair import units
from lair.constants import Rd, cp
from lair.air.meteorology import ideal_gas_law, hypsometric, poisson


#: Sounding data directory
SOUNDING_DIR = os.path.join(GROUP_DIR, 'soundings')


class Sounding:
    """
    Upper air sounding data.

    Attributes
    ----------
    path : str
        The path to the sounding data.
    filename : str
        The filename of the sounding data.
    station : str
        The station identifier.
    time : datetime
        The date and time of the sounding.
    data : pd.DataFrame
        The sounding data.

    Methods
    -------
    interpolate(start=1289, stop=5000, interval=10)
        Interpolate sounding data to regular height intervals.
    """
    _attrs = ['station', 'time', 'station_number', 'latitude', 'longitude', 'elevation', 'pw']

    units = {'pressure': 'hPa', 'height': 'meter', 'temperature': 'degC', 'dewpoint': 'degC',
             'direction': 'degrees', 'speed': 'knot', 'u_wind': 'knot', 'v_wind': 'knot', 
             'latitude': 'degrees', 'longitude': 'degrees', 'elevation': 'meter', 'pw': 'millimeter'}

    def __init__(self, path: str):
        """
        Initialize a Sounding object.

        Strips the station identifier and time from the filename and reads the data.

        Parameters
        ----------
        path : str
            The path to the sounding data.
        """
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


def merge(soundings: list) -> pd.DataFrame:
    """
    Merge a list of sounding data.

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


def download_sounding(station, date, dst=None) -> str:
    """
    Download an upper air sounding from the Wyoming archive.

    Parameters
    ----------
    station : str
        The 4-letter station identifier.  # FIXME: 3-letter?
    date : datetime
        The date and time.
    dst : str
        The destination directory.

    Returns
    -------
    str
        The path to the downloaded sounding data.
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
    """
    Download upper air soundings from the Wyoming archive.

    Parameters
    ----------
    station : str
        The 4-letter station identifier.
    start : datetime
        The start date and time.
    end : datetime
        The end date and time.
    dst : str
        The destination directory.
    months : list
        The months to download.
    """
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
    for file in files:
        # Skip files that don't match the date range
        date = file.split('_')[1].split('.')[0]
        date = dt.datetime.strptime(date, '%Y%m%d%H')
        if start:
            if date <= start:
                continue
        if end:
            if date > end:
                continue

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


def valleyheatdeficit(data: xr.Dataset, integration_height=2200) -> xr.DataArray:
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
        The height to integrate to [m].
    
    Returns
    -------
    xr.DataArray
        The valley heat deficit [MJ/m^2].
    """
    h0 = data.elevation

    # Subset to the heights between the surface and the integration height
    data = data.sel(height=slice(h0, integration_height))
    
    T = data.temperature.pint.quantify('degC').pint.to('degK')
    p = data.pressure.pint.quantify('hPa').pint.to('Pa')

    # Calculate potential temperature using poisson's equation
    theta = poisson(T=T, p=p, p0=1e5 * units('Pa'))
    theta_h = theta.sel(height=integration_height, method='nearest')

    # Calculate virtual temperature to account for water vapor
    Tv = hypsometric(p1=p.isel(height=slice(0, -1)).values,
                     p2=p.isel(height=slice(1, None)).values,
                     deltaz=data.interpolation_interval * units('m'))
    layer_heights = T.height.values[:-1] + data.interpolation_interval / 2
    Tv = xr.DataArray(Tv, coords=[data.time, layer_heights],
                      dims=['time', 'height'])\
            .interp_like(T, method='linear')\
            .pint.quantify('degK')

    # Calculate the density using the ideal gas law
    rho = ideal_gas_law(solve_for='rho', p=p, T=Tv, R=Rd)
    # Set pint units - Setting units to kg/m3 doesnt change the numbers
    # pint-xarray hasnt implemented .to_base_units() yet
    # when they do, we can change this to .pint.to_base_units()
    rho = rho.pint.to('kg/m^3')

    # Calculate the heat deficit by integrating using the trapezoid method
    heat_deficit = (cp * rho * (theta_h - theta)).dropna('height', how='all')\
        .integrate('height') * (1 * units('m'))  # J/m2

    return heat_deficit.pint.to('MJ/m^2').pint.dequantify()
