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

from lair.config import GROUP_DIR


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
        for attr in sounding._attrs:
            if hasattr(sounding, attr):
                df[attr] = getattr(sounding, attr)
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

