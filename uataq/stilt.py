#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:27:35 2023

@author: James Mineau (James.Mineau@utah.edu)

Module for preparing data for STILT runs & processing STILT output
"""


import os
import datetime as dt
from functools import cached_property

# TODO this needs to be a parameter
STILT_projects_dir = ('/uufs/chpc.utah.edu/common/home'
                      '/lin-group15/jkm/STILT')

TIME_FORMAT = '%Y%m%d%H%M'


def fix_sim_links(old_out_dir, new_out_dir):
    '''
    Fix sim symlinks if stilt wd was changed

    Parameters
    ----------
    old_out_dir : TYPE
        old out directory where symlinks currently point to and shouldnt.
    new_out_dir : TYPE
        new out directory where symlinks should point to.

    Returns
    -------
    None.

    '''

    for subdir in ['footprints', 'particles']:
        for file in os.listdir(os.path.join(new_out_dir, subdir)):
            filepath = os.path.join(new_out_dir, subdir, file)

            # Check if file is link
            if os.path.islink(filepath):
                old_link = os.readlink(filepath)

                # Check if old_link is to file in old_out_dir
                if old_link.startswith(old_out_dir):
                    # Get sim_id of simulation
                    sim_id = os.path.basename(os.path.dirname(old_link))

                    # Remove old_link
                    os.remove(filepath)

                    # Create new link to new_out_dir
                    new_link = os.path.join(new_out_dir, 'by-id', sim_id, file)
                    os.symlink(new_link, filepath)


class Receptors:
    def __init__(self, loc, times):

        self.data = self.generate(loc, times)

    def generate(self, loc, times):
        '''
        Generate receptor dataframe when given a location and list of datetimes
        '''
        import pandas as pd

        if isinstance(loc, str):
            lati, long, zagl = Receptors.get_site_location(loc)
        else:
            lati, long, zagl = loc

        times.name = 'run_time'

        N = len(times)

        receptors = pd.DataFrame({'long': [long] * N,
                                  'lati': [lati] * N,
                                  'zagl': [zagl] * N},
                                 index=times)

        return receptors

    def update(self, receptors):
        '''
        Update data with more receptors from a different Receptors instance
        '''
        import pandas as pd

        assert type(receptors) == Receptors

        self.data = pd.concat([self.data, receptors.data])

        return None

    def save(self, out_path):
        self.data.to_csv(out_path)

        return None

    @staticmethod
    def get_site_location(site):
        from . import uucon
        site_config = uucon.site_config.loc[site]

        lati = site_config.lati.astype(float)
        long = site_config.long.astype(float)
        zagl = site_config.zagl.astype(float)

        return lati, long, zagl

    @staticmethod
    def nearest_mesowest(loc):
        # TODO
        pass


class Footprints:

    def __init__(self, footprint_dir, subset=False, engine='rasterio',
                 cache=False, reload_cache=False):

        assert os.path.exists(footprint_dir)  # make sure footprint dir exists

        self.footprint_dir = footprint_dir
        self.files = None

        if cache:
            from utils.records import Cacher
            if cache is True:
                cache = os.path.join(footprint_dir, 'footprints_cache.pkl')
            self.read = Cacher(self.read, cache, reload=reload_cache,
                               verbose=True)
        self.cache = cache

        self.foots = self.read(footprint_dir, subset, engine)
        self.weighted = self.apply_weights()

        self.receptor_locs = {}

    def __repr__(self):
        # TODO I think you should be able to recreate instance from repr
        #   which probably isnt true here
        # I think this should be __str__
        from math import log10, floor

        def round_to_1(x):
            return round(x, -int(floor(log10(abs(x)))))

        start = str(self.foots.sim_time.values[0])[:10]
        end = str(self.foots.sim_time.values[-1])[:10]
        time_range = f'{start} ~ {end}'

        resolution = round_to_1(self.foots.rio.resolution()[0])
        digits = len(str(resolution).split('.')[1])
        bounds = [round(x, digits) for x in self.foots.rio.bounds()]

        return (f'Footprints(time range="{time_range}", '
                f'resolution="{resolution}", bounds="{bounds}", '
                f'dir="{self.footprint_dir}", cache="{self.cache}")')

    def read(self, footprint_dir, subset, engine):
        import xarray as xr

        def _preprocess(ds):
            filename = ds.encoding['source']

            if subset:
                ds = self._sub(ds, subset)

            ds = ds.sum(dim='time')  # temporal sum
            ds = ds.expand_dims({'sim_time':  # simulation time to concat on
                                 [Footprints.get_sim_id(filename)['time']]})

            return ds

        print(f'Reading footprints from {footprint_dir}')

        self.files = Footprints.get_files(footprint_dir)

        processed_foots = [_preprocess(xr.open_dataset(file, engine=engine))
                           for file in self.files]
        foots = xr.concat(processed_foots, dim='sim_time').foot
        foots = foots.sortby('sim_time')

        return foots

    def _sub(self, ds, subset):
        if ds.rio.y_dim == 'y':
            y_slicer = slice(subset[3], subset[1])
        elif ds.rio.y_dim[:3].lower() == 'lat':
            y_slicer = slice(subset[1], subset[3])

        return ds.sel({ds.rio.x_dim: slice(subset[0], subset[2]),
                       ds.rio.y_dim: y_slicer})

    def sub(self, subset):
        return self._sub(self.foots, subset)

    def get_receptors_locs(self):
        assert len(self.files) >= 1

        sim_ids = [Footprints.get_sim_id(file) for file in self.files]

        # Get sim_locs of all the files without time - list of dicts
        sim_locs = [{'lati': sim_id['lati'],
                     }
                    for var in ['lati', 'long', 'zagl']
                    for sim_id in sim_ids]

        sim_locs = [{k: v for k, v in ID.items() if k != 'time'}
                    for ID in sim_ids]

        # remove duplicate locs
        sim_locs = [dict(t) for t in {tuple(loc.items()) for loc in sim_locs}]

        return sim_locs

    @cached_property
    def area(self):
        from utils.grid import area_DataArray

        return area_DataArray(self.foots)

    def apply_weights(self):
        import numpy as np

        weights = np.cos(np.deg2rad(self.foots[self.foots.rio.y_dim]))
        weighted = self.foots.weighted(weights)
        return weighted

    def validate_transport(self):
        from synoptic.services import stations_timeseries

        # TODO
        #   first lets just use the station closest to the receptor
        #   later, we can try to combine all the stations within the STILT domain

        pass

    @staticmethod
    def get_files(footprint_dir):
        '''Get footprint files in footprint dir'''
        footprint_files = [os.path.join(footprint_dir, file)
                           for file in os.listdir(footprint_dir)
                           if file.endswith('foot.nc')]

        return sorted(footprint_files)

    @staticmethod
    def get_sim_id(file):
        time, lati, long, zagl, _ = os.path.basename(file).split('_')

        sim_id = {'time': dt.datetime.strptime(time, TIME_FORMAT),
                  'lati': lati,
                  'long': long,
                  'zagl': zagl}
        return sim_id

    @staticmethod
    def plot(foot, ax=None, crs=None, label_deci=1, tiler=None, tiler_zoom=9,
             bounds=None, x_buff=0.1, y_buff=0.1, labelsize=None,
             more_lon_ticks=0, more_lat_ticks=0):
        import cartopy.crs as ccrs
        from functools import partial
        from matplotlib.ticker import FuncFormatter as FFmt
        import matplotlib.pyplot as plt
        import numpy as np

        from utils.plotter import log10formatter
        from utils.grid import add_latlon_ticks

        if tiler is not None:
            crs = tiler.crs
            if ax is None:
                fig, ax = plt.subplots(subplot_kw={'projection': crs})
            ax.add_image(tiler, tiler_zoom)
        elif crs is None:
            crs = ccrs.PlateCarree()

        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': crs})

        if bounds is None:
            bounds = foot.rio.bounds()
        extent = [bounds[0]-x_buff, bounds[2]+x_buff,
                  bounds[1]-y_buff, bounds[3]+y_buff]

        label = r'footprint [ppm$ \left/ \frac{\mu mol}{m^2 s}\right.$]'
        log10formatter_part = partial(log10formatter, deci=label_deci)

        np.log10(foot).plot(ax=ax, transform=ccrs.PlateCarree(), alpha=0.5,
                            cmap='cool',
                            cbar_kwargs={'orientation': 'vertical',
                                         'pad': 0.03, 'label': label,
                                         'format': FFmt(log10formatter_part)})

        add_latlon_ticks(ax, extent, x_rotation=30, labelsize=labelsize,
                         more_lon_ticks=more_lon_ticks,
                         more_lat_ticks=more_lat_ticks)

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.set(ylabel=None,
               xlabel=None)

        return ax


class STILT:
    def __init__(self, project, project_dir=None):
        self.project = project
        self.project_dir = project_dir
        self.stilt_wd = os.path.join(STILT_projects_dir, self.project)  # FIXME
        self.footprint_dir = os.path.join(self.stilt_wd, 'out/footprints')

        if (not os.path.exists(self.stilt_wd)) & (project_dir is not None):
            print('Initializing STILT project...')
            self.init_project(self.stilt_wd, project_dir)

    def __repr__(self):
        return f'STILT(project="{self.project}")'

    def init_project(self, stilt_wd, project_dir):
        # Change working directory to STILT_projects
        os.chdir(STILT_projects_dir)

        # Initialize STILT project
        os.system(f"Rscript -e \"uataq::stilt_init(\"{self.project}\")\"")

        # Create sym-link in project_dir to stilt_wd
        os.symlink(stilt_wd, os.path.join(project_dir, 'STILT'))

    def generate_receptors(self, loc, times):
        self.receptors = Receptors(loc, times)

        return self.receptors

    def check_sims(self, site, times):
        # Check whether all time steps were simulated

        footprint_files = Footprints.get_files(self.footprint_dir)

        missed_sims = []
        for sim_time in times:
            sim_str = dt.datetime.strftime(sim_time, TIME_FORMAT)

            if not any(os.path.basename(file).startswith(sim_str)
                       for file in footprint_files):
                missed_sims.append(sim_time)

        return missed_sims

    def read_foots(self, footprint_dir=None, subset=False, engine='rasterio',
                   cache=False, reload_cache=False):
        if footprint_dir is None:
            footprint_dir = self.footprint_dir
        else:
            self.footprint_dir = footprint_dir

        self.footprints = Footprints(footprint_dir, subset, engine,
                                     cache, reload_cache)
        self.foots = self.footprints.foots

        return self.foots
