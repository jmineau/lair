"""
SODAR data from MesoWest.

Hopefully will add more functionality in the future.
"""

import os
import pandas as pd
import xarray as xr


# from lair.uataq.filesystem import DataFile, filter_datafiles  # this probably shouldn't be here


# class SodarFile(DataFile):
#     # TODO
#     pass

#     def parse(self):
#         pass


class Sodar:
    """
    Horel Group Sodar
    """
    # TODO move to mesowest module - import HorelH5File
    # TODO:
    #  - qaqc lvl is under hdf5archive/_Processed
    #  - how should this work with get_obs
    #  - 
    
    model = 'sodar'
    species_measured = ('wind',)

    sodar_dir = os.path.join(MESOWEST_DIR, 'sodar_data')
    archive_dir = os.path.join(sodar_dir, 'hdf5archive')

    def __init__(self, SID: str):
        self.SID = SID.upper()

        self.metafile = os.path.join(self.archive_dir, 
                                     f'{self.SID}_full_metadata_log.h5')
        self.meta = pd.read_hdf(self.metafile, key='metagroup/metadata')
        self.variables = pd.read_hdf(self.metafile, key='metagroup/variables')
        self.variables.set_index('SHORTNAME', inplace=True)

    def get_files(self, lvl: str='raw', time_range=(None, None)):
        files = []
        for file in os.listdir(self.archive_dir):
            if file.startswith(self.SID) and file.endswith('sodar.h5'):
                file_path = os.path.join(self.archive_dir, file)
                date_str = file[6:13].replace('_', '-')
                period = pd.Period(date_str, freq='M')
                files.append(SodarFile(file_path))
        return filter_datafiles(files, time_range)
    
    @staticmethod
    def parse(file, variables):
        # FIXME I don't want to have to pass variables every time
        # but variables may change between sites - need to check this
        # if not, then I can just make variables an attribute of the class
        with pytbls.open_file(file, mode='r') as f:
            table = f.root['obsdata/observations']
            data = table.read()

        var_names = data.dtype.names

        time = pd.to_datetime(data['DATTIM'], unit='s')

        data_vars = {'STATION_ID': (['Time_UTC'], data['STATION_ID']),
                    **{var: (['Time_UTC', 'level'], data[var])
                        for var in var_names
                        if var not in ['DATTIM', 'STATION_ID']}}

        ds = xr.Dataset(
            coords={'Time_UTC': time},
            data_vars=data_vars)

        ds = ds.where(ds != -9999)

        for variable in variables.index:
            ds[variable] /= variables.loc[variable, 'MULT']

        return ds

    def read_data(self, lvl='raw', time_range=(None, None), num_processes=1):
        files = self.get_files(lvl, time_range)
        read_files = parallelize_file_parser(self.parse, num_processes=num_processes)
        data = xr.merge(read_files(files, variables=self.variables))

        return data.sel(Time_UTC=slice(time_range[0], time_range[1]))
    
    @staticmethod
    def get_winds_at_height(data, height):
        import numpy as np  # FIXME
        level = np.where(data.HEIGHT[0] == height)[0][0]
        winds = data.sel(level=level).to_pandas()[['WD', 'WS']]

        winds.rename(columns={'WD': 'direction', 'WS': 'speed'}, inplace=True)

        winds.dropna(how='all', inplace=True)

        return winds
