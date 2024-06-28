"""
lair.air.carbontracker
~~~~~~~~~~~~~~~~~~~~~~

Module for reading CarbonTracker data.
"""

def read_carbontracker(path, parallel=True):
    def preprocess(ds):
        time_components = ds['time_components'].values
        time = [dt.datetime(*row) for row in time_components]
        
        ds = ds.assign_coords(time=time)
        
        ds = ds.drop_vars('time_components')
        
        return ds
    
    files = list_files(path, '*nc', full_names=True, recursive=True)
    
    ds = xr.open_mfdataset(files, preprocess=preprocess, parallel=parallel)
    
    return ds