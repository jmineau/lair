#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 15:35:03 2023

@author: James Mineau (James.Mineau@utah.edu)

Module for working with files and directories
"""

from collections import namedtuple

DataFile = namedtuple('DataFile', ['path', 'date'])


def unzip(zf, dir_path=None):
    '''Unzip file into dir_path if given'''
    import zipfile

    with zipfile.ZipFile(zf, 'r') as zip_ref:
        zip_ref.extractall(dir_path)


def read_kml(kml_path):
    '''Read kml file from path'''
    from fastkml import kml

    with open(kml_path, 'rt') as KML:
        k = kml.KML()
        k.from_string(KML.read().encode('utf-8'))
    return k


def filter_files(files, time_range=None):
    import pandas as pd

    from uataq.pipeline.preprocess import process_time_range

    start_time, end_time = process_time_range(time_range)

    filtered_files = []

    df = pd.DataFrame(files, columns=DataFile._fields)
    df = df.set_index('date').sort_index()

    filtered_files = df.loc[start_time: end_time, 'path'].tolist()

    if len(filtered_files) == 0:
        raise ValueError('No data within given time_range')

    return filtered_files


def parallelize_file_parser(file_parser, num_processes=None):
    from functools import partial
    import multiprocessing

    def parallelizer(files, **kwargs):

        if num_processes == 1:
            # Don't start multiprocessing pool
            dfs = [file_parser(file, **kwargs) for file in files]
            return dfs

        # Create a multiprocessing Pool
        processes = num_processes if num_processes \
            else multiprocessing.cpu_count()

        pool = multiprocessing.Pool(processes=processes)

        # Apply the decorated function in parallel to the list of files
        dfs = pool.map(partial(file_parser, **kwargs), files)

        # Close the pool to free resources
        pool.close()
        pool.join()

        return dfs

    return parallelizer


class Cacher:
    """
    A class that caches function results to a file for future use.

    Parameters
    ----------
    func : function
        The function to be cached.
    cache_file : str
        The name of the file to cache results to.
    """

    import pickle as pkl

    def __init__(self, func, cache_file, verbose=False, reload=False):
        """
        Initializes a Cacher object.

        Parameters
        ----------
        func : function
            The function to be cached.
        cache_file : str
            The name of the file to cache results to.
        """
        assert cache_file.endswith('.pkl')

        self.func = func
        self.cache_file = cache_file
        self.verbose = verbose
        self.reload = reload

        self.cache_index = self.load_cache_index()

    def load_cache_index(self):
        """
        Loads the cache index from a file.

        Returns
        -------
        dict
            A dictionary of cached results and their corresponding file
            positions.
        """
        if self.reload:
            # TODO causes previous caches to become unreadable
            #   need to delete previous cache or append to index_file
            # Force func to be executed by setting cache_index to empty
            if self.verbose:
                print(f'Forcing {self.func.__name__} to execute')
            return {}

        try:
            with open(self.cache_file + '.index', 'rb') as f:
                cache_index = self.pkl.load(f)
        except FileNotFoundError:
            cache_index = {}

        return cache_index

    def save_cache_index(self):
        """
        Saves the cache index to a file.
        """
        with open(self.cache_file + '.index', 'wb') as f:
            self.pkl.dump(self.cache_index, f,
                          protocol=self.pkl.HIGHEST_PROTOCOL)

    def __call__(self, *args):
        """
        Returns the cached result if available, otherwise calls the function
        and caches the result for future use.

        Parameters
        ----------
        *args : tuple
            The arguments to be passed to the function.

        Returns
        -------
        Any
            The result of the function call.
        """

        key = self.pkl.dumps(args)  # serialize func args

        if key in self.cache_index:
            # Load the result from the cache_file using its index
            if self.verbose:
                print(f"Returning cached result for {args}")

            with open(self.cache_file, 'rb') as f:
                f.seek(self.cache_index[key])  # go to index in cache_file
                result = self.pkl.load(f)

        else:
            # Call the function and update the cache
            result = self.func(*args)

            with open(self.cache_file, 'ab') as f:
                f.seek(0, 2)  # go to the end of the pickle file
                pos = f.tell()  # get position of the end

                # dump result to the end of the pickle
                self.pkl.dump(result, f, protocol=self.pkl.HIGHEST_PROTOCOL)

            self.cache_index[key] = pos

            if self.verbose:
                print(f"Added {args} to cache")

            self.save_cache_index()  # update cache_index
        return result
