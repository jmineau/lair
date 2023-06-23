#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 12:22:40 2023

@author: James Mineau (James.Mineau@utah.edu)
"""

import calendar
import functools
import inspect


def process_time_range(time_range):
    if time_range is None:
        # Handle the case when time_range is None
        start_time = None
        end_time = None

    elif isinstance(time_range, str):
        count = time_range.count('-')

        if count == 0:
            # Handle the case when time_range is a string
            #   representing a single year (YYYY)
            assert len(time_range) == 4
            start_time = f"{time_range}-01-01"
            end_time = f"{time_range}-12-31"
        elif count == 1:
            # Handle the case when time_range is a string
            #   representing a single month (YYYY-MM)
            assert len(time_range) == 7
            year, month = time_range.split("-")
            start_time = f"{year}-{month}-01"
            last_day = calendar.monthrange(int(year), int(month))[1]
            end_time = f"{year}-{month}-{last_day:02d}"
        elif count == 2:
            # Handle the case when time_range is a string
            #   representing a single day (YYYY-MM-DD)
            assert len(time_range) == 10
            start_time = time_range
            end_time = time_range
        else:
            raise ValueError('Invalid time_range format')

    elif isinstance(time_range, (list, tuple)) and len(time_range) == 2:
        # Handle the case when time_range is a tuple
        #   representing start_time and end_time
        start_time, end_time = time_range

    else:
        raise ValueError("Invalid time_range format")

    # Return the processed start_time and end_time
    return start_time, end_time


def preprocessor(func):

    @functools.wraps(func)
    def _preprocess(*args, **kwargs):
        bound_args = inspect.signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Preprocess arguments
        if 'site' in bound_args.arguments:
            bound_args.arguments['site'] = bound_args.arguments['site'].lower()

        if 'species' in bound_args.arguments:
            species = bound_args.arguments['species']
            if isinstance(species, str):
                species = [species.upper()]
            elif isinstance(species, list):
                species = [s.upper() for s in species]
            bound_args.arguments['species'] = species
        elif 'specie' in bound_args.arguments:
            specie = bound_args.arguments['specie']
            bound_args.arguments['specie'] = specie.upper()

        if 'lvl' in bound_args.arguments:
            lvl = bound_args.arguments['lvl'].lower()
            if lvl not in ['raw', 'qaqc', 'calibrated']:
                raise ValueError("Invalid value for 'lvl' parameter.")
            bound_args.arguments['lvl'] = lvl

        if 'time_range' in bound_args.arguments:
            time_range = bound_args.arguments['time_range']
            bound_args.arguments['time_range'] = process_time_range(time_range)

        if 'num_processes' in bound_args.arguments:
            num_processes = bound_args.arguments['num_processes']
            assert num_processes is None or (isinstance(num_processes, int)
                                             and num_processes > 0)

        # Call the original function with preprocessed arguments
        return func(*bound_args.args, **bound_args.kwargs)

    return _preprocess
