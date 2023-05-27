#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 09:56:03 2023

@author: James Mineau (James.Mineau@utah.edu)

Module to dispatch calls to read observations from instruments to their
respective instrument functions.
"""

from config import instrument_config
from .errors import InstrumentNotFoundError


instrument_mapper = {
    'metone_es642': 'metone',
    'esampler': 'metone'}


def select_instruments(site, specie, time_range):
    selection = []

    try:
        instruments = instrument_config[site][specie]
    except KeyError:
        raise InstrumentNotFoundError(site=site, specie=specie,
                                      time_range=time_range)

    for instrument in instruments:
        name = instrument['instrument']
        start_time = instrument['start_time']
        end_time = instrument.get('end_time')

        if time_range == [None, None] or (
                start_time <= time_range[0] and
                (end_time is None or end_time >= time_range[1])):
            selection.append(name)

    return selection


def dispatcher(func):
    """
    Decorator for multiple dispatch based on instrument
    """
    # Dictionary to store registered dispatch arguments and their corresponding functions
    registry = {}

    def register(instrument, func=None):
        """
        Register a specific implementation for a dispatch argument (instrument).

        This function can be used as a decorator or as a regular function call.
        If used as a decorator, it captures the dispatch argument and the function being decorated.
        If used as a regular function call, it registers the dispatch argument and the provided function.
        """
        if func is None:
            return lambda f: register(instrument, f)

        # Register the dispatch argument and its corresponding function
        registry[instrument] = func
        return func

    def decorated(site, specie, lvl='raw', time_range=[None, None], **kwargs):
        """
        Decorated function that performs the dispatch and calls the appropriate implementation.

        If an instrument is found, the corresponding function is called.
        If more than one instrument is read, the returned object takes the form {instrument: data}
        """

        # select instruments based on site, specie and time_range
        instruments = select_instruments(site, specie, time_range)

        # map instrument name if needed
        instruments = [instrument_mapper.get(instrument, instrument)
                       for instrument in instruments]

        if len(instruments) == 1:
            instrument_func = registry[instruments[0]]

            return instrument_func(site, specie, lvl, time_range, **kwargs)

        elif len(instruments) > 1:
            instrument_data = {}
            for instrument in instruments:
                instrument_func = registry[instrument]

                data = instrument_func(site, specie, lvl, time_range, **kwargs)
                instrument_data[instrument] = data

            return instrument_data

    decorated.register = register  # Attach the register function
    decorated.registry = registry  # Attach the registry dictionary

    return decorated
