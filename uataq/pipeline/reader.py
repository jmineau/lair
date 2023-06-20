#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:40:06 2023

@author: James Mineau (James.Mineau@utah.edu)
"""

from config import instrument_config
from .errors import InstrumentNotFoundError
from . import instruments
from . import merger
from .preprocess import preprocessor


SPECIE_MAPPER = {
    'PM25': 'PM_25',
    'PM2.5': 'PM_25'}


@preprocessor
def select_instruments(site, specie, time_range=None):
    start_time, end_time = time_range
    selection = []

    specie = SPECIE_MAPPER.get(specie, specie)

    try:
        instruments = instrument_config[site][specie]
    except KeyError:
        raise InstrumentNotFoundError(site=site, specie=specie,
                                      time_range=time_range)

    for instrument in instruments:
        name = instrument['instrument']
        installation_date = instrument['installation_date']
        removal_date = instrument.get('removal_date')

        if [start_time, end_time] == [None, None] or (
                installation_date <= start_time and
                (removal_date is None or removal_date >= end_time)):
            selection.append(name)

    return selection


def dispatcher(func):
    """
    Decorator for multiple dispatch based on instrument
    """

    # Store registered dispatch arguments and their corresponding functions
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

    def decorated(site, species, lvl='raw', time_range=None, **kwargs):
        """
        Decorated function that performs the dispatch and calls the appropriate implementation.
        """

        specie_data = {}

        for specie in species:
            # select instruments based on site, specie and time_range
            instruments = select_instruments(site, specie=specie,
                                             time_range=time_range)

            instrument_data = {}

            for instrument in instruments:

                try:
                    instrument_func = registry[instrument]
                except KeyError as e:
                    raise NotImplementedError('read_obs function '
                                              f'not implemented for {e}')

                other_specie = None

                # Handle special case for "lgr_ugga" instrument
                if instrument == 'lgr_ugga':
                    other_specie = 'CH4' if specie == 'CO2' else 'CO2'

                    if all(s in species for s in ['CO2', 'CH4']):
                        # Both "co2" and "ch4" are present
                        #   call lgr.read_obs once
                        species.remove(other_specie)
                        specie, other_specie = ('CO2', 'CH4'), None

                data = instrument_func(site, specie, lvl=lvl,
                                       time_range=time_range, **kwargs)

                instrument_data[instrument] = data

            if len(instruments) == 1:
                data = instrument_data[instrument]
            elif len(instruments) > 1:
                data = merger.merge(instrument_data)  # TODO

            specie_data[specie] = data

        if len(species) == 1:
            return specie_data[specie]

        return specie_data

    decorated.register = register  # Attach the register function
    decorated.registry = registry  # Attach the registry dictionary

    return decorated


@preprocessor
@dispatcher
def read_obs(site, species, lvl, time_range, **kwargs):
    """
    Function for reading observations to be decorated by dispatcher.
    """
    pass


# Register instrument reading functions

read_obs.register('2b', instruments.bb.read_obs)
read_obs.register('esampler', instruments.metone.read_obs)
read_obs.register('gps', instruments.gps.read_obs)
# read_obs.register('lgr_no2', instruments.lgr_no2.read_obs)
read_obs.register('lgr_ugga', instruments.lgr_ugga.read_obs)
read_obs.register('licor_6262', instruments.licor_6262.read_obs)
read_obs.register('licor_7000', instruments.licor_7000.read_obs)
# read_obs.register('magee_ae33', instruments.magee.read_obs)
read_obs.register('metone', instruments.metone.read_obs)
read_obs.register('metone_es642', instruments.metone.read_obs)
# read_obs.register('teledyne_t200', instruments.teledyne_t200.read_obs)
# read_obs.register('teledyne_t300', instruments.teledyne_t300.read_obs)
# read_obs.register('teledyne_t400', instruments.teledyne_t400.read_obs)
read_obs.register('teledyne_t500u', instruments.teledyne_t500u.read_obs)
# read_obs.register('teom_1400ab', instruments.teom.read_obs)
