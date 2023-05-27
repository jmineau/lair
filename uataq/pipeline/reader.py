#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:40:06 2023

@author: James Mineau (James.Mineau@utah.edu)
"""

# from . import bb
from .dispatch import dispatcher
from . import errors
# from .lgr import no2 as lgr_no2
from .lgr import ugga as lgr_ugga
# from . import metone
# from .teledyne import t500u as teledyne_t500u


@dispatcher
def read_obs(site, specie, lvl, time_range, **kwargs):
    """
    Main function for reading observations.
    """
    pass


# Register instrument reading functions

# read_obs.register('2b', bb.read_obs)
# read_obs.register('lgr_no2', lgr_no2.read_obs)
read_obs.register('lgr_ugga', lgr_ugga.read_obs)
# read_obs.register('metone', metone.read_obs)
# read_obs.register('teledyne_t500u', teledyne_t500u.read_obs)
