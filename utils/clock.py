#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 22:32:43 2023

@author: James Mineau
"""

import datetime as dt
import pytz

SEASONS = {1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA',
           7: 'JJA', 8: 'JJA', 9: 'SON', 10: 'SON', 11: 'SON', 12: 'DJF'}


def UTC2MTN(data):
    mtn = pytz.timezone('US/Mountain')
    return data.tz_localize(pytz.utc).tz_convert(mtn).tz_localize(None)


def MTN2UTC(data):
    mtn = pytz.timezone('US/Mountain')
    return data.tz_localize(mtn).tz_convert(pytz.utc).tz_localize(None)


def UTC2MST(data):
    data.index = data.index - dt.timedelta(hours=7)
    return data
