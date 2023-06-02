#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:40:57 2023

@author: James Mineau (James.Mineau@utah.edu)

Module of uataq pipeline functions for metone esampler instrument
"""

# horel-group calls it esampler, lin-group calls it metone

# lin-group9/measurements/data/trx02 has pm2.5 data from 2016-02-04 ~ 2017-06-20
#   there are two directories, 'met' & 'metone', which span the same time
#   range, but have different file contents

# TRX01 began collecting pm2.5 data Dec. 2014, but I am unable to locate the
#   data before Nov. 2018

# horel-group/uutrax/esampler has pm2.5 data for all 3 trax cars from Nov. 2018
#   onwards

SITES = ('dbk', 'sug', 'wbb', 'trx01', 'trx02', 'trx03')


def read(site):
    if site.lower().startswith('trx'):
        return read_trax(site)


def read_trax(vehicle):
    pass
