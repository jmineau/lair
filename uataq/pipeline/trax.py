#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:46:06 2023

@author: James Mineau (James.Mineau@utah.edu)

Module to process & read TRAX data
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, List, Optional, Union

from . import bb, lgr_no2, lgr_ugga, metone, teledyne_t500u
from config import MOBILE_DIR, HOREL_TRAX_DIR


VEHICLES = ('trx01', 'trx02', 'trx03')


@dataclass
class Species:
    instrument: str
    reader: Union[Callable, List[Callable]]
    column_name: str
    units: str
    latex_name: str
    qc_limits: List[Optional[float]] = field(default_factory=[None, None])


SPECIES = {
    'o3': Species(reader=bb.RAW, column_name='o3_ppb',
                  units='ppb', qc_limits=[-200, 400], latex_name='O$_3$'),
    # O3 is in multiple locations, how should I deal with this

    'pm25': Species(reader=metone.RAW,
                    column_name='pm25_ugm3', units=r'$\frac{\mu g}{m^3}$',
                    latex_name='PM2.5'),

    'co2': Species(reader=lgr_ugga.CALIBRATED,  # TODO uucon partial Site
                   column_name='co2d_ppm', units='ppm', latex_name='CO$_2$'),

    'ch4': Species(reader=lgr_ugga.CALIBRATED,
                   column_name='ch4d_ppm', units='ppm', latex_name='CH$_4$'),

    'no2': Species(reader=[teledyne.RAW, lgr_no2.RAW],
                   column_name='no2_ppb', units='ppb', latex_name='NO$_2$')
    }


def load_obs(vehicle, specie, months):
    s = SPECIES[specie]

    obs = s.reader(vehicle)


def load_gps(group='lin'):  # lin or horel group gps
    pass


def read_polygon_masks(kml_dir):  # Get RSC and TRAX line polygons
    from utils import read_kml

    polygons = {}
    kmls = [file for file in os.listdir(kml_dir) if file.endswith('kml')]

    for group in ['JRRSC', 'MRSC', 'TRAX']:
        polygons[group] = {}
        for kml_file in kmls:
            if kml_file.startswith(group):
                KML = read_kml(path.join(kml_dir, kml_file))
                poly = Polygon(KML._features[0]._features[0].geometry)
                mask = kml_file.split('.')[0][len(group) + 1:]  # polygon key
                polygons[group][mask] = poly
    return polygons


def process(vehicle, specie, data, inlet_lag, verbose=True):
    '''Process transects'''
    pass


def read():
    pass
