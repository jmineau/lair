#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 12:05:30 2023

@author: James Mineau (James.Mineau@utah.edu)

Pipeline errors
"""


class ParsingError(Exception):
    # Exception to catch custom parsing errors
    pass


class InstrumentNotFoundError(Exception):
    def __init__(self, site, specie, time_range):
        msg = ('Instrument not found for site - specie combo'
               ' in given time_range')
        super().__init__(msg)
