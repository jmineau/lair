#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 10:20:25 2023

@author: James Mineau (James.Mineau@utah.edu)
"""

from dataclasses import dataclass, field
from typing import List, Sequence, Optional, Union

from config import site_config
from . import pipeline as pipe


@dataclass
class Site:
    ID: str
    species: Union[str, List[str]]
    lvl: str = 'raw'
    time_range: List[Optional[float]] = field(default_factory=lambda: [None, None])

    verbose: bool = False

    def __post_init__(self):
        self.ID = self.ID.lower()
        if isinstance(self.species, str):
            self.species = self.species.lower()
        elif isinstance(self.species, Sequence):
            self.species = [s.lower() for s in self.species]

        self.latitude = site_config.loc[self.ID, 'lati']
        self.longitude = site_config.loc[self.ID, 'long']
        self.zagl = site_config.loc[self.ID, 'zagl']

    def _read(self, **kwargs):
        data = pipe.reader.read_obs(self.ID, self.species, self.lvl,
                                    self.time_range, **kwargs)

        return data
