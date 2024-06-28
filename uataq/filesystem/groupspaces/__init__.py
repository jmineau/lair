from os import listdir
from os.path import dirname, isfile, join

groupspaces = dirname(__file__)
__all__ = [f[:-3] for f in listdir(groupspaces)
           if isfile(join(groupspaces, f))
           and not f.startswith('_')]

from . import *
