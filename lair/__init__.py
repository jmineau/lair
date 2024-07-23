from . import config
from . import air
from . import uataq
from . import utils


__version__ = '2024.07.0'


import cf_xarray.units  # must be imported before pint_xarray
import pint
import pint_xarray
from pint_xarray import unit_registry as units

mass_flux = pint.Context('mass_flux')
mass_flux.add_transformation('[substance] / [area] / [time]',
                             '[mass] / [area] / [time]',
                        lambda units, substance, mw: substance * mw)
mass_flux.add_transformation('[mass] / [area] / [time]',
                             '[substance] / [area] / [time]',
                        lambda units, mass, mw: mass / mw)
units.add_context(mass_flux)
