from . import config

import cf_xarray.units  # must be imported before pint_xarray
import pint
from pint.delegates.formatter._compound_unit_helpers import sort_by_dimensionality
import pint_xarray
from pint_xarray import unit_registry as units

# Custom pint context to convert fluxes from mass <--> substance
mass_flux = pint.Context('mass_flux')
mass_flux.add_transformation('[substance] / [area] / [time]',
                             '[mass] / [area] / [time]',
                        lambda units, substance, mw: substance * mw)
mass_flux.add_transformation('[mass] / [area] / [time]',
                             '[substance] / [area] / [time]',
                        lambda units, mass, mw: mass / mw)
units.add_context(mass_flux)

# Set default pint sorting function to sort by dimensionality
units.formatter.default_sort_func = sort_by_dimensionality

from . import air
from . import inversion
from . import uataq
from . import utils


__version__ = '2025.12.3'
