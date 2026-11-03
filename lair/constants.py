"""
Physical constants, as pint quantities of :data:`lair.units`.

``lair.meteorology`` uses them as plain SI numbers. pint has no constants of its
own yet (https://github.com/hgrecco/pint/issues/1078), so lair keeps them here.

Each constant is a pint quantity: a number with its unit. Multiply it into a
calculation and the units follow, or take the number in the unit you want with
``m_as``. The atmospheric constants are rounded textbook values; the SI
constants (:data:`Na`, :data:`kb`, :data:`h`, :data:`c`) are exact. Each page
says what its constant is for and how far the rounding is from the precise
value, so you can judge whether it matters for your work.

Examples
--------
The potential temperature exponent, ``Rd / cp``, has no unit left:

>>> from lair import constants
>>> kappa = constants.Rd / constants.cp
>>> round(float(kappa.m_as("dimensionless")), 4)
0.2856

A constant as a plain number in the unit you give, for code that does
not use pint:

>>> float(constants.R_earth.m_as("m"))
6371000.0

The scale height of the atmosphere, ``Rd * T / g``, at 250 K:

>>> from lair import units
>>> H = constants.Rd * (250 * units.K) / constants.g
>>> round(float(H.m_as("km")), 2)
7.32
"""

from lair import units

#: The Avogadro constant, 6.02214076e23 per mole.
#:
#: Exact: the 2019 redefinition of the SI fixed it. It is the number of
#: molecules in one mole of a substance. :data:`Rstar` is :data:`kb` times it.
#:
#: .. seealso:: :data:`kb`, :data:`Rstar`
#:
#: :meta hide-value:
Na = 6.02214076e23 * units("1 / mol")

#: The Boltzmann constant, 1.380649e-23 J/K.
#:
#: Exact: the 2019 redefinition of the SI fixed it. It is the gas constant of
#: one molecule, so the ideal gas law is ``p = N * kb * T / V`` for ``N``
#: molecules. :func:`lair.meteorology.ideal_gas_law` uses it when you give it a
#: number of molecules ``N``.
#:
#: .. seealso:: :data:`Rstar`, :data:`Na`, :func:`lair.meteorology.ideal_gas_law`
#:
#: :meta hide-value:
kb = 1.380649e-23 * units("J / K")

#: The density of liquid water, 997 kg/m^3.
#:
#: This is its density at about 25 degC (997.05 kg/m^3); at 0 degC it is
#: 999.84 and at 4 degC, where it is greatest, about 1000. Use it for rain or
#: cloud water near room temperature; it is not a function of temperature
#: here. The functions of lair do not use it.
#:
#: .. seealso:: :data:`c_l`, :data:`L`
#:
#: :meta hide-value:
rho_w = 997 * units("kg / m**3")

#: The dry adiabatic lapse rate, 9.8 K/km.
#:
#: How fast a rising parcel of dry air cools as it expands. The textbook
#: value is ``g / cp`` rounded to two digits. With the :data:`g` and
#: :data:`cp` of this module, ``g / cp`` is 9.76 K/km, so 9.8 is 0.4% above it:
#: a calculation that needs constants that agree with each other should use
#: ``g / cp``. The functions of lair do not use this constant.
#:
#: .. seealso:: :data:`g`, :data:`cp`, :func:`lair.meteorology.poisson`
#:
#: :meta hide-value:
Gamma_d = 9.8 * units("K / km")

#: The specific gas constant of dry air, 287.05 J/(kg K).
#:
#: The universal gas constant :data:`Rstar` divided by the molar mass of dry
#: air (28.9647 g/mol) is 287.055, so the value is that one rounded to the
#: hundredth. It sets the pressure, density and temperature of dry air through
#: the ideal gas law, and the exponent ``Rd / cp`` of Poisson's equation.
#: :func:`lair.meteorology.hypsometric`, :func:`~lair.meteorology.poisson`,
#: :func:`~lair.meteorology.inv_poisson` and
#: :func:`lair.pcaps.valleyheatdeficit` use it.
#:
#: .. seealso:: :data:`Rv`, :data:`cp`, :data:`epsilon`, :data:`Rstar`
#:
#: :meta hide-value:
Rd = 287.05 * units("J / kg / K")

#: The specific gas constant of water vapor, 461.5 J/(kg K).
#:
#: The universal gas constant :data:`Rstar` divided by the molar mass of water
#: (18.01528 g/mol) is 461.52, so this is that value rounded to 4 digits. It
#: gives :data:`epsilon`, which :func:`lair.meteorology.mixing_ratio` uses.
#:
#: .. seealso:: :data:`Rd`, :data:`epsilon`, :data:`Rstar`
#:
#: :meta hide-value:
Rv = 461.5 * units("J / kg / K")

#: The acceleration of gravity, 9.81 m/s^2.
#:
#: The standard value, fixed by definition, is 9.80665 m/s^2; 9.81 is 0.03%
#: larger. The real value varies by about 0.5% over the Earth's surface, with
#: latitude and height. :func:`lair.meteorology.hypsometric` uses it to
#: relate pressure and height.
#:
#: .. seealso:: :func:`lair.meteorology.hypsometric`, :data:`Rd`, :data:`Gamma_d`
#:
#: :meta hide-value:
g = 9.81 * units("m / s**2")

#: The specific heat of dry air at constant pressure, 1005 J/(kg K).
#:
#: The heat that warms 1 kg of dry air by 1 K while its pressure stays the
#: same. It changes very little over the temperatures of the troposphere, so
#: a constant is enough there. :func:`lair.meteorology.poisson` and
#: :func:`~lair.meteorology.inv_poisson` use ``Rd / cp`` as the exponent, and
#: :func:`lair.pcaps.valleyheatdeficit` uses it to turn a potential
#: temperature deficit into heat.
#:
#: .. seealso:: :data:`cv`, :data:`Rd`, :data:`Gamma_d`, :func:`lair.meteorology.poisson`
#:
#: :meta hide-value:
cp = 1005 * units("J / kg / K")

#: The specific heat of liquid water, 4186 J/(kg K).
#:
#: The heat that warms 1 kg of liquid water by 1 K. This is the value near
#: 15 degC; at 25 degC it is 4182, and it changes by about 1% between 0 and
#: 100 degC. The functions of lair do not use it.
#:
#: .. seealso:: :data:`cp`, :data:`rho_w`, :data:`L_f`
#:
#: :meta hide-value:
c_l = 4186 * units("J / kg / K")

#: The specific heat of dry air at constant volume, 718 J/(kg K).
#:
#: The heat that warms 1 kg of dry air by 1 K while its volume stays the same.
#: Mayer's relation ties it to :data:`cp`: ``cp - cv`` is :data:`Rd`, and
#: 1005 - 287.05 is 717.95, which rounds to this value. The functions of lair
#: do not use it.
#:
#: .. seealso:: :data:`cp`, :data:`Rd`
#:
#: :meta hide-value:
cv = 718 * units("J / kg / K")

#: The latent heat of fusion of water, 3.337e5 J/kg.
#:
#: The heat that melts 1 kg of ice at 0 degC. The value usually given is 3.3355e5
#: (333.55 kJ/kg); sources differ in the 4th digit, and this is 0.05% above that
#: one. The functions of lair do not use it, but :data:`L_s` is the sum of it
#: and :data:`L`.
#:
#: .. seealso:: :data:`L`, :data:`L_s`
#:
#: :meta hide-value:
L_f = 3.337e5 * units("J / kg")

#: The latent heat of vaporization of water, 2.501e6 J/kg.
#:
#: The heat that evaporates 1 kg of liquid water at 0 degC. It decreases as
#: the water is warmer: at the boiling point it is 2.257e6, so for air near
#: the surface the error is a few percent at most. The functions of lair do not use it.
#:
#: .. seealso:: :data:`L_f`, :data:`L_s`, :data:`c_l`
#:
#: :meta hide-value:
L = 2.501e6 * units("J / kg")

#: The latent heat of sublimation of water, 2.8347e6 J/kg.
#:
#: The heat that turns 1 kg of ice to vapor, without passing through liquid
#: water. It is :data:`L` plus :data:`L_f` (2.501e6 + 3.337e5), the heat to
#: melt the ice and then evaporate the water. The functions of lair do not use
#: it.
#:
#: .. seealso:: :data:`L`, :data:`L_f`
#:
#: :meta hide-value:
L_s = L + L_f

#: The Planck constant, 6.62607015e-34 J s.
#:
#: Exact: the 2019 redefinition of the SI fixed it. It relates the energy of a
#: photon to its frequency, ``E = h * nu``. The functions of lair do not use
#: it.
#:
#: .. seealso:: :data:`c`, :data:`kb`
#:
#: :meta hide-value:
h = 6.62607015e-34 * units("J * s")

#: The mean radius of the Earth, 6371 km.
#:
#: The radius of a sphere with the same volume as the Earth; the IUGG mean
#: radius is 6371.0088 km. The Earth is not a sphere (its equatorial radius is
#: 6378.137 km and its polar radius is 6356.752 km), so a distance computed with
#: this value is right to about 0.3% at most. The unit is kilometers, unlike
#: the other constants, which are in SI base units: use ``R_earth.m_as("m")``
#: for meters. The functions of lair do not use it.
#:
#: .. seealso:: :data:`g`
#:
#: :meta hide-value:
R_earth = 6371 * units("km")

#: The ratio of the gas constants of dry air and water vapor, ``Rd / Rv``.
#:
#: It is dimensionless and equal to 0.62199, calculated from the rounded
#: :data:`Rd` and :data:`Rv`; the ratio of the molar masses of water and dry
#: air is 0.62197. It converts a vapor pressure to a mixing ratio:
#: :func:`lair.meteorology.mixing_ratio` uses it.
#:
#: .. seealso:: :data:`Rd`, :data:`Rv`, :func:`lair.meteorology.mixing_ratio`
#:
#: :meta hide-value:
epsilon = Rd / Rv

#: The speed of light in a vacuum, 299792458 m/s.
#:
#: Exact: the meter is defined from it. The functions of lair do not use it.
#:
#: .. seealso:: :data:`h`
#:
#: :meta hide-value:
c = 299792458 * units("m / s")

#: The Stefan-Boltzmann constant, 5.67e-8 W/(m^2 K^4).
#:
#: The power a black body radiates per square meter is ``sigma * T**4``. The
#: precise value is 5.670374419e-8 (it follows from :data:`kb`, :data:`h` and
#: :data:`c`), so this one is 0.007% low. The functions of lair do not use it.
#:
#: .. seealso:: :data:`kb`, :data:`h`, :data:`c`
#:
#: :meta hide-value:
sigma = 5.67e-8 * units("W / m**2 / K**4")

#: The universal gas constant, 8.314462618 J/(mol K).
#:
#: It is :data:`kb` times :data:`Na`, so it is exact. It is the gas constant for
#: one mole of any ideal gas; dividing it by a molar mass gives the specific gas
#: constant of that gas, as :data:`Rd` and :data:`Rv` are. :func:`lair.meteorology.ideal_gas_law`
#: uses it when you give it an amount of substance ``n``.
#:
#: .. seealso:: :data:`Rd`, :data:`Rv`, :data:`kb`, :data:`Na`, :func:`lair.meteorology.ideal_gas_law`
#:
#: :meta hide-value:
Rstar = kb * Na
