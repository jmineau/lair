"""
Meteorological calculations.

Inspired by AOS 330 at UW-Madison with Grant Petty.

Units contract: **plain SI in, plain SI out.** Inputs are floats, numpy arrays
or xarray DataArrays in SI units (K, Pa, m, kg, mol, m^3, J/kg/K, ...), and the
results are the same kinds of plain numbers in SI units (e.g. ``hypsometric``
returns meters, ``ideal_gas_law("p", ...)`` returns Pa). The physical constants
are used as plain SI floats internally.

pint Quantities (and pint-quantified xarray DataArrays) are accepted for
convenience: they are converted to SI magnitudes on the way in (so
``850 * units("hPa")`` becomes ``85000.0`` and ``0 degC`` becomes ``273.15``),
and the result is still a plain SI number, never a Quantity. Attach units to the
result yourself if you want them.

.. note::
    It would be nice to wrap these functions with `pint` end to end, but numpy
    and xarray inputs make that awkward.
    See https://github.com/xarray-contrib/pint-xarray/pull/143
"""

import functools
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import numpy as np
import pint
import xarray as xr

from lair import constants

#: Scalars, numpy arrays or xarray DataArrays in SI units, as inputs and results
Numeric = Any

# Physical constants as plain SI floats
Rstar = float(constants.Rstar.m_as("J / K / mol"))
Rd = float(constants.Rd.m_as("J / kg / K"))
kb = float(constants.kb.m_as("J / K"))
cp = float(constants.cp.m_as("J / kg / K"))
g = float(constants.g.m_as("m / s**2"))
epsilon = float(constants.epsilon.m_as("dimensionless"))


#: Standard Atmosphere, in SI: T [K], p [Pa], rho [kg/m^3], z [m]
standard: dict[str, float] = {
    "T": 288.15,
    "p": 101325.0,
    "rho": 1.225,
    "z": 0.0,
}

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _to_si(x: Any) -> Any:
    """Return ``x`` as a plain SI magnitude if it carries pint units."""
    # pint's stubs type pint.Quantity as a TypeVar; at runtime it is the class
    # every registry's Quantity subclasses
    # pyrefly: ignore[invalid-argument]
    if isinstance(x, pint.Quantity):
        return x.to_base_units().magnitude
    # pyrefly: ignore[invalid-argument]
    if isinstance(x, xr.DataArray) and isinstance(x.data, pint.Quantity):
        return x.copy(data=x.data.to_base_units().magnitude)
    return x


def _si_inputs(func: Callable[_P, _R]) -> Callable[_P, _R]:
    """Convert any pint Quantity arguments of ``func`` to SI magnitudes."""

    @functools.wraps(func)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        si_args = [_to_si(a) for a in args]
        si_kwargs = {k: _to_si(v) for k, v in kwargs.items()}
        # Same arguments, only their units stripped
        # pyrefly: ignore[invalid-param-spec]
        return func(*si_args, **si_kwargs)

    return wrapper


#############
# Functions #
#############


@_si_inputs
def ideal_gas_law(
    solve_for: str,
    p: Numeric = None,
    V: Numeric = None,
    T: Numeric = None,
    m: Numeric = None,
    n: Numeric = None,
    N: Numeric = None,
    rho: Numeric = None,
    alpha: Numeric = None,
    R: Numeric = None,
) -> Numeric:
    """
    Solve the ideal gas law for one variable.

    The solver uses one of these forms of the law, where R* is the universal
    gas constant and kb Boltzmann's constant::

        pV = nR * T
        pV = mRT
        pV = NkbT
        p = ρRT
        pα = RT

    Give the inputs one form needs to solve for the variable directly, with no
    intermediate step. Inputs and the result are in SI (pint Quantities are
    converted to SI first; see the module docstring).

    Parameters
    ----------
    solve_for : str
        The variable to solve for: ``'p'`` (or ``'pressure'``, ``'pres'``),
        ``'V'`` (``'volume'``, ``'vol'``), ``'T'`` (``'temperature'``,
        ``'temp'``), ``'rho'`` (``'density'``), ``'m'`` (``'mass'``), ``'n'``
        (``'moles'``) or ``'N'`` (``'number'``).
    p : Numeric, optional
        Pressure (Pa).
    V : Numeric, optional
        Volume (m^3).
    T : Numeric, optional
        Temperature (K).
    m : Numeric, optional
        Mass (kg).
    n : Numeric, optional
        Amount of substance (mol).
    N : Numeric, optional
        Number of molecules.
    rho : Numeric, optional
        Density (kg/m^3).
    alpha : Numeric, optional
        Specific volume (m^3/kg).
    R : Numeric, optional
        Specific gas constant (J/kg/K): 287.05 for dry air.

    Returns
    -------
    Numeric
        The variable asked for, in SI.

    Raises
    ------
    ValueError
        If ``solve_for`` is not one of the names above.

    See Also
    --------
    hypsometric : Thickness of a layer between two pressures.

    Examples
    --------
    The density of dry air at sea level in the standard atmosphere:

    >>> from lair.meteorology import ideal_gas_law
    >>> round(float(ideal_gas_law("rho", p=101325.0, T=288.15, R=287.05)), 3)
    1.225
    """
    # Compare against None (not truthiness) so array inputs and zero values
    # (e.g. n=0) work
    if solve_for in ["pressure", "pres", "p"]:
        if V is None:
            if rho is None:
                rho = 1 / alpha
            x = rho * R * T
        else:
            if n is not None:
                x = n * Rstar * T / V
            elif m is not None:
                x = m * R * T / V
            else:
                x = N * kb * T / V
    elif solve_for in ["volume", "vol", "V"]:
        if n is not None:
            x = n * Rstar * T / p
        elif m is not None:
            x = m * R * T / p
        else:
            x = N * kb * T / p
    elif solve_for in ["temperature", "temp", "T"]:
        if V is None:
            if rho is None:
                rho = 1 / alpha
            x = p / (rho * R)
        else:
            if n is not None:
                x = p * V / (n * Rstar)
            elif m is not None:
                x = p * V / (m * R)
            else:
                x = p * V / (N * kb)
    elif solve_for in ["density", "rho"]:
        x = p / (R * T)
    elif solve_for in ["mass", "m"]:
        x = p * V / (R * T)
    elif solve_for in ["moles", "n"]:
        x = p * V / (Rstar * T)
    elif solve_for in ["number", "N"]:
        x = p * V / (kb * T)
    else:
        raise ValueError("Invalid solve_for")

    return x


@_si_inputs
def hypsometric(
    Tv: Numeric = None,
    p1: Numeric = None,
    p2: Numeric = None,
    Z1: Numeric = None,
    Z2: Numeric = None,
    deltaz: Numeric = None,
) -> Numeric:
    """
    Solve the hypsometric equation for one variable.

    The equation is ``Z2 - Z1 = Rd * Tv * ln(p1 / p2) / g``. Give all but one
    of its variables (``deltaz`` stands for ``Z2 - Z1``) and the missing one is
    returned: the thickness when neither height nor ``deltaz`` is given. Inputs
    and the result are in SI (pint Quantities are converted to SI first; see
    the module docstring).

    Parameters
    ----------
    Tv : Numeric, optional
        Mean virtual temperature of the layer (K).
    p1 : Numeric, optional
        Pressure at the bottom of the layer (Pa).
    p2 : Numeric, optional
        Pressure at the top of the layer (Pa).
    Z1 : Numeric, optional
        Geopotential height of the bottom of the layer (m).
    Z2 : Numeric, optional
        Geopotential height of the top of the layer (m).
    deltaz : Numeric, optional
        Thickness of the layer (m).

    Returns
    -------
    Numeric
        The missing variable, in SI.

    Raises
    ------
    ValueError
        If the inputs do not determine one variable.

    See Also
    --------
    virt_T : The virtual temperature ``Tv``.

    Examples
    --------
    The thickness of the 1000 to 850 hPa layer at a mean virtual temperature
    of 280 K, and the pressure 1300 m above the 1000 hPa level:

    >>> from lair.meteorology import hypsometric
    >>> round(float(hypsometric(Tv=280.0, p1=100000.0, p2=85000.0)), 1)
    1331.5
    >>> round(float(hypsometric(Tv=280.0, p1=100000.0, deltaz=1300.0)))
    85328
    """
    # Compare against None (not truthiness) so a surface height of 0 m and
    # array inputs work
    if deltaz is not None or (Z1 is not None and Z2 is not None):
        if deltaz is None:
            deltaz = Z2 - Z1

        if Tv is None:
            return deltaz * g / (Rd * np.log(p1 / p2))
        elif p1 is None:
            return p2 * np.exp(deltaz * g / (Rd * Tv))
        elif p2 is None:
            return p1 * np.exp(-deltaz * g / (Rd * Tv))

    elif Z1 is None and Z2 is None:
        return Rd * Tv * np.log(p1 / p2) / g
    elif Z1 is None:
        return Z2 - Rd * Tv * np.log(p1 / p2) / g
    elif Z2 is None:
        return Z1 + Rd * Tv * np.log(p1 / p2) / g

    raise ValueError("Invalid input combination")


@_si_inputs
def virt_T(T: Numeric, q: Numeric) -> Numeric:
    """
    Calculate the virtual temperature.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    q : float
        Specific humidity in kg/kg.

    Returns
    -------
    float
        Virtual temperature in Kelvin.

    See Also
    --------
    hypsometric : Uses the mean virtual temperature of a layer.

    Examples
    --------
    >>> from lair.meteorology import virt_T
    >>> round(float(virt_T(293.15, 0.01)), 2)
    294.94
    """
    return T * (1 + 0.61 * q)


@_si_inputs
def poisson(T: Numeric, p: Numeric, p0: Numeric = 1e5) -> Numeric:
    """
    Calculate the potential temperature (Poisson's equation).

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    p : float
        Pressure in Pascals.
    p0 : float, optional
        Reference pressure in Pascals. Default is 1000 hPa.

    Returns
    -------
    float
        Potential temperature in Kelvin.

    See Also
    --------
    inv_poisson : The temperature from the potential temperature.

    Examples
    --------
    Air at 0 degrees Celsius and 850 hPa, in SI or as pint quantities (the
    result is in kelvin either way):

    >>> from lair import units
    >>> from lair.meteorology import poisson
    >>> round(float(poisson(T=273.15, p=85000.0)), 2)
    286.13
    >>> round(float(poisson(T=0 * units.degC, p=850 * units.hPa)), 2)
    286.13
    """
    return T * (p0 / p) ** (Rd / cp)


@_si_inputs
def inv_poisson(p: Numeric, theta: Numeric, p0: Numeric = 1e5) -> Numeric:
    """
    Calculate the temperature from the potential temperature.

    This is Poisson's equation solved for the temperature.

    Parameters
    ----------
    p : float
        Pressure in Pascals.
    theta : float
        Potential temperature in Kelvin.
    p0 : float, optional
        Reference pressure in Pascals. Default is 1000 hPa.

    Returns
    -------
    float
        Temperature in Kelvin.

    See Also
    --------
    poisson : The potential temperature from the temperature.

    Examples
    --------
    >>> from lair.meteorology import inv_poisson
    >>> round(float(inv_poisson(p=85000.0, theta=286.67)), 2)
    273.67
    """
    return theta * (p / p0) ** (Rd / cp)


@_si_inputs
def sat_vapor_pres(T: Numeric) -> Numeric:
    """
    Calculate the saturation vapor pressure.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Saturation vapor pressure in Pascals.

    See Also
    --------
    T_from_e : Its inverse, the temperature at a vapor pressure.
    sat_vapor_pres_ice : The same over ice.
    mixing_ratio : The mixing ratio at a vapor pressure.

    Examples
    --------
    >>> from lair.meteorology import sat_vapor_pres
    >>> round(float(sat_vapor_pres(293.15)), 1)
    2363.3
    """
    return 2.53e11 * np.exp(-5420 / T)


@_si_inputs
def sat_vapor_pres_ice(T: Numeric) -> Numeric:
    """
    Calculate the saturation vapor pressure over ice.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.

    Returns
    -------
    float
        Saturation vapor pressure over ice in Pascals.

    See Also
    --------
    sat_vapor_pres : The same over liquid water.
    """
    # Petty: e_si = 3.41e12 Pa exp(-6130 K / T); ~611 Pa at the triple point
    return 3.41e12 * np.exp(-6130 / T)


@_si_inputs
def mixing_ratio(e: Numeric, p: Numeric) -> Numeric:
    """
    Calculate the mixing ratio.

    w = ε e / (p - e), with ε = Rd / Rv (exact; the common approximation
    ε e / p is about 1-3% low near the surface).

    Parameters
    ----------
    e : float
        Vapor pressure in Pascals.
    p : float
        Pressure in Pascals.

    Returns
    -------
    float
        Mixing ratio in kg/kg.

    See Also
    --------
    sat_vapor_pres : The vapor pressure at saturation.

    Examples
    --------
    The saturation mixing ratio at 20 degrees Celsius and sea-level pressure:

    >>> from lair.meteorology import mixing_ratio, sat_vapor_pres
    >>> round(float(mixing_ratio(sat_vapor_pres(293.15), 101325.0)), 5)
    0.01485
    """
    return epsilon * e / (p - e)


@_si_inputs
def T_from_e(e: Numeric) -> Numeric:  # Pa
    """
    Calculate the temperature from vapor pressure.

    Parameters
    ----------
    e : float
        Vapor pressure in Pascals.

    Returns
    -------
    float
        Temperature in Kelvin.

    See Also
    --------
    sat_vapor_pres : Its inverse.
    """
    return -5420 / np.log(e / 2.53e11)  # K
