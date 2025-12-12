"""
Meteorological calculations.

Inspired by AOS 330 at UW-Madison with Grant Petty.

.. note::
    It would be nice to be able to wrap these funtions with `pint` - however,
    because I input both numpy and xarray arrays, this will not work.
    Waiting for https://github.com/xarray-contrib/pint-xarray/pull/143
    For now, we will assume all inputs are in SI units.
"""

import numpy as np

from lair.constants import Rstar, Rd, kb, Na, cp, g, epsilon
from lair import units


#: Standard Atmosphere
standard: dict[str, float] = {
    'T': 288.15 * units('K'),
    'p': 1013.25 * units('hPa'),
    'rho': 1.225 * units('kg / m**3'),
    'z': 0 * units('m')
}

#############
# Functions #
#############

def ideal_gas_law(solve_for, p=None, V=None, T=None, 
                  m=None, n=None, N=None,
                  rho=None, alpha=None, R=None):
    """
    Ideal gas law equation solver.
    Solver attempts to solve for the specified variable using the following
    forms of the ideal gas law:

    pV = nR*T
    pV = mRT
    pV = NkbT
    p = ρRT
    pα = RT

    Input variables must be able to solve for the desired variable using the
    above equations without intermediate steps. All inputs must be in SI.

    p : pressure (Pa)
    V : volume (m^3)
    T : temperature (K)
    m : mass (kg)
    n : moles (mol)
    N : number of molecules (#)
    ρ : density (kg/m^3)
    α : specific volume (m^3/kg)
    R : specific gas constant (J/kg/K)

    Can be used to solve for pressure, volume, temperature, density, mass, 
    moles, or number of molecules.
    """

    if solve_for in ['pressure', 'pres', 'p']:
        if not V:
            rho = rho or 1/alpha
            x = rho * R * T
        else:
            if n:
                x = n * Rstar * T / V
                x = m * R / V
            else:
                x = N * kb * T / V
    elif solve_for in ['volume', 'vol', 'V']:
        if n:
            x = n * Rstar * T / p
        elif m:
            x = m * R / p
        else:
            x = N * kb * T / p
    elif solve_for in ['temperature', 'temp', 'T']:
        if not V:
            rho = rho or 1/alpha
            x = p / (rho * R)
        else:
            if n:
                x = p * V / (n * Rstar)
            elif m:
                x = p * V / (m * R)
            else:
                x = p * V / (N * kb)
    elif solve_for in ['density', 'rho']:
        x = p / (R * T)
    elif solve_for in ['mass', 'm']:
        x = p * V / (R * T)
    elif solve_for in ['moles', 'n']:
        x = p * V / (Rstar * T)
    elif solve_for in ['number', 'N']:
        x = p * V / (kb * T)
    else:
        raise ValueError('Invalid solve_for')

    return x


def hypsometric(Tv=None, p1=None, p2=None,
                Z1=None, Z2=None, deltaz=None):
    """
    Hyposometric equation solver.

    Z2 - Z1 = Rd * Tv * ln(p1/p2) / g

    Input variables must be able to solve for the desired variable using the
    above equation without intermediate steps. All inputs must be in SI.

    Tv : mean virtual temperature of layer (K)
    p1 : pressure at bottom of layer (Pa)
    p2 : pressure at top of layer (Pa)
    Z1 : geopotential height at bottom of layer (m)
    Z2 : geopotential height at top of layer (m)
    deltaz : thickness of layer (m)

    Can be used to solve for any of the variables in the equation or deltaz.
    """
    if deltaz or (Z1 and Z2):
        deltaz = deltaz or Z2 - Z1

        if Tv is None:
            return deltaz * g / (Rd * np.log(p1/p2))
        elif p1 is None:
            return p2 * np.exp(deltaz * g / (Rd * Tv))
        elif p2 is None:
            return p1 * np.exp(-deltaz * g / (Rd * Tv))

    if not any([deltaz, Z1, Z2]):
        return Rd * Tv * np.log(p1/p2) / g
    elif Z1 is None:
        return Z2 - Rd * Tv * np.log(p1/p2) / g
    elif Z2 is None:
        return Z1 + Rd * Tv * np.log(p1/p2) / g

    raise ValueError('Invalid input combination')


def virt_T(T: float, q: float) -> float:
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
    """
    return T * (1 + 0.61 * q)


def poisson(T: float, p: float, p0: float = 1e5) -> float:
    """
    Calculate the potential temperature. (Poission's equation)

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
    """
    return T * (p0/p)**(Rd/cp)


def inv_poisson(p: float, theta: float, p0: float = 1e5) -> float:
    """
    Calculate the temperature from potential temperature. (Inverse Poission's equation)

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
    """
    return theta * (p/p0)**(Rd/cp)


def sat_vapor_pres(T: float) -> float:
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
    """
    return 2.53e11 * np.exp(-5420/T)


def sat_vapor_pres_ice(T: float) -> float:
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
    """
    return 3.41e11 * np.exp(-6130/T)


def mixing_ratio(e: float, p: float) -> float:
    """
    Calculate the mixing ratio.

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
    """
    return epsilon * e / p


def T_from_e(e: float) -> float: #Pa
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
    """
    return -5420/np.log(e/2.53e11) #K
