"""
lair.air.meteorology
~~~~~~~~~~~~~~~~~~~~

Module for meteorological calculations.

Inspired by AOS 330 at UW-Madison with Grant Petty.
"""

import numpy as np

from lair.constants import Rstar, Rd, kb, Na, cp, g, epsilon
from lair.units import K, Pa, hPa, kg, m, sec, J, W, C2K

# Standard Atmosphere
T_stnd   = C2K(15) *K
p_stnd   = 1013.25 *hPa
rho_stnd = 1.225 *kg/m**3
z_stnd   = 0 *m


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


def virt_T(T, q):  # K, kg kg-1
    return T * (1 + 0.61 * q)

def poisson(T, p, p0 = 1000*hPa): #K, Pa
    return T * (p0/p)**(Rd/cp) #theta (potential temp) K

def inv_poisson(p, theta, p0 = 1000*hPa): #Pa, K
    return theta * (p/p0)**(Rd/cp)

def sat_vapor_pres(T): #K
    return 2.53e11 * np.exp(-5420/T) #Pa

def sat_vapor_pres_ice(T): #K
    return 3.41e11 * np.exp(-6130/T) #Pa

def mixing_ratio(e, p): #Pa
    return epsilon * e / p #kg/kg

def T_from_e(e): #Pa
    return -5420/np.log(e/2.53e11) #K
