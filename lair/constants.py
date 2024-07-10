"""
lair.constants
~~~~~~~~~~~~~~

Module for storing constants.
Constants are s
"""

from lair.units import J, kg, m, sec, K, km, W

R_earth = 6371 * km # radius of the Earth
c       = 299792458 *m/sec  # speed of light
h       = 6.62607015e-34 *J*sec  # Planck constant
kb      = 1.380649e-23 *J/K  # Boltzmann constant
sigma   = 5.67e-8 *W/m**2/K**4  # Stefan-Boltzmann constant
Na      = 6.02214076e23  # Avogadro constant
Rstar   = kb * Na  # universal gas constant
Rv      = 461.5 *J/kg/K  # gas constant water vapor
Rd      = 287.05 *J/kg/K  # gas constant dry air
epsilon = Rd/Rv  # ratio of gas constants
cv      = 718 *J/kg/K  # heat capacity water vapor
cp      = 1005 *J/kg/K  # heat capacity dry air
c_l     = 4186 *J/kg/K #heat capacity liquid water
g       = 9.81 *m/sec**2
L       = 2.501e6 *J/kg # latent heat of vaporization
L_f     = 3.337e5 *J/kg # latent heat of fusion
L_s     = L + L_f  # latent heat of sublimation
Gamma_d = 9.8 *K/km  # dry adiabatic lapse rate
rho_w = 997 *kg/m**3 # density of water