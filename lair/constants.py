"""
Mathematical, physical & scienfitic constants.

All in scientific units.
"""

from lair.units import J, kg, m, sec, K, km, W


Na = 6.02214076e23            #: Avogadro constant [1/mol]
kb = 1.380649e-23 *J/K        #: Boltzmann constant [J/K]
rho_w = 997 *kg/m**3          #: Density of water [kg/m^3]
Gamma_d = 9.8 *K/km           #: Dry adiabatic lapse rate [K/km]
Rd = 287.05 *J/kg/K           #: Gas constant for dry air [J/kg/K]
Rv = 461.5 *J/kg/K            #: Gas constant for water vapor [J/kg/K]
g = 9.81 *m/sec**2            #: Gravitational acceleration [m/sec^2]
cp = 1005 *J/kg/K             #: Heat capacity of dry air [J/kg/K]
c_l = 4186 *J/kg/K            #: Heat capacity of liquid water [J/kg/K]
cv = 718 *J/kg/K              #: Heat capacity of water vapor [J/kg/K]
L_f = 3.337e5 *J/kg           #: Latent heat of fusion [J/kg]
L = 2.501e6 *J/kg             #: Latent heat of vaporization [J/kg]
L_s = L + L_f                 #: Latent heat of sublimation [J/kg]
h = 6.62607015e-34 *J*sec     #: Planck constant [J*sec]
R_earth = 6371 * km           #: Radius of the Earth [km]
epsilon = Rd/Rv               #: Ratio of gas constants [#]
c = 299792458 *m/sec          #: Speed of light [m/sec]
sigma = 5.67e-8 *W/m**2/K**4  #: Stefan-Boltzmann constant [W/m^2/K^4]
Rstar = kb * Na               #: Universial gas constant TODO
