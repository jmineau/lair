"""
lair.units
~~~~~~~~~~

Module for storing units.
"""

############
# SI Units #
############

sec  = 1.0  # time
m    = 1.0  # length
kg   = 1.0  # mass
A    = 1.0  # current
K    = 1.0  # temperature
mol  = 1.0  # amount of substance
cd   = 1.0  # luminous intensity

# SI Derived Units
rad  = m / m  # angle
sr   = m**2 / m**2  # solid angle
N    = kg*m/sec**2  # force
Pa   = kg/m/sec**2  # pressure (N/m^2 = J/m^3)
J    = kg*m**2/sec**2  # energy (N*m = Pa*m^3)
W = kg*m**2/sec**3  # power (J/sec)

####################
# Conversion Units #
####################

# Time

minute = 60 *sec
hour   = 60 *minute

# Length

cm     = 1.0e-2*m
mm     = 1.0e-3*m
km     = 1000 *m
foot   = m /3.28
mile   = 5280 *foot
nautical_mile = 1852 *m

# Mass

gram   = 1.0e-3*kg
mg     = 1.0e-3*gram
ug     = 1.0e-6*gram

# Pressure

hPa    = 100 *Pa
atm    = 101325 *Pa

# Energy

MJ    = 1.0e6 * J

# Area

m2     = m**2
cm2    = cm**2
acre   = 43560 *foot**2
ha     = 10000 *m**2

# Volume

cc     = cm**3
ml     = cc
liter  = 1000 *ml

# Speed

kt    = nautical_mile/hour  # knot (kn)
kmh    = km/hour
mph    = mile/hour


class C:
    def __mul__(self, other):
        return other + 273.15
    __rmul__ = __mul__

    def __div__(self, other):
        return other - 273.15
    __rdiv__ = __div__

def K2C(K):
    return K - 273.15

def C2K(C):
    return C + 273.15

def F2C(F):
    return (F-32)*(5/9)

def C2F(C):
    return C*(9/5) + 32