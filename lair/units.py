"""
Units and some conversions.
"""

############
# SI Units #
############

sec  = 1.0  #: time
m    = 1.0  #: length
kg   = 1.0  #: mass
A    = 1.0  #: current
K    = 1.0  #: temperature
mol  = 1.0  #: amount of substance
cd   = 1.0  #: luminous intensity

# SI Derived Units
rad  = m / m  #: angle
sr   = m**2 / m**2  #: solid angle
N    = kg*m/sec**2  #: force
Pa   = kg/m/sec**2  #: pressure (N/m^2 = J/m^3)
J    = kg*m**2/sec**2  #: energy (N*m = Pa*m^3)
W = kg*m**2/sec**3  #: power (J/sec)

####################
# Conversion Units #
####################

# Time

minute = 60 *sec  #: minute
hour   = 60 *minute  #: hour

# Length

cm     = 1.0e-2*m  #: centimeter
mm     = 1.0e-3*m  #: millimeter
km     = 1000 *m  #: kilometer
foot   = m /3.28  #: foot
mile   = 5280 *foot  #: mile
nautical_mile = 1852 *m  #: nautical mile

# Mass

gram   = 1.0e-3*kg  #: gram
mg     = 1.0e-3*gram  #: milligram
ug     = 1.0e-6*gram  #: microgram

# Pressure

hPa    = 100 *Pa  #: hectopascal
atm    = 101325 *Pa  #: atmosphere

# Energy

MJ    = 1.0e6 * J  #: megajoule

# Area

m2     = m**2  #: square meter
cm2    = cm**2  #: square centimeter
acre   = 43560 *foot**2  #: acre
ha     = 10000 *m**2  #: hectare

# Volume

cc     = cm**3  #: cubic centimeter
ml     = cc  #: milliliter
liter  = 1000 *ml  #: liter

# Speed

kt    = nautical_mile/hour  #: knot (kn)
kmh    = km/hour  #: kilometer/hour
mph    = mile/hour  #: mile/hour


class C:
    """
    Celsius temperature.
    """
    def __mul__(self, other):
        return other + 273.15
    __rmul__ = __mul__

    def __div__(self, other):
        return other - 273.15
    __rdiv__ = __div__

def K2C(K: float) -> float:
    """
    Convert Kelvin to Celsius.
    """
    return K - 273.15

def C2K(C: float) -> float:
    """
    Convert Celsius to Kelvin.
    """
    return C + 273.15

def F2C(F: float) -> float:
    """
    Convert Fahrenheit to Celsius.
    """
    return (F-32)*(5/9)

def C2F(C: float) -> float:
    """
    Convert Celsius to Fahrenheit.
    """
    return C*(9/5) + 32
