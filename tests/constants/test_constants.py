"""Tests for lair.constants.

The constants are pint quantities (carrying SI-ish units). We check both the
numeric magnitude and the dimensionality, plus the derived relationships.
"""

import pytest

from lair import constants as c
from lair import units


@pytest.mark.parametrize(
    "name, magnitude, unit",
    [
        ("Na", 6.02214076e23, "1 / mol"),
        ("kb", 1.380649e-23, "J / K"),
        ("rho_w", 997, "kg / m**3"),
        ("Gamma_d", 9.8, "K / km"),
        ("Rd", 287.05, "J / kg / K"),
        ("Rv", 461.5, "J / kg / K"),
        ("g", 9.81, "m / s**2"),
        ("cp", 1005, "J / kg / K"),
        ("h", 6.62607015e-34, "J * s"),
        ("R_earth", 6371, "km"),
        ("c", 299792458, "m / s"),
        ("sigma", 5.67e-8, "W / m**2 / K**4"),
    ],
)
def test_constant_values_and_units(name, magnitude, unit):
    """Each constant has the expected magnitude and is dimensionally correct."""
    const = getattr(c, name)
    assert const.magnitude == pytest.approx(magnitude)
    # Same dimensionality as the documented unit (allows equivalent spellings).
    assert const.dimensionality == (1 * units(unit)).dimensionality


def test_epsilon_is_ratio_of_gas_constants():
    """epsilon = Rd / Rv, dimensionless, ~0.622."""
    assert c.epsilon.check("[]")  # dimensionless
    assert c.epsilon.magnitude == pytest.approx((c.Rd / c.Rv).magnitude)
    assert c.epsilon.magnitude == pytest.approx(0.622, abs=1e-3)


def test_latent_heat_of_sublimation_is_sum():
    """L_s = L + L_f (vaporization + fusion)."""
    assert c.L_s.magnitude == pytest.approx((c.L + c.L_f).magnitude)


def test_universal_gas_constant_from_kb_na():
    """Rstar = kb * Na ~ 8.314 J/K/mol."""
    assert c.Rstar.magnitude == pytest.approx(8.314, abs=1e-3)
    assert c.Rstar.dimensionality == (1 * units("J / K / mol")).dimensionality
