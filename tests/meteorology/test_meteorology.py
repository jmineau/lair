"""Tests for lair.meteorology.

All functions assume SI inputs (the module deliberately does not wrap with pint;
see its docstring). Some return plain floats, some return numpy floats, and a
few that combine with pint constants return pint quantities — the tests handle
each accordingly.
"""

import numpy as np
import pytest

from lair import meteorology as met


def _mag(x):
    """Magnitude of x whether it's a pint Quantity or a plain number."""
    return x.magnitude if hasattr(x, "magnitude") else x


class TestVirtualTemperature:
    def test_dry_air_unchanged(self):
        # With zero specific humidity, Tv == T.
        assert met.virt_T(288.0, 0.0) == pytest.approx(288.0)

    def test_moist_air_is_warmer(self):
        assert met.virt_T(300.0, 0.01) == pytest.approx(300.0 * (1 + 0.61 * 0.01))
        assert met.virt_T(300.0, 0.01) > 300.0


class TestPoisson:
    def test_potential_temp_at_reference_equals_temp(self):
        # At p == p0 the exponent term is 1, so theta == T.
        assert met.poisson(280.0, 1e5) == pytest.approx(280.0)

    def test_potential_temp_above_surface_is_warmer(self):
        # Lower pressure aloft -> potential temperature exceeds temperature.
        assert met.poisson(280.0, 8e4) > 280.0

    def test_inverse_round_trips(self):
        theta = met.poisson(295.0, 7e4)
        assert met.inv_poisson(7e4, theta) == pytest.approx(295.0)


class TestSaturationVaporPressure:
    def test_increases_with_temperature(self):
        assert met.sat_vapor_pres(300.0) > met.sat_vapor_pres(280.0)

    def test_known_value(self):
        # 2.53e11 * exp(-5420 / 300)
        expected = 2.53e11 * np.exp(-5420 / 300.0)
        assert met.sat_vapor_pres(300.0) == pytest.approx(expected)

    def test_ice_below_liquid_at_subfreezing(self):
        # Over ice the saturation vapor pressure is lower than over water.
        assert met.sat_vapor_pres_ice(260.0) < met.sat_vapor_pres(260.0)

    def test_T_from_e_inverts_sat_vapor_pres(self):
        e = met.sat_vapor_pres(295.0)
        assert met.T_from_e(e) == pytest.approx(295.0)


class TestMixingRatio:
    def test_proportional_to_vapor_pressure(self):
        # w = epsilon * e / p  (epsilon ~ 0.622)
        w = met.mixing_ratio(2000.0, 1e5)
        assert _mag(w) == pytest.approx(0.622 * 2000.0 / 1e5, rel=1e-3)


class TestIdealGasLaw:
    def test_density_form(self):
        # rho = p / (R T)
        rho = met.ideal_gas_law("density", p=1e5, R=287.05, T=300.0)
        assert _mag(rho) == pytest.approx(1e5 / (287.05 * 300.0))

    def test_temperature_from_density(self):
        T = met.ideal_gas_law("temp", p=1e5, rho=1.2, R=287.05)
        assert _mag(T) == pytest.approx(1e5 / (1.2 * 287.05))

    def test_invalid_solve_for_raises(self):
        with pytest.raises(ValueError):
            met.ideal_gas_law("entropy", p=1e5, T=300.0)

    def test_mass_form(self):
        # m = p V / (R T)
        m = met.ideal_gas_law("mass", p=1e5, V=1.0, R=287.05, T=300.0)
        assert _mag(m) == pytest.approx(1e5 / (287.05 * 300.0))

    def test_moles_form(self):
        # n = p V / (R* T)
        from lair.constants import Rstar

        n = met.ideal_gas_law("moles", p=1e5, V=1.0, T=300.0)
        assert _mag(n) == pytest.approx(1e5 / (_mag(Rstar) * 300.0))

    def test_number_form(self):
        # N = p V / (kb T)
        from lair.constants import kb

        N = met.ideal_gas_law("number", p=1e5, V=1.0, T=300.0)
        assert _mag(N) == pytest.approx(1e5 / (_mag(kb) * 300.0))

    def test_volume_from_moles(self):
        from lair.constants import Rstar

        V = met.ideal_gas_law("volume", p=1e5, n=1.0, T=300.0)
        assert _mag(V) == pytest.approx(_mag(Rstar) * 300.0 / 1e5)

    def test_volume_from_mass(self):
        # V = m R T / p
        V = met.ideal_gas_law("volume", p=1e5, m=1.0, R=287.05, T=300.0)
        assert _mag(V) == pytest.approx(287.05 * 300.0 / 1e5)

    def test_pressure_from_moles_and_volume(self):
        # p = n R* T / V
        from lair.constants import Rstar

        p = met.ideal_gas_law("pressure", V=1.0, n=1.0, T=300.0)
        assert _mag(p) == pytest.approx(_mag(Rstar) * 300.0)

    def test_pressure_from_mass_and_volume(self):
        # p = m R T / V
        p = met.ideal_gas_law("pressure", V=1.0, m=1.0, R=287.05, T=300.0)
        assert _mag(p) == pytest.approx(287.05 * 300.0)

    def test_temperature_from_volume_and_moles(self):
        from lair.constants import Rstar

        T = met.ideal_gas_law("temperature", p=1e5, V=1.0, n=1.0)
        assert _mag(T) == pytest.approx(1e5 / _mag(Rstar))


class TestHypsometric:
    def test_thickness_positive_for_decreasing_pressure(self):
        # Solve for layer thickness given Tv and bounding pressures.
        deltaz = met.hypsometric(Tv=288.0, p1=1e5, p2=9e4)
        assert _mag(deltaz) == pytest.approx(
            287.05 * 288.0 * np.log(1e5 / 9e4) / 9.81, rel=1e-6
        )

    def test_tv_from_heights_with_surface_at_zero(self):
        # Z1 = 0 m is a valid height, not a missing value
        dz = 287.05 * 288.0 * np.log(1e5 / 9e4) / 9.81
        Tv = met.hypsometric(p1=1e5, p2=9e4, Z1=0.0, Z2=dz)
        assert _mag(Tv) == pytest.approx(288.0, rel=1e-6)

    def test_top_height_from_bottom_height(self):
        # Adding to Z1 needs unit-carrying inputs: the constants are pint
        # quantities, so a bare-float Tv leaves the thickness in odd units
        from lair import units

        Z2 = met.hypsometric(
            Tv=288.0 * units("K"), p1=1e5, p2=9e4, Z1=100.0 * units("m")
        )
        assert Z2.to("m").magnitude == pytest.approx(
            100.0 + 287.05 * 288.0 * np.log(1e5 / 9e4) / 9.81, rel=1e-6
        )

    def test_invalid_combination_raises(self):
        # Over-specified inputs (heights *and* a full pressure/temperature set)
        # fall through every solvable branch to the explicit guard.
        with pytest.raises(ValueError):
            met.hypsometric(Tv=288.0, p1=1e5, p2=9e4, Z1=100.0, Z2=1000.0)


def test_standard_atmosphere_values():
    """The standard-atmosphere dict carries the documented surface values."""
    assert met.standard["T"].to("K").magnitude == pytest.approx(288.15)
    assert met.standard["p"].to("hPa").magnitude == pytest.approx(1013.25)
    assert met.standard["rho"].to("kg / m**3").magnitude == pytest.approx(1.225)
