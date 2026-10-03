"""Tests for lair.meteorology.

Contract: plain SI in, plain SI out. pint Quantity inputs are converted to SI
magnitudes first, so every function returns plain numbers / arrays, never a
Quantity.
"""

import numpy as np
import pytest

import pint
import xarray as xr

from lair import meteorology as met
from lair import units

# Same values as lair.constants (SI)
RD, G, RSTAR, KB = 287.05, 9.81, 1.380649e-23 * 6.02214076e23, 1.380649e-23


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

    def test_ice_known_values(self):
        # Petty: e_si = 3.41e12 Pa * exp(-6130 / T). ~611 Pa at the triple
        # point (same as over liquid) and ~103 Pa at -20 C
        assert met.sat_vapor_pres_ice(273.15) == pytest.approx(611.0, rel=0.01)
        assert met.sat_vapor_pres_ice(273.15) == pytest.approx(
            met.sat_vapor_pres(273.15), rel=0.01
        )
        assert met.sat_vapor_pres_ice(253.15) == pytest.approx(103.0, rel=0.02)

    def test_T_from_e_inverts_sat_vapor_pres(self):
        e = met.sat_vapor_pres(295.0)
        assert met.T_from_e(e) == pytest.approx(295.0)


class TestMixingRatio:
    def test_exact_form(self):
        # w = epsilon * e / (p - e), epsilon = Rd / Rv = 287.05 / 461.5.
        # By hand: 0.621993 * 2000 / 83000 = 0.0149878 kg/kg (the e/p
        # approximation would give 0.0146351, 2.4% low)
        w = met.mixing_ratio(2000.0, 85000.0)
        assert isinstance(w, float)
        assert w == pytest.approx(0.0149878, rel=1e-5)
        assert w == pytest.approx(287.05 / 461.5 * 2000.0 / 83000.0)

    def test_dry_air_is_zero(self):
        assert met.mixing_ratio(0.0, 85000.0) == 0.0


class TestIdealGasLaw:
    def test_density_form(self):
        # rho = p / (R T)
        rho = met.ideal_gas_law("density", p=1e5, R=287.05, T=300.0)
        assert rho == pytest.approx(1e5 / (287.05 * 300.0))

    def test_temperature_from_density(self):
        T = met.ideal_gas_law("temp", p=1e5, rho=1.2, R=287.05)
        assert T == pytest.approx(1e5 / (1.2 * 287.05))

    def test_invalid_solve_for_raises(self):
        with pytest.raises(ValueError):
            met.ideal_gas_law("entropy", p=1e5, T=300.0)

    def test_mass_form(self):
        # m = p V / (R T)
        m = met.ideal_gas_law("mass", p=1e5, V=1.0, R=287.05, T=300.0)
        assert m == pytest.approx(1e5 / (287.05 * 300.0))

    def test_moles_form(self):
        # n = p V / (R* T)

        n = met.ideal_gas_law("moles", p=1e5, V=1.0, T=300.0)
        assert n == pytest.approx(1e5 / (RSTAR * 300.0))

    def test_number_form(self):
        # N = p V / (kb T)

        N = met.ideal_gas_law("number", p=1e5, V=1.0, T=300.0)
        assert N == pytest.approx(1e5 / (KB * 300.0))

    def test_volume_from_moles(self):

        V = met.ideal_gas_law("volume", p=1e5, n=1.0, T=300.0)
        assert V == pytest.approx(RSTAR * 300.0 / 1e5)

    def test_volume_from_mass(self):
        # V = m R T / p
        V = met.ideal_gas_law("volume", p=1e5, m=1.0, R=287.05, T=300.0)
        assert V == pytest.approx(287.05 * 300.0 / 1e5)

    def test_pressure_from_moles_and_volume(self):
        # p = n R* T / V

        p = met.ideal_gas_law("pressure", V=1.0, n=1.0, T=300.0)
        assert p == pytest.approx(RSTAR * 300.0)

    def test_pressure_from_mass_and_volume(self):
        # p = m R T / V
        p = met.ideal_gas_law("pressure", V=1.0, m=1.0, R=287.05, T=300.0)
        assert p == pytest.approx(287.05 * 300.0)

    def test_temperature_from_volume_and_moles(self):

        T = met.ideal_gas_law("temperature", p=1e5, V=1.0, n=1.0)
        assert T == pytest.approx(1e5 / RSTAR)

    def test_pressure_from_density_arrays(self):
        # Array inputs must not be tested for truthiness
        rho = np.array([1.2, 1.1])
        T = np.array([290.0, 280.0])
        p = met.ideal_gas_law("pressure", rho=rho, R=287.05, T=T)
        np.testing.assert_allclose(p, rho * 287.05 * T)

    def test_pressure_from_specific_volume_arrays(self):
        alpha = np.array([0.8, 0.9])
        p = met.ideal_gas_law("pressure", alpha=alpha, R=287.05, T=300.0)
        np.testing.assert_allclose(p, 287.05 * 300.0 / alpha)

    def test_temperature_from_volume_arrays(self):

        V = np.array([1.0, 2.0])
        n = np.array([1.0, 3.0])
        T = met.ideal_gas_law("temperature", p=1e5, V=V, n=n)
        np.testing.assert_allclose(T, 1e5 * V / (n * RSTAR))

    def test_volume_from_mass_arrays(self):
        m = np.array([1.0, 2.0])
        V = met.ideal_gas_law("volume", p=1e5, m=m, R=287.05, T=300.0)
        np.testing.assert_allclose(V, m * 287.05 * 300.0 / 1e5)

    def test_zero_moles_is_a_value_not_missing(self):
        # n = 0 is valid input (zero pressure), not "n not given"
        p = met.ideal_gas_law("pressure", V=1.0, n=0.0, T=300.0)
        assert p == pytest.approx(0.0)


class TestHypsometric:
    def test_thickness_positive_for_decreasing_pressure(self):
        # Solve for layer thickness given Tv and bounding pressures.
        deltaz = met.hypsometric(Tv=288.0, p1=1e5, p2=9e4)
        assert deltaz == pytest.approx(
            287.05 * 288.0 * np.log(1e5 / 9e4) / 9.81, rel=1e-6
        )

    def test_tv_from_heights_with_surface_at_zero(self):
        # Z1 = 0 m is a valid height, not a missing value
        dz = 287.05 * 288.0 * np.log(1e5 / 9e4) / 9.81
        Tv = met.hypsometric(p1=1e5, p2=9e4, Z1=0.0, Z2=dz)
        assert Tv == pytest.approx(288.0, rel=1e-6)

    def test_thickness_is_plain_metres(self):
        # By hand: 287.05 * 280 * ln(1e5 / 9e4) / 9.81 = 863.226 m
        deltaz = met.hypsometric(Tv=280.0, p1=1e5, p2=9e4)
        assert isinstance(deltaz, float)
        assert deltaz == pytest.approx(863.2259, rel=1e-6)

    def test_top_height_from_bottom_height(self):
        # Plain floats: Z1 adds to the thickness in metres
        Z2 = met.hypsometric(Tv=280.0, p1=1e5, p2=9e4, Z1=1289.0)
        assert isinstance(Z2, float)
        assert Z2 == pytest.approx(1289.0 + 863.2259, rel=1e-6)

    def test_bottom_height_from_top_height(self):
        Z1 = met.hypsometric(Tv=280.0, p1=1e5, p2=9e4, Z2=2152.2259)
        assert Z1 == pytest.approx(1289.0, rel=1e-6)

    def test_pressure_from_thickness(self):
        p2 = met.hypsometric(Tv=280.0, p1=1e5, deltaz=863.2259)
        p1 = met.hypsometric(Tv=280.0, p2=9e4, deltaz=863.2259)
        assert p2 == pytest.approx(9e4, rel=1e-6)
        assert p1 == pytest.approx(1e5, rel=1e-6)

    def test_array_inputs(self):
        Tv = np.array([280.0, 290.0])
        deltaz = met.hypsometric(Tv=Tv, p1=1e5, p2=9e4)
        assert not isinstance(deltaz, pint.Quantity)
        np.testing.assert_allclose(deltaz, RD * Tv * np.log(1e5 / 9e4) / G)

    def test_invalid_combination_raises(self):
        # Over-specified inputs (heights *and* a full pressure/temperature set)
        # fall through every solvable branch to the explicit guard.
        with pytest.raises(ValueError):
            met.hypsometric(Tv=288.0, p1=1e5, p2=9e4, Z1=100.0, Z2=1000.0)


class TestPlainSIOutputs:
    """Plain SI in gives plain SI out with sensible magnitudes (issue #41)."""

    def test_pressure_from_moles_is_pascals(self):
        # By hand: 1 mol * 8.314463 J/mol/K * 300 K / 1 m3 = 2494.34 Pa
        p = met.ideal_gas_law("p", n=1.0, V=1.0, T=300.0)
        assert isinstance(p, float)
        assert p == pytest.approx(2494.3388, rel=1e-6)

    def test_potential_temperature_is_kelvin(self):
        # By hand: 273.15 * (1e5 / 85000) ** (287.05 / 1005) = 286.128 K
        theta = met.poisson(273.15, 85000.0)
        assert isinstance(theta, float)
        assert theta == pytest.approx(286.1282, rel=1e-6)

    def test_xarray_in_xarray_out(self):
        T = xr.DataArray([270.0, 280.0], dims="height")
        theta = met.poisson(T, xr.DataArray([85000.0, 80000.0], dims="height"))
        assert isinstance(theta, xr.DataArray)
        assert not isinstance(theta.data, pint.Quantity)


class TestQuantityInputs:
    """pint Quantities are converted to SI magnitudes; outputs are plain."""

    def test_hypsometric_converts_to_si(self):
        deltaz = met.hypsometric(
            Tv=280.0 * units("K"), p1=1000.0 * units("hPa"), p2=900.0 * units("hPa")
        )
        assert isinstance(deltaz, float)
        assert deltaz == pytest.approx(863.2259, rel=1e-6)

    def test_offset_units_convert(self):
        # 0 degC -> 273.15 K, 850 hPa -> 85000 Pa
        theta = met.poisson(
            units.Quantity(0.0, "degC"), 850.0 * units("hPa"), p0=1e5 * units("Pa")
        )
        assert theta == pytest.approx(286.1282, rel=1e-6)

    def test_heights_in_km(self):
        Z2 = met.hypsometric(Tv=280.0, p1=1e5, p2=9e4, Z1=1.289 * units("km"))
        assert Z2 == pytest.approx(1289.0 + 863.2259, rel=1e-6)

    def test_ideal_gas_law_with_constant_quantity(self):
        from lair.constants import Rd

        rho = met.ideal_gas_law("rho", p=850.0 * units("hPa"), T=270.0, R=Rd)
        assert isinstance(rho, float)
        assert rho == pytest.approx(85000.0 / (RD * 270.0))

    def test_quantified_dataarray(self):
        p = xr.DataArray([850.0, 800.0], dims="height").pint.quantify("hPa")
        T = xr.DataArray([-3.15, 6.85], dims="height").pint.quantify("degC")
        theta = met.poisson(T, p)
        assert isinstance(theta, xr.DataArray)
        assert not isinstance(theta.data, pint.Quantity)
        np.testing.assert_allclose(
            theta.values,
            [270.0, 280.0] * (1e5 / np.array([85000.0, 80000.0])) ** (RD / 1005),
        )


def test_standard_atmosphere_values():
    """The standard atmosphere is plain SI: K, Pa, kg/m3, m."""
    assert met.standard == pytest.approx(
        {"T": 288.15, "p": 101325.0, "rho": 1.225, "z": 0.0}
    )
    assert all(isinstance(v, float) for v in met.standard.values())
