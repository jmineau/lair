"""Tests for lair.pollutants."""

import pytest

from lair.pollutants import POLLUTANTS, Pollutant, get_pollutant


class TestGetPollutant:
    def test_case_insensitive(self):
        assert get_pollutant("ch4") is POLLUTANTS["CH4"]
        assert get_pollutant("NOX").name == "NOx"
        assert get_pollutant(" pm2.5 ").name == "PM2.5"

    def test_aliases(self):
        assert get_pollutant("PM25") is POLLUTANTS["PM2.5"]

    def test_unknown_lists_known(self):
        with pytest.raises(KeyError, match="Known"):
            get_pollutant("XYZ")


class TestLabel:
    def test_latex_with_units(self):
        assert get_pollutant("CH4").label() == r"$\mathrm{CH_4}$ [ppm]"
        assert (
            get_pollutant("PM2.5").label() == r"$\mathrm{PM_{2.5}}$ [$\mu$g m$^{-3}$]"
        )

    def test_plain(self):
        assert get_pollutant("PM2.5").label(latex=False) == "PM2.5 [ug/m3]"
        assert get_pollutant("O3").label(units=False, latex=False) == "O3"


class TestRegistry:
    @pytest.mark.parametrize("pollutant", list(POLLUTANTS.values()), ids=str)
    def test_entries_are_consistent(self, pollutant: Pollutant):
        lo, hi = pollutant.expected_range
        assert lo < hi
        assert pollutant.latex.startswith("$")
        assert POLLUTANTS[pollutant.name] is pollutant

    def test_frozen(self):
        with pytest.raises(AttributeError):
            get_pollutant("CO2").units = "ppb"  # type: ignore[misc]


class TestMolarMass:
    def test_gases(self):
        pytest.importorskip("molmass")
        assert get_pollutant("CH4").molar_mass.to("g/mol").magnitude == pytest.approx(
            16.04, abs=0.01
        )
        # NOx is on an NO2 basis, as in lair.inventories.molecular_weight
        assert get_pollutant("NOx").molar_mass == get_pollutant("NO2").molar_mass

    def test_aerosol_has_none(self):
        with pytest.raises(ValueError, match="no chemical formula"):
            _ = get_pollutant("PM2.5").molar_mass
