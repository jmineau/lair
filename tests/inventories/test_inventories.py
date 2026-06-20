"""Tests for lair.inventories.

Requires the `geo` extra (imports lair.geo) plus molmass. The concrete inventory
loaders (EDGAR/EPA/GFEI/Vulcan/WetCHARTs) read CHPC group data and are left for
`chpc`-marked tests; the module-level unit/sector helpers are tested here.
"""

import numpy as np
import pytest
import xarray as xr

# exc_type=ImportError: lair._optional re-raises missing extras as a plain
# ImportError (not ModuleNotFoundError), which importorskip ignores by default.
inventories = pytest.importorskip(
    "lair.inventories", reason="requires the `geo` extra", exc_type=ImportError
)


class TestMolecularWeight:
    def test_methane(self):
        assert inventories.molecular_weight("CH4").to("g/mol").magnitude == pytest.approx(
            16.0425, abs=1e-3
        )

    def test_carbon_dioxide(self):
        assert inventories.molecular_weight("CO2").to("g/mol").magnitude == pytest.approx(
            44.0096, abs=1e-3
        )


class TestSumSectors:
    def test_sums_variables_into_total(self):
        ds = xr.Dataset(
            {
                "energy": (("y", "x"), np.ones((2, 2))),
                "agriculture": (("y", "x"), 2 * np.ones((2, 2))),
            }
        )
        ds["energy"].attrs["units"] = "kg/m2/s"
        ds["agriculture"].attrs["units"] = "kg/m2/s"
        total = inventories.sum_sectors(ds)
        assert np.unique(total.values).tolist() == [3.0]
        assert total.attrs["units"] == "kg/m2/s"
        assert total.attrs["long_name"] == "Total Emissions"


class TestConvertUnits:
    def test_substance_to_mass_flux(self):
        # mol CH4 / m2 / s -> kg / m2 / s via the mass_flux pint context.
        da = xr.DataArray(np.ones((2, 2)), dims=["y", "x"]).pint.quantify(
            "mole/meter**2/second"
        )
        out = inventories.convert_units(da, "CH4", "kg/meter**2/second")
        # 1 mol/m2/s * 16.0425 g/mol = 0.0160425 kg/m2/s
        assert float(out.pint.dequantify().values[0, 0]) == pytest.approx(0.0160425, abs=1e-6)

    def test_rejects_non_xarray(self):
        with pytest.raises(TypeError):
            inventories.convert_units([1, 2, 3], "CH4", "kg/m2/s")


@pytest.fixture
def inventory():
    """A minimal in-memory Inventory: two sectors on a 2x2 annual grid."""
    import pandas as pd

    time = pd.date_range("2020-01-01", periods=2, freq="YS")
    lat = np.array([40.0, 41.0])
    lon = np.array([-112.0, -111.0])
    shp = (2, 2, 2)
    ds = xr.Dataset(
        {
            "energy": (("time", "lat", "lon"), np.ones(shp)),
            "agriculture": (("time", "lat", "lon"), 2 * np.ones(shp)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    for var in ds.data_vars:
        ds[var].attrs["units"] = "kg/m**2/s"
    return inventories.Inventory(ds, pollutant="CH4", src_units="kg/m**2/s")


class TestBaseInventory:
    def test_standard_name(self, inventory):
        assert inventory.get_standard_name() == "annual_CH4_emissions"

    def test_get_units(self, inventory):
        assert inventory.get_units() == ("kg", "m**2", "s")

    def test_get_files_none_for_in_memory(self, inventory):
        assert inventory.get_files() is None

    def test_total_emissions_sums_sectors(self, inventory):
        assert np.unique(inventory.total_emissions.values).tolist() == [3.0]

    def test_collapsed_has_sector_dim(self, inventory):
        assert "sector" in inventory.collapsed.dims

    def test_data_exposes_variables(self, inventory):
        assert set(inventory.data.data_vars) == {"energy", "agriculture"}

    def test_convert_units_returns_new(self, inventory):
        converted = inventory.convert_units("mol/m**2/s")
        assert converted is not inventory
        assert converted.get_units()[0] == "mol"

    def test_convert_units_inplace(self, inventory):
        same = inventory.convert_units("mol/m**2/s", inplace=True)
        assert same is inventory
        assert inventory.get_units()[0] == "mol"

    def test_absolute_emissions(self, inventory):
        absolute = inventory.absolute_emissions
        assert set(absolute.data_vars) == {"energy", "agriculture"}
        assert absolute.attrs["long_name"] == "Absolute Emissions"

    def test_integrate_per_time_step(self, inventory):
        integrated = inventory.integrate()
        assert integrated.sizes["time"] == 2
        assert bool((integrated.values > 0).all())

    def test_missing_units_raises(self):
        import pandas as pd

        ds = xr.Dataset(
            {"energy": (("time", "lat", "lon"), np.ones((1, 1, 1)))},
            coords={"time": [pd.Timestamp("2020-01-01")], "lat": [40.0], "lon": [-112.0]},
        )
        with pytest.raises(ValueError):
            inventories.Inventory(ds, pollutant="CH4")
