"""Tests for lair.inventories.

Requires the `geo` extra (imports lair.geo) plus molmass. The base Inventory
machinery and unit/sector helpers are tested on synthetic data, and Vulcan on a
tiny synthetic archive shaped like the real v3 files. The other concrete loaders
(EDGAR/EPA/GFEI/WetCHARTs) need the real archive (LAIR_INVENTORY_DIR).
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

    @pytest.mark.parametrize("time_step, seconds", [("daily", 86400), ("hourly", 3600)])
    def test_absolute_emissions_sub_monthly(self, inventory, time_step, seconds):
        inv = inventories.Inventory(
            inventory.data.pint.dequantify(), pollutant="CH4",
            src_units="kg/m**2/s", time_step=time_step,
        )
        absolute = inv.absolute_emissions
        # 1 kg/m2/s over one gridcell (km2 -> m2) for one time step
        expected = inv.gridcell_area.values * 1e6 * seconds
        np.testing.assert_allclose(absolute["energy"].isel(time=0).values, expected)

    def test_missing_units_raises(self):
        import pandas as pd

        ds = xr.Dataset(
            {"energy": (("time", "lat", "lon"), np.ones((1, 1, 1)))},
            coords={"time": [pd.Timestamp("2020-01-01")], "lat": [40.0], "lon": [-112.0]},
        )
        with pytest.raises(ValueError):
            inventories.Inventory(ds, pollutant="CH4")


class TestInventoryDir:
    def test_unset_env_raises(self, monkeypatch):
        monkeypatch.delenv("LAIR_INVENTORY_DIR", raising=False)
        with pytest.raises(ValueError, match="LAIR_INVENTORY_DIR"):
            inventories.EDGARv8("CH4")


VULCAN_SECTORS = ["onroad", "elec_prod"]

#: Vulcan files are tC; lair loads them as CO2 mass (M(CO2)/M(C))
C_TO_CO2 = 44.0095 / 12.0107


@pytest.fixture
def vulcan_dir(tmp_path):
    """A tiny Vulcan v3 archive shaped like the real files: (time, y, x) on the
    Vulcan LCC grid with 2D lat/lon coords, one file per sector and bound."""
    import pandas as pd
    from pyproj import Transformer

    x = np.arange(-1.5e6, -1.5e6 + 10_000, 1000.0)  # 10 x 1 km cells
    y = np.arange(4e5, 4e5 + 8_000, 1000.0)          # 8 x 1 km cells
    to_ll = Transformer.from_crs(inventories.Vulcan.crs, "EPSG:4326", always_xy=True)
    lon, lat = to_ll.transform(*np.meshgrid(x, y))
    time = pd.to_datetime(["2014-07-02T12:00", "2015-07-02T12:00"])

    d = tmp_path / "vulcan" / "v3" / "data" / "native" / "annual"
    d.mkdir(parents=True)
    for sector in VULCAN_SECTORS + ["total"]:
        for bound, value in [("mn", 2.0), ("lo", 1.0), ("hi", 3.0)]:
            emis = np.full((2, y.size, x.size), value)
            if sector == "elec_prod":
                # Point-source sectors are NaN except where there are sources
                emis[:] = np.nan
                emis[:, 4, 5] = 100 * value
            ds = xr.Dataset(
                {
                    "carbon_emissions": (("time", "y", "x"), emis),
                    "time_bnds": (("time", "nv"), np.stack([time, time], axis=1)),
                    "crs": ((), np.int16(0)),
                },
                coords={"time": time, "y": y, "x": x,
                        "lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
            )
            ds.carbon_emissions.attrs["units"] = "Mg km-2 year-1"
            ds.to_netcdf(d / f"Vulcan_v3_US_annual_1km_{sector}_{bound}.nc4")
    return tmp_path


class TestVulcan:
    def test_loads_projected_grid(self, vulcan_dir):
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        assert set(v.data.data_vars) == {"onroad", "elec"}  # 'total' excluded
        assert v.data.rio.x_dim == "x" and v.data.rio.y_dim == "y"
        assert v.data.time.dt.month.values.tolist() == [1, 1]
        # 1 km^2 cells on the projected grid
        np.testing.assert_allclose(v.gridcell_area.values, 1.0)

    def test_env_var_fallback(self, vulcan_dir, monkeypatch):
        monkeypatch.setenv("LAIR_INVENTORY_DIR", str(vulcan_dir))
        v = inventories.Vulcan()
        assert v.vulcan_dir == str(vulcan_dir / "vulcan")

    def test_clip_returns_clipped_copy(self, vulcan_dir):
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        bbox = (-1.5e6, 4e5, -1.5e6 + 4_000, 4e5 + 3_000)  # data CRS (metres)
        clipped = v.clip(bbox=bbox)
        assert clipped is not v
        assert clipped._is_clipped and not v._is_clipped
        assert clipped.data.sizes["x"] < v.data.sizes["x"]
        assert v.data.sizes["x"] == 10  # original untouched

    def test_clip_inplace(self, vulcan_dir):
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        same = v.clip(bbox=(-1.5e6, 4e5, -1.5e6 + 4_000, 4e5 + 3_000), inplace=True)
        assert same is v and v._is_clipped
        assert v.data.sizes["x"] < 10

    def test_reproject_requires_clip(self, vulcan_dir):
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        with pytest.raises(ValueError, match="clipped"):
            v.reproject(0.01)

    def test_carbon_converted_to_co2(self, vulcan_dir):
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        onroad = v.data["onroad"].pint.dequantify()
        np.testing.assert_allclose(onroad.values, 2.0 * C_TO_CO2, rtol=1e-4)

    def test_no_emission_cells_are_zero(self, vulcan_dir):
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        elec = v.data["elec"].pint.dequantify()
        assert not bool(elec.isnull().any())
        # one source cell, 2 years, converted from tC to CO2
        assert float(elec.sum()) == pytest.approx(2 * 200.0 * C_TO_CO2, rel=1e-4)

    def test_reproject_returns_latlon(self, vulcan_dir):
        pytest.importorskip("xesmf")
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        out = v.clip(bbox=(-1.5e6, 4e5, -1.5e6 + 9_000, 4e5 + 7_000)).reproject(0.02)
        assert out.crs.epsg == 4326
        assert {"lat", "lon"} <= set(out.data.dims)

    def test_reproject_conserves_point_sources(self, vulcan_dir):
        # NaN "no emission" cells must not wipe out neighbouring point sources
        pytest.importorskip("xesmf")
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        clipped = v.clip(bbox=(-1.5e6, 4e5, -1.5e6 + 9_000, 4e5 + 7_000))
        out = clipped.reproject(0.02)

        def elec_total(inv, dims):
            absolute = inv.absolute_emissions["elec"].pint.quantify()
            return float(absolute.sum(dims).pint.to("Mg").pint.dequantify().sum())

        src = elec_total(clipped, ["x", "y"])
        assert src > 0
        assert elec_total(out, ["lat", "lon"]) == pytest.approx(src, rel=0.03)

    def test_uncertainties(self, vulcan_dir):
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        lower = v.get_uncertainties("lower")
        upper = v.get_uncertainties("upper")
        assert set(lower.data_vars) == {"onroad", "elec"}
        assert float(lower["onroad"].max()) == pytest.approx(1.0 * C_TO_CO2, rel=1e-4)
        assert float(upper["onroad"].max()) == pytest.approx(3.0 * C_TO_CO2, rel=1e-4)


class TestPollutantNames:
    def test_nox_uses_no2_mass(self):
        assert inventories.molecular_weight("NOx").magnitude == pytest.approx(
            inventories.molecular_weight("NO2").magnitude)

    @pytest.mark.parametrize("given, kept", [("NOx", "NOx"), ("ch4", "CH4"), ("CO2", "CO2")])
    def test_pollutant_case(self, inventory, given, kept):
        inv = inventories.Inventory(inventory.data.pint.dequantify(), pollutant=given,
                                    src_units="kg/m**2/s")
        assert inv.pollutant == kept


class TestEPAv2ExpressMonthly:
    """express + scale_by_month: after 2018 only three sectors keep a monthly
    pattern; the other scaled sectors fall back to their annual rate."""

    def test_no_nan_after_2018(self):
        import pandas as pd

        epa = inventories.EPAv2.__new__(inventories.EPAv2)
        epa.express = True
        epa._lat_deci = epa._lon_deci = 2
        lat, lon = [40.0, 40.1], [-112.0, -111.9]
        scalable = inventories.EPAv2._express_vars_scalable_past_2018
        names = scalable + ["Enteric_Fermentation", "Landfills_MSW"]
        annual = xr.Dataset(
            {n: (("time", "lat", "lon"), np.full((2, 2, 2), 3.0)) for n in names},
            coords={"time": pd.to_datetime(["2018-01-01", "2019-01-01"]), "lat": lat, "lon": lon},
        )
        sf = xr.Dataset(
            {n: (("time", "lat", "lon"), np.full((12, 2, 2), 2.0)) for n in names},
            coords={"time": pd.date_range("2018-01-01", periods=12, freq="MS"), "lat": lat, "lon": lon},
        )
        epa.get_monthly_scale_factors = lambda: sf

        out = epa._scale_by_month(annual)
        y2019 = out.sel(time="2019")
        assert y2019.sizes["time"] == 12
        for n in names:
            assert not bool(y2019[n].isnull().any()), n
        assert float(y2019["Manure_Management"].max()) == 6.0     # scaled
        assert float(y2019["Enteric_Fermentation"].max()) == 3.0  # annual rate
