"""Tests for lair.inventories.

Requires the `geo` extra (imports lair.geo) plus molmass. The base Inventory
machinery and unit/sector helpers are tested on synthetic data, and the concrete
loaders (Vulcan, EDGAR, EPA v2, WetCHARTs) on tiny synthetic archives shaped like
the real files. GFEI and EPA v1 are not covered: they need the real archive
(LAIR_INVENTORY_DIR).
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
        assert inventories.molecular_weight("CH4").to(
            "g/mol"
        ).magnitude == pytest.approx(16.0425, abs=1e-3)

    def test_carbon_dioxide(self):
        assert inventories.molecular_weight("CO2").to(
            "g/mol"
        ).magnitude == pytest.approx(44.0096, abs=1e-3)


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
        assert float(out.pint.dequantify().values[0, 0]) == pytest.approx(
            0.0160425, abs=1e-6
        )

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

    def test_coords_stay_unquantified(self, inventory):
        # A coordinate index wrapped in a pint Quantity cannot be aligned
        # against a plain one, which breaks anything combining the data with
        # something derived from it.
        for coord in inventory._data.coords:
            assert not hasattr(inventory._data[coord].data, "units"), coord

    def test_absolute_emissions_with_units_on_coords(self):
        # Real inventory files carry `units: degrees_north` on lat/lon, which
        # pint-xarray quantifies unless told not to. The plain fixture has no
        # units on its coords and so never exercised this path.
        import pandas as pd

        time = pd.date_range("2020-01-01", periods=2, freq="YS")
        ds = xr.Dataset(
            {"energy": (("time", "lat", "lon"), np.ones((2, 2, 2)))},
            coords={
                "time": time,
                "lat": ("lat", np.array([40.0, 41.0]), {"units": "degrees_north"}),
                "lon": ("lon", np.array([-112.0, -111.0]), {"units": "degrees_east"}),
            },
        )
        ds["energy"].attrs["units"] = "kg/m**2/s"
        inv = inventories.Inventory(ds, pollutant="CH4", src_units="kg/m**2/s")
        assert not hasattr(inv._data.lat.data, "units")
        assert bool((inv.integrate().values > 0).all())

    def test_clip_keeps_integrate_working(self, inventory):
        # clip() assigns through the data setter, which re-quantifies.
        clipped = inventory.clip(bbox=(-112.5, 39.5, -110.5, 41.5))
        assert bool((clipped.integrate().values > 0).all())

    def test_integrate_per_time_step(self, inventory):
        integrated = inventory.integrate()
        assert integrated.sizes["time"] == 2
        assert bool((integrated.values > 0).all())

    @pytest.mark.parametrize("time_step, seconds", [("daily", 86400), ("hourly", 3600)])
    def test_absolute_emissions_sub_monthly(self, inventory, time_step, seconds):
        inv = inventories.Inventory(
            inventory.data.pint.dequantify(),
            pollutant="CH4",
            src_units="kg/m**2/s",
            time_step=time_step,
        )
        absolute = inv.absolute_emissions
        # 1 kg/m2/s over one gridcell (km2 -> m2) for one time step
        expected = inv.gridcell_area.values * 1e6 * seconds
        np.testing.assert_allclose(absolute["energy"].isel(time=0).values, expected)

    def test_missing_units_raises(self):
        import pandas as pd

        ds = xr.Dataset(
            {"energy": (("time", "lat", "lon"), np.ones((1, 1, 1)))},
            coords={
                "time": [pd.Timestamp("2020-01-01")],
                "lat": [40.0],
                "lon": [-112.0],
            },
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
    y = np.arange(4e5, 4e5 + 8_000, 1000.0)  # 8 x 1 km cells
    to_ll = Transformer.from_crs(
        inventories.Vulcan.native_crs, "EPSG:4326", always_xy=True
    )
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
                coords={
                    "time": time,
                    "y": y,
                    "x": x,
                    "lat": (("y", "x"), lat),
                    "lon": (("y", "x"), lon),
                },
            )
            ds.carbon_emissions.attrs["units"] = "Mg km-2 year-1"
            ds.to_netcdf(d / f"Vulcan_v3_US_annual_1km_{sector}_{bound}.nc4")
    return tmp_path


class TestVulcan:
    def test_loads_projected_grid(self, vulcan_dir):
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        assert set(v.data.data_vars) == {"onroad", "elec_prod"}  # 'total' excluded
        assert v.data.rio.x_dim == "x" and v.data.rio.y_dim == "y"
        assert v.data.time.dt.month.values.tolist() == [1, 1]
        # 1 km^2 cells on the projected grid, in the data's (y, x) order
        assert v.gridcell_area.dims == ("y", "x")
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
        elec = v.data["elec_prod"].pint.dequantify()
        assert not bool(elec.isnull().any())
        # one source cell, 2 years, converted from tC to CO2
        assert float(elec.sum()) == pytest.approx(2 * 200.0 * C_TO_CO2, rel=1e-4)

    def test_data_stays_lazy(self, vulcan_dir):
        # The full US grid is ~20 GB in memory: nothing up to integrate()
        # should load it (#27)
        dask_array = pytest.importorskip("dask.array")
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        clipped = v.clip(bbox=(-1.5e6, 4e5, -1.5e6 + 9_000, 4e5 + 7_000))
        for inv in (v, clipped):
            for var in inv.data.data_vars.values():
                assert isinstance(var.data, dask_array.Array)
        assert isinstance(clipped.integrate().data, dask_array.Array)

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
            absolute = inv.absolute_emissions["elec_prod"].pint.quantify()
            return float(absolute.sum(dims).pint.to("Mg").pint.dequantify().sum())

        src = elec_total(clipped, ["x", "y"])
        assert src > 0
        assert elec_total(out, ["lat", "lon"]) == pytest.approx(src, rel=0.03)

    def test_uncertainties(self, vulcan_dir):
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        lower = v.get_uncertainties("lower")
        upper = v.get_uncertainties("upper")
        assert set(lower.data_vars) == {"onroad", "elec_prod"}
        assert float(lower["onroad"].max()) == pytest.approx(1.0 * C_TO_CO2, rel=1e-4)
        assert float(upper["onroad"].max()) == pytest.approx(3.0 * C_TO_CO2, rel=1e-4)


class TestPollutantNames:
    def test_nox_uses_no2_mass(self):
        assert inventories.molecular_weight("NOx").magnitude == pytest.approx(
            inventories.molecular_weight("NO2").magnitude
        )

    @pytest.mark.parametrize(
        "given, kept", [("NOx", "NOx"), ("ch4", "CH4"), ("CO2", "CO2")]
    )
    def test_pollutant_case(self, inventory, given, kept):
        inv = inventories.Inventory(
            inventory.data.pint.dequantify(), pollutant=given, src_units="kg/m**2/s"
        )
        assert inv.pollutant == kept


def _epa_v2(express):
    """An uninitialized EPAv2, for exercising _scale_by_month directly."""
    epa = inventories.EPAv2.__new__(inventories.EPAv2)
    epa.express = express
    epa._lat_deci = epa._lon_deci = 2
    return epa


#: Like the real v2 files, only some sectors have monthly scale factors
EPA_SCALED = inventories.EPAv2._express_vars_scalable_past_2018 + [
    "Combustion_Stationary"
]
EPA_ANNUAL_ONLY = ["Enteric_Fermentation", "Landfills_MSW"]


def _epa_annual_and_sf(years):
    import pandas as pd

    lat, lon = [40.0, 40.1], [-112.0, -111.9]
    annual = xr.Dataset(
        {
            n: (("time", "lat", "lon"), np.full((len(years), 2, 2), 3.0))
            for n in EPA_SCALED + EPA_ANNUAL_ONLY
        },
        coords={
            "time": pd.to_datetime([f"{y}-01-01" for y in years]),
            "lat": lat,
            "lon": lon,
        },
    )
    sf = xr.Dataset(
        {n: (("time", "lat", "lon"), np.full((12, 2, 2), 2.0)) for n in EPA_SCALED},
        coords={
            "time": pd.date_range("2018-01-01", periods=12, freq="MS"),
            "lat": lat,
            "lon": lon,
        },
    )
    return annual, sf


class TestEPAv2Monthly:
    """scale_by_month keeps every sector: those with monthly scale factors are
    scaled, the rest (enteric fermentation, landfills, coal, ...) hold their
    annual rate in every month."""

    def test_keeps_sectors_without_scale_factors(self):
        epa = _epa_v2(express=False)
        annual, sf = _epa_annual_and_sf([2018])
        epa.get_monthly_scale_factors = lambda: sf

        out = epa._scale_by_month(annual)
        assert set(out.data_vars) == set(EPA_SCALED + EPA_ANNUAL_ONLY)
        assert out.sizes["time"] == 12
        assert float(out["Manure_Management"].max()) == 6.0  # scaled
        assert float(out["Enteric_Fermentation"].min()) == 3.0  # annual rate
        assert float(out["Enteric_Fermentation"].max()) == 3.0

    def test_express_no_nan_after_2018(self):
        epa = _epa_v2(express=True)
        annual, sf = _epa_annual_and_sf([2018, 2019])
        epa.get_monthly_scale_factors = lambda: sf

        out = epa._scale_by_month(annual)
        assert set(out.data_vars) == set(EPA_SCALED + EPA_ANNUAL_ONLY)
        y2019 = out.sel(time="2019")
        assert y2019.sizes["time"] == 12
        for n in out.data_vars:
            assert not bool(out[n].isnull().any()), n
        # after 2018 only the three express sectors keep a monthly pattern
        assert float(y2019["Manure_Management"].max()) == 6.0
        assert float(y2019["Combustion_Stationary"].max()) == 3.0  # annual rate
        assert float(y2019["Enteric_Fermentation"].max()) == 3.0  # annual rate
        assert float(out.sel(time="2018")["Combustion_Stationary"].max()) == 6.0


@pytest.fixture
def epa_v2_dir(tmp_path):
    """A tiny EPA v2 archive shaped like the real files: one annual file per year
    with ``emi_ch4_<code>_<name>`` variables, and monthly scale factors for only
    some of the sectors."""
    import pandas as pd

    lat, lon = np.array([40.05, 40.15]), np.array([-111.95, -111.85])
    d = tmp_path / "EPA" / "v2"
    (d / "monthly_scale_factors").mkdir(parents=True)
    names = ["1A_Combustion_Stationary", "3B_Manure_Management"]
    names += ["3A_Enteric_Fermentation", "5A1_Landfills_MSW"]
    for year in [2017, 2018]:
        ds = xr.Dataset(
            {
                f"emi_ch4_{n}": (("time", "lat", "lon"), np.ones((1, 2, 2)))
                for n in names + ["grid_cell_area"]
            },
            coords={"time": [pd.Timestamp(f"{year}-01-01")], "lat": lat, "lon": lon},
        )
        ds = ds.rename({"emi_ch4_grid_cell_area": "grid_cell_area"})
        ds.to_netcdf(d / f"Gridded_GHGI_Methane_v2_{year}.nc")
        sf = xr.Dataset(
            {
                f"monthly_scale_factor_{n}": (
                    ("time", "lat", "lon"),
                    np.ones((12, 2, 2)),
                )
                for n in names[:2]
            },
            coords={
                "time": pd.date_range(f"{year}-01-01", periods=12, freq="MS"),
                "lat": lat,
                "lon": lon,
            },
        )
        sf.to_netcdf(
            d
            / "monthly_scale_factors"
            / f"Gridded_GHGI_Methane_v2_Monthly_Scale_Factors_{year}.nc"
        )
    return tmp_path


class TestEPAv2:
    def test_monthly_total_matches_annual(self, epa_v2_dir):
        annual = inventories.EPAv2(inventory_dir=epa_v2_dir)
        monthly = inventories.EPAv2(scale_by_month=True, inventory_dir=epa_v2_dir)
        assert set(monthly.data.data_vars) == set(annual.data.data_vars)
        # scale factors of 1: the months of a year add up to the annual total
        a = float(annual.integrate().sel(time="2018").sum())
        m = float(monthly.integrate().sel(time="2018").sum())
        assert m == pytest.approx(a, rel=1e-6)


def _edgar_v8_file(path, long_name, year, monthly):
    import pandas as pd

    lat, lon = np.array([40.05, 40.15]), np.array([-111.95, -111.85])
    attrs = {"long_name": long_name, "year": str(year), "units": "kg m-2 s-1"}
    if monthly:
        time = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
        fluxes = (("time", "lat", "lon"), np.ones((12, 2, 2)), attrs)
        coords = {"time": time, "lat": lat, "lon": lon}
    else:
        fluxes = (("lat", "lon"), np.ones((2, 2)), attrs)
        coords = {"lat": lat, "lon": lon}
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"fluxes": fluxes}, coords=coords).to_netcdf(path)


class TestEDGARv8:
    def test_annual_drops_fuel_exploitation_total(self, tmp_path):
        # Annual CH4 has PRO_FFF alongside its COAL/GAS/OIL components
        d = tmp_path / "EDGAR" / "v8" / "CH4"
        for code, name in [
            ("PRO_FFF", "Fuel exploitation"),
            ("PRO_COAL", "Fuel exploitation COAL"),
            ("PRO_GAS", "Fuel exploitation GAS"),
            ("PRO_OIL", "Fuel exploitation OIL"),
            ("ENF", "Enteric fermentation"),
        ]:
            f = d / code / f"v8.0_FT2022_GHG_CH4_2020_{code}_flx.nc"
            _edgar_v8_file(f, name, 2020, monthly=False)
        e = inventories.EDGARv8("CH4", inventory_dir=tmp_path)
        assert set(e.data.data_vars) == {
            "Fuel_exploitation_COAL",
            "Fuel_exploitation_GAS",
            "Fuel_exploitation_OIL",
            "Enteric_fermentation",
        }

    def test_monthly_keeps_fuel_exploitation(self, tmp_path):
        # Monthly CH4 has only FUEL_EXPLOITATION (no COAL/GAS/OIL split)
        d = tmp_path / "EDGAR" / "v8" / "monthly" / "CH4"
        for code, name in [
            ("FUEL_EXPLOITATION", "Fuel exploitation"),
            ("AGRICULTURE", "Agriculture"),
        ]:
            f = d / code / f"v8.0_FT2022_GHG_CH4_2020_{code}_flx.nc"
            _edgar_v8_file(f, name, 2020, monthly=True)
        e = inventories.EDGARv8("CH4", time_step="monthly", inventory_dir=tmp_path)
        assert set(e.data.data_vars) == {"Fuel_exploitation", "Agriculture"}


class TestEDGARv7:
    @staticmethod
    def _write(root, pollutant, code):
        d = root / "EDGAR" / "v7" / pollutant / code
        d.mkdir(parents=True)
        var = f"emi_{pollutant.lower()}"
        xr.Dataset(
            {var: (("lat", "lon"), np.ones((2, 2)), {"units": "kg m-2 s-1"})},
            coords={"lat": [40.05, 40.15], "lon": [-111.95, -111.85]},
        ).to_netcdf(d / f"v7.0_FT2021_{pollutant}_2021_{code}.0.1x0.1.nc")

    def test_other_pollutant(self, tmp_path):
        self._write(tmp_path, "N2O", "AGS")
        e = inventories.EDGARv7("N2O", inventory_dir=tmp_path)
        assert set(e.data.data_vars) == {"Agricultural_soils"}

    def test_supersonic_aviation_sector(self, tmp_path):
        self._write(tmp_path, "CH4", "TNR_Aviation_SPS")
        e = inventories.EDGARv7("CH4", inventory_dir=tmp_path)
        assert set(e.data.data_vars) == {"Aviation_supersonic"}


class TestEDGARSectorNames:
    def test_known_sector(self):
        edgar = inventories.EDGARv8.__new__(inventories.EDGARv8)
        assert edgar.get_sector_name("TNR_Aviation_SPS") == "Aviation_supersonic"

    def test_unknown_sector_falls_back_to_code(self):
        edgar = inventories.EDGARv8.__new__(inventories.EDGARv8)
        assert edgar.get_sector_name("NEW_SECTOR") == "NEW_SECTOR"


class TestPerVariableUnits:
    def _ds(self, units_b):
        import pandas as pd

        ds = xr.Dataset(
            {
                "a": (("time", "lat", "lon"), np.ones((1, 2, 2))),
                "b": (("time", "lat", "lon"), np.ones((1, 2, 2))),
            },
            coords={
                "time": [pd.Timestamp("2020-01-01")],
                "lat": [40.0, 41.0],
                "lon": [-112.0, -111.0],
            },
        )
        ds["a"].attrs["units"] = "kg m-2 s-1"
        if units_b is not None:
            ds["b"].attrs["units"] = units_b
        return ds

    def test_each_variable_keeps_its_units(self):
        inv = inventories.Inventory(self._ds("mol m-2 s-1"), pollutant="CH4")
        assert inv._data["a"].pint.units == inventories.units("kg m-2 s-1")
        assert inv._data["b"].pint.units == inventories.units("mol m-2 s-1")

    def test_common_units_become_src_units(self):
        inv = inventories.Inventory(self._ds("kg m-2 s-1"), pollutant="CH4")
        assert inv.src_units == "kg m-2 s-1"

    def test_variable_without_units_raises(self):
        with pytest.raises(ValueError, match="b"):
            inventories.Inventory(self._ds(None), pollutant="CH4")

    def test_src_units_overrides_attrs(self):
        inv = inventories.Inventory(
            self._ds("mol m-2 s-1"), pollutant="CH4", src_units="kg m-2 s-1"
        )
        assert inv._data["b"].pint.units == inventories.units("kg m-2 s-1")


@pytest.fixture
def wetcharts_dir(tmp_path):
    """A tiny WetCHARTs v1.3.1 file: int32 model codes, mid-month times and
    NaN outside wetlands (most of the real grid)."""
    import pandas as pd

    lat = np.arange(40.25, 42, 0.5)
    lon = np.arange(-112.75, -111, 0.5)
    t = pd.date_range("2010-01-01", periods=12, freq="MS") + pd.Timedelta("14D")
    emis = np.ones((2, 12, lat.size, lon.size))
    emis[1] *= 3.0
    emis[:, :, 0, 0] = np.nan  # a non-wetland cell
    ds = xr.Dataset(
        {
            "wetland_CH4_emissions": (
                ("model", "time", "lat", "lon"),
                emis,
                {"units": "mg m-2 d-1"},
            ),
            "time_bnds": (("time", "nv"), np.stack([t, t], axis=1)),
            "crs": ((), "a"),
        },
        coords={
            "model": np.array([1913, 1914], dtype="int32"),
            "time": t,
            "lat": lat,
            "lon": lon,
        },
    )
    d = tmp_path / "WetCHARTs" / "v1.3.1"
    d.mkdir(parents=True)
    ds.to_netcdf(d / "WetCHARTs_v1_3_1_2010.nc")
    return tmp_path


class TestWetCHARTs:
    @pytest.mark.parametrize("model", [1914, "1914"])
    def test_select_model(self, wetcharts_dir, model):
        w = inventories.WetCHARTs(model=model, inventory_dir=wetcharts_dir)
        assert float(w.data["wetlands"].max()) == 3.0

    @pytest.mark.parametrize("model, value", [(None, 2.0), ("median", 2.0)])
    def test_ensemble_statistic(self, wetcharts_dir, model, value):
        w = inventories.WetCHARTs(model=model, inventory_dir=wetcharts_dir)
        assert float(w.data["wetlands"].max()) == value

    def test_non_wetland_cells_are_zero(self, wetcharts_dir):
        w = inventories.WetCHARTs(model=1913, inventory_dir=wetcharts_dir)
        wetlands = w.data["wetlands"]
        assert not bool(wetlands.isnull().any())
        assert float(wetlands.isel(time=0, lat=0, lon=0)) == 0.0

    def test_resample_conserves_total(self, wetcharts_dir):
        # NaN cells must not poison the coarse cells they fall in
        pytest.importorskip("xesmf")
        w = inventories.WetCHARTs(model=1913, inventory_dir=wetcharts_dir)
        before = float(w.integrate().isel(time=0))
        after = float(w.resample(1.0).integrate().isel(time=0))
        assert after == pytest.approx(before, rel=0.01)
