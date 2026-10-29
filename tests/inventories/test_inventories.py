"""
Tests for lair.inventories.

Requires the `geo` extra (imports lair.geo) plus molmass. The base Inventory
machinery and unit/sector helpers are tested on synthetic data, and the concrete
loaders (EDGAR, EPA, GFEI, Vulcan, WetCHARTs) on tiny synthetic archives shaped
like the real files, including their time encodings (checked with ncdump -h).
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

    @staticmethod
    def _three_sectors():
        ds = xr.Dataset(
            {
                name: (("y", "x"), value * np.ones((2, 2)))
                for name, value in [("energy", 1.0), ("waste", 2.0), ("PostMeter", 4.0)]
            }
        )
        for v in ds.data_vars:
            ds[v].attrs["units"] = "kg/m2/s"
        return ds

    def test_exclude_leaves_sectors_out(self):
        total = inventories.sum_sectors(self._three_sectors(), exclude=["PostMeter"])
        assert np.unique(total.values).tolist() == [3.0]
        assert total.attrs["units"] == "kg/m2/s"

    def test_exclude_unknown_sector_raises(self):
        with pytest.raises(ValueError, match="Not sectors in this inventory"):
            inventories.sum_sectors(self._three_sectors(), exclude=["Post_Meter"])

    @staticmethod
    def _two_sectors(units_a, units_b):
        ds = xr.Dataset(
            {"a": (("y", "x"), np.ones((2, 2))), "b": (("y", "x"), np.ones((2, 2)))}
        )
        ds["a"].attrs["units"] = units_a
        ds["b"].attrs["units"] = units_b
        return ds

    def test_sectors_in_different_units_raise(self):
        # 1 kg + 1 g is 1.001 kg, not 2 "kg"
        ds = self._two_sectors("kg m-2 s-1", "g m-2 s-1")
        with pytest.raises(
            ValueError, match=r"different units.*a: kg m-2 s-1, b: g m-2"
        ):
            inventories.sum_sectors(ds)

    def test_a_sector_without_units_raises(self):
        ds = self._two_sectors("kg m-2 s-1", "kg m-2 s-1")
        del ds["b"].attrs["units"]
        with pytest.raises(ValueError, match="different units.*b: None"):
            inventories.sum_sectors(ds)

    def test_two_spellings_of_one_unit_are_the_same(self):
        ds = self._two_sectors("kg/m2/s", "kg m-2 s-1")
        total = inventories.sum_sectors(ds)
        assert np.unique(total.values).tolist() == [2.0]
        assert total.attrs["units"] == "kg/m2/s"

    def test_excluded_sector_may_differ(self):
        ds = self._two_sectors("kg m-2 s-1", "g m-2 s-1")
        ds["c"] = ds["a"] * 2
        ds["c"].attrs["units"] = "kg m-2 s-1"
        total = inventories.sum_sectors(ds, exclude=["b"])
        assert np.unique(total.values).tolist() == [3.0]
        assert total.attrs["units"] == "kg m-2 s-1"


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

    def test_pint_units_in_pint_units_out(self):
        da = xr.DataArray(np.ones((2, 2)), dims=["y", "x"]).pint.quantify(
            "mole/meter**2/second"
        )
        out = inventories.convert_units(da, "CH4", "kg/meter**2/second")
        assert out.pint.units is not None
        assert out.attrs["units"] == "kg/meter**2/second"

    def test_rejects_non_xarray(self):
        with pytest.raises(TypeError):
            inventories.convert_units([1, 2, 3], "CH4", "kg/m2/s")


class TestConvertUnitsPlain:
    """A plain array has a ``units`` attribute and no pint units (plan 001, step 10)."""

    # 1 nmol m-2 s-1 of CH4 in kg km-2 d-1: 1e-9 mol * 16.0425 g/mol * 1e-3 kg/g
    # * 1e6 m2/km2 * 86400 s/d
    PER_NMOL = 1e-9 * 16.0425e-3 * 1e6 * 86400

    @pytest.fixture
    def flux(self):
        return xr.DataArray(
            [[10.0, 20.0], [30.0, 40.0]],
            dims=("lat", "lon"),
            coords={
                "lat": ("lat", [40.0, 41.0], {"units": "degrees_north"}),
                "lon": ("lon", [-112.0, -111.0], {"units": "degrees_east"}),
            },
            name="flux",
            attrs={"units": "nmol m-2 s-1", "long_name": "methane flux"},
        )

    def test_array_in_array_out(self, flux):
        out = inventories.convert_units(flux, "CH4", "kg km-2 d-1")
        np.testing.assert_allclose(out.values, flux.values * self.PER_NMOL, rtol=1e-5)
        # Plain out, and only the units attribute and the values changed
        assert out.pint.units is None
        assert out.name == "flux"
        assert out.attrs == {"units": "kg km-2 d-1", "long_name": "methane flux"}
        xr.testing.assert_identical(out.lat, flux.lat)
        xr.testing.assert_identical(out.lon, flux.lon)

    def test_input_is_not_changed(self, flux):
        before = flux.copy(deep=True)
        inventories.convert_units(flux, "CH4", "kg km-2 d-1")
        xr.testing.assert_identical(flux, before)

    def test_same_as_a_pint_array(self, flux):
        plain = inventories.convert_units(flux, "CH4", "kg km-2 d-1")
        quantified = inventories.convert_units(
            flux.pint.quantify(), "CH4", "kg km-2 d-1"
        )
        np.testing.assert_allclose(plain.values, quantified.pint.dequantify().values)

    def test_there_and_back(self, flux):
        kg = inventories.convert_units(flux, "CH4", "kg km-2 d-1")
        back = inventories.convert_units(kg, "CH4", "nmol m-2 s-1")
        np.testing.assert_allclose(back.values, flux.values)
        assert back.attrs["units"] == "nmol m-2 s-1"

    def test_dask_stays_lazy(self, flux):
        out = inventories.convert_units(flux.chunk(), "CH4", "kg km-2 d-1")
        assert out.chunks is not None
        np.testing.assert_allclose(
            out.compute().values, flux.values * self.PER_NMOL, rtol=1e-5
        )

    def test_dataset_converts_every_variable(self, flux):
        ds = xr.Dataset({"a": flux, "b": 2 * flux}, attrs={"title": "fluxes"})
        ds["b"].attrs = flux.attrs.copy()
        out = inventories.convert_units(ds, "CH4", "kg km-2 d-1")
        for var, scale in [("a", 1), ("b", 2)]:
            np.testing.assert_allclose(
                out[var].values, scale * flux.values * self.PER_NMOL, rtol=1e-5
            )
            assert out[var].pint.units is None
            assert out[var].attrs["units"] == "kg km-2 d-1"
        assert out.attrs == {"title": "fluxes"}
        assert ds["a"].attrs["units"] == "nmol m-2 s-1"

    def test_array_without_units_raises(self, flux):
        flux.attrs.pop("units")
        with pytest.raises(ValueError, match=r"no units attribute on \['flux'\]"):
            inventories.convert_units(flux, "CH4", "kg km-2 d-1")
        with pytest.raises(ValueError, match=r"no units attribute on \['data'\]"):
            inventories.convert_units(flux.rename(None), "CH4", "kg km-2 d-1")

    def test_dataset_variable_without_units_raises(self, flux):
        ds = xr.Dataset({"a": flux, "b": xr.zeros_like(flux).drop_attrs()})
        with pytest.raises(ValueError, match=r"no units attribute on \['b'\]"):
            inventories.convert_units(ds, "CH4", "kg km-2 d-1")


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

    def test_absolute_emissions_attrs_describe_the_emissions(self, inventory):
        # The grid's area is multiplied in; nothing of the grid comes with it.
        # axis: 'Y' on a data variable marks it as the Y coordinate (issue #136).
        absolute = inventory.absolute_emissions
        for var in absolute.data_vars:
            assert absolute[var].attrs == {"units": "kg"}

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

    @pytest.mark.parametrize(
        "time_step, freq, src_units",
        [
            ("annual", "YS", "Mg km-2 a-1"),
            ("monthly", "MS", "Mg km-2 month-1"),
            ("daily", "D", "Mg km-2 d-1"),
            ("hourly", "h", "Mg km-2 hr-1"),
        ],
    )
    def test_rate_per_step_is_one_step(self, time_step, freq, src_units):
        # A rate per time step integrates over exactly one step: no round trip
        # through seconds and pint's Julian (365.25-day) year or month.
        import pandas as pd

        time = pd.date_range("2019-01-01", periods=2, freq=freq)
        if time_step == "annual":
            assert time.year.tolist() == [2019, 2020]  # non-leap and leap year
        ds = xr.Dataset(
            {"energy": (("time", "lat", "lon"), np.ones((2, 2, 2)))},
            coords={"time": time, "lat": [40.0, 41.0], "lon": [-112.0, -111.0]},
        )
        inv = inventories.Inventory(
            ds, pollutant="CH4", src_units=src_units, time_step=time_step
        )
        absolute = inv.absolute_emissions["energy"]
        assert absolute.attrs["units"] == "Mg"
        # 1 Mg km-2 per step over each gridcell's area in km2
        expected = np.broadcast_to(inv.gridcell_area.values, absolute.shape)
        np.testing.assert_allclose(absolute.values, expected, rtol=1e-12)

    def test_per_second_rate_uses_calendar_year(self, inventory):
        # Sub-step rates use the true calendar: 365 days in 2019, 366 in 2020.
        import pandas as pd

        time = pd.date_range("2019-01-01", periods=2, freq="YS")
        ds = inventory.data.pint.dequantify().assign_coords(time=time)
        inv = inventories.Inventory(ds, pollutant="CH4", src_units="kg/m**2/s")
        absolute = inv.absolute_emissions["energy"]
        per_m2 = absolute / (inv.gridcell_area * 1e6)
        np.testing.assert_allclose(
            per_m2.isel(lat=0, lon=0).values, [365 * 86400, 366 * 86400]
        )

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


def _flux_of_one(time_step, freq, src_units, start="2019-01-01", periods=2):
    """An inventory of 2 by 2 cells with a flux of 1 in ``src_units`` everywhere."""
    import pandas as pd

    time = pd.date_range(start, periods=periods, freq=freq)
    ds = xr.Dataset(
        {"energy": (("time", "lat", "lon"), np.ones((periods, 2, 2)))},
        coords={"time": time, "lat": [40.05, 40.15], "lon": [-111.95, -111.85]},
    )
    return inventories.Inventory(
        ds, pollutant="CH4", src_units=src_units, time_step=time_step
    )


def _values(inv):
    """The flux of the first cell in each time step."""
    return inv.data["energy"].values[:, 0, 0]


class TestConvertUnitsCalendar:
    """
    An inventory's convert_units gives a per-step unit the length of each step.

    A flux per year on annual data is per that year (365 or 366 days), a flux per
    month on monthly data per that month, not pint's year of 365.25 days and its
    twelfth. 2019 is not a leap year and 2020 is.
    """

    def test_per_second_to_per_year(self):
        inv = _flux_of_one("annual", "YS", "kg/m**2/s").convert_units("kg/m**2/yr")
        np.testing.assert_allclose(_values(inv), [365 * 86400, 366 * 86400])
        assert inv.data["energy"].attrs["units"] == "kg/m**2/yr"

    def test_per_year_to_per_second(self):
        inv = _flux_of_one("annual", "YS", "Mg km-2 a-1").convert_units("Mg km-2 s-1")
        np.testing.assert_allclose(
            _values(inv), [1 / (365 * 86400), 1 / (366 * 86400)], rtol=1e-12
        )

    def test_per_year_round_trip(self):
        inv = _flux_of_one("annual", "YS", "Mg km-2 a-1")
        back = inv.convert_units("kg m-2 s-1").convert_units("Mg km-2 a-1")
        np.testing.assert_allclose(_values(back), [1.0, 1.0], rtol=1e-12)

    def test_per_year_to_per_year(self):
        inv = _flux_of_one("annual", "YS", "Mg km-2 a-1").convert_units("Gg km-2 yr-1")
        np.testing.assert_allclose(_values(inv), [1e-3, 1e-3], rtol=1e-12)

    def test_per_second_to_per_month(self):
        inv = _flux_of_one("monthly", "MS", "kg/m**2/s", start="2020-01-01")
        inv = inv.convert_units("kg/m**2/month")
        np.testing.assert_allclose(_values(inv), [31 * 86400, 29 * 86400])

    def test_unit_of_another_length_uses_pint(self):
        # Per day on annual data is not per step: pint's day, as before
        inv = _flux_of_one("annual", "YS", "kg/m**2/s").convert_units("kg/m**2/d")
        np.testing.assert_allclose(_values(inv), [86400, 86400])

    def test_mass_to_amount_per_step(self):
        # The mass_flux context still applies: 1 kg of CH4 is 1000 / 16.04 mol
        inv = _flux_of_one("annual", "YS", "kg/m**2/s").convert_units("mol/m**2/yr")
        mw = float(inventories.molecular_weight("CH4").magnitude)
        np.testing.assert_allclose(
            _values(inv), np.array([365, 366]) * 86400 * 1000 / mw, rtol=1e-12
        )

    def test_total_does_not_depend_on_the_unit(self):
        inv = _flux_of_one("annual", "YS", "kg/m**2/s")
        kg = inv.absolute_emissions["energy"].values
        gg = inv.convert_units("Gg/km**2/yr").absolute_emissions["energy"].values
        np.testing.assert_allclose(gg * 1e6, kg, rtol=1e-12)

    def test_standardize_units_uses_the_calendar(self):
        import pandas as pd

        time = pd.date_range("2019-01-01", periods=2, freq="YS")
        ds = xr.Dataset(
            {"energy": (("time", "lat", "lon"), np.ones((2, 2, 2)))},
            coords={"time": time, "lat": [40.05, 40.15], "lon": [-111.95, -111.85]},
        )
        inv = inventories.Inventory(
            ds, pollutant="CH4", src_units="Mg km-2 a-1", standardize_units=True
        )
        # 1 Mg km-2 in the year, per day of that year, in kg km-2 d-1
        np.testing.assert_allclose(_values(inv), [1e3 / 365, 1e3 / 366], rtol=1e-12)

    def test_plain_array_keeps_pints_year(self):
        # convert_units of a plain array has no time steps
        flux = xr.DataArray([1.0], dims="x", attrs={"units": "kg/m**2/s"})
        out = inventories.convert_units(flux, "CH4", "kg/m**2/yr")
        np.testing.assert_allclose(out.values, [365.25 * 86400])


# --- Attributes ----------------------------------------------------------------
# lair.inventories must not change xarray's global options (GitHub issue #38):
# the operations that need attributes kept set keep_attrs themselves.


def test_import_leaves_keep_attrs_default():
    assert xr.get_options()["keep_attrs"] == "default"


def test_import_does_not_change_user_arithmetic():
    # A kg m-2 s-1 flux times an area must behave exactly as without lair
    # (older xarray drops the attrs, newer keeps them; either way, the same)
    import subprocess
    import sys

    code = (
        "import numpy as np, xarray as xr\n"
        "flux = xr.DataArray(np.ones(2), dims='x', attrs={'units': 'kg m-2 s-1'})\n"
        "print((flux * 4.0).attrs)\n"
        "import lair.inventories\n"
        "print((flux * 4.0).attrs)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert out[0] == out[1]


@pytest.fixture
def described():
    """
    An Inventory whose variables carry long_name/standard_name as the loaders' do.
    """
    import pandas as pd

    time = pd.date_range("2020-01-01", periods=2, freq="YS")
    shp = (2, 4, 4)  # 0.5 degree cells
    lat = ("lat", 40.25 + 0.5 * np.arange(4), {"units": "degrees_north"})
    lon = ("lon", -112.75 + 0.5 * np.arange(4), {"units": "degrees_east"})
    ds = xr.Dataset(
        {
            "energy": (("time", "lat", "lon"), np.ones(shp)),
            "agriculture": (("time", "lat", "lon"), 2 * np.ones(shp)),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    for var in ds.data_vars:
        ds[var].attrs = {
            "units": "kg/m**2/s",
            "long_name": f"{var}_Emissions",
            "standard_name": "annual_CH4_emissions",
        }
    return inventories.Inventory(ds, pollutant="CH4")


def _assert_described(data, units):
    for var in data.data_vars:
        attrs = data[var].attrs
        assert attrs["long_name"] == f"{var}_Emissions", var
        assert attrs["units"] == units, var


class TestAttributes:
    def test_data(self, described):
        _assert_described(described.data, "kg/m**2/s")
        assert described.data["energy"].attrs["standard_name"] == (
            "annual_CH4_emissions"
        )

    def test_total_emissions(self, described):
        total = described.total_emissions
        assert total.attrs == {"long_name": "Total Emissions", "units": "kg/m**2/s"}

    def test_collapsed(self, described):
        assert described.collapsed.attrs["units"] == "kg/m**2/s"

    def test_absolute_emissions(self, described):
        absolute = described.absolute_emissions
        assert absolute.attrs["long_name"] == "Absolute Emissions"
        assert absolute.attrs["standard_name"] == "annual_emissions_per_gridcell"
        _assert_described(absolute, "kg")

    def test_integrate(self, described):
        total = described.integrate()
        # Not absolute_emissions' "annual_emissions_per_gridcell": cells are summed
        assert total.attrs == {"long_name": "Total Emissions", "units": "kg"}
        assert total.dims == ("time",)

    def test_convert_units(self, described):
        converted = described.convert_units("mol/km**2/d")
        _assert_described(converted.data, "mol/km**2/d")

    def test_clip(self, described):
        clipped = described.clip(bbox=(-113.0, 40.0, -112.0, 41.0))
        assert clipped.data.sizes["lat"] == 2
        _assert_described(clipped.data, "kg/m**2/s")

    def test_resample(self, described):
        pytest.importorskip("xesmf")
        coarse = described.resample(1.0)
        assert coarse.data.sizes["lat"] == 2
        _assert_described(coarse.data, "kg/m**2/s")

    def test_resample_keeps_total_when_resolution_does_not_divide(self, described):
        # 2 degrees of latitude and longitude in 0.9 degree cells: the third cell
        # reaches past the edge (it is a fifth inside), and its emissions count
        pytest.importorskip("xesmf")
        coarse = described.resample(0.9)
        assert coarse.data.sizes["lat"] == 3
        before = described.integrate().values
        after = coarse.integrate().values
        np.testing.assert_allclose(after, before, rtol=1e-3)

    def test_regrid(self, described):
        pytest.importorskip("xesmf")
        out_grid = xr.Dataset(
            coords={
                "lat": ("lat", [40.5, 41.5], {"units": "degrees_north"}),
                "lon": ("lon", [-112.5, -111.5], {"units": "degrees_east"}),
            }
        )
        regridded = described.regrid(out_grid)
        _assert_described(regridded.data, "kg/m**2/s")

    def test_plot_labels_units(self, described):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ax = described.plot()
        colorbar = ax.figure.axes[-1]
        assert colorbar.get_ylabel() == "Total Emissions [kg/m**2/s]"
        plt.close(ax.figure)

    def test_wetcharts_ensemble_mean(self, wetcharts_dir):
        w = inventories.WetCHARTs(inventory_dir=wetcharts_dir)
        attrs = w.data["wetlands"].attrs
        assert attrs["long_name"] == "Wetland_CH4_Emissions"
        assert attrs["standard_name"] == "monthly_CH4_emissions"
        assert attrs["units"] == "mg/m**2/d"

    def test_vulcan(self, vulcan_dir):
        v = inventories.Vulcan(inventory_dir=vulcan_dir)
        attrs = v.data["onroad"].attrs
        assert attrs["units"] == "Mg/km**2/yr"
        assert "tonnes of CO2" in attrs["comment"]
        lower = v.get_uncertainties("lower")
        assert lower["onroad"].attrs["units"] == "Mg km-2 year-1"
        assert "tonnes of CO2" in lower["onroad"].attrs["comment"]

    def test_epa_v2_monthly(self, epa_v2_dir):
        monthly = inventories.EPAv2(scale_by_month=True, inventory_dir=epa_v2_dir)
        for var in monthly.data.data_vars:
            attrs = monthly.data[var].attrs
            assert attrs["long_name"] == f"{var}_Emissions", var
            assert attrs["IPCC_Code"], var


class TestInventoryDir:
    def test_unset_env_raises(self, monkeypatch):
        monkeypatch.delenv("LAIR_INVENTORY_DIR", raising=False)
        with pytest.raises(ValueError, match="LAIR_INVENTORY_DIR"):
            inventories.EDGARv8("CH4")


VULCAN_SECTORS = ["onroad", "elec_prod"]

#: Vulcan files are tC; lair loads them as CO2 mass (M(CO2)/M(C))
C_TO_CO2 = 44.0095 / 12.0107


def _vulcan_grid():
    """A 10 x 8 km patch of the Vulcan LCC grid with its 2D lat/lon."""
    from pyproj import Transformer

    x = np.arange(-1.5e6, -1.5e6 + 10_000, 1000.0)  # 10 x 1 km cells
    y = np.arange(4e5, 4e5 + 8_000, 1000.0)  # 8 x 1 km cells
    to_ll = Transformer.from_crs(
        inventories.Vulcan.native_crs, "EPSG:4326", always_xy=True
    )
    lon, lat = to_ll.transform(*np.meshgrid(x, y))
    return x, y, lat, lon


def _cf_time(starts, step, units):
    """
    Time and time_bnds variables encoded like the Vulcan/WetCHARTs files.

    ``time`` is at the middle of each step with ``bounds = "time_bnds"``, and the
    bounds without units of their own (CF: they take those of ``time``).
    """
    starts = np.asarray(starts, dtype=float)
    bnds = np.stack([starts, starts + step], axis=1)
    time = (
        "time",
        bnds.mean(axis=1),
        {"units": units, "calendar": "standard", "bounds": "time_bnds"},
    )
    return time, (("time", "nv"), bnds)


def _write_vulcan(root, time_step, sectors, bounds=("mn",)):
    """
    Write a tiny Vulcan v3 archive shaped like the real files.

    It is (time, y, x) on the Vulcan LCC grid with 2D lat/lon coords, one file per
    sector and bound (annual) or per sector and day (hourly).
    """
    x, y, lat, lon = _vulcan_grid()
    d = root / "vulcan" / "v3" / "data" / "native" / time_step
    d.mkdir(parents=True)
    if time_step == "annual":
        # 2014 and 2015: days since 2010-01-01, labeled mid-year
        files = {
            (sector, bound, value): _cf_time(
                [1461, 1826], [365, 365], "days since 2010-01-01 00:00:00 UTC"
            )
            for sector in sectors
            for bound, value in [("mn", 2.0), ("lo", 1.0), ("hi", 3.0)]
            if bound in bounds
        }
        names = {k: f"Vulcan_v3_US_annual_1km_{k[0]}_{k[1]}.nc4" for k in files}
    else:
        # 2015-01-01 and 02, 24 hours each, labeled at the half hour
        files, names = {}, {}
        for sector in sectors:
            for day in (1, 2):
                key = (sector, day, 2.0)
                start = 43824 + 24 * (day - 1)  # hours since 2010-01-01
                files[key] = _cf_time(
                    start + np.arange(24), 1, "hours since 2010-01-01 00:00:00 UTC"
                )
                names[key] = f"Vulcan.v3.US.hourly.1km.{sector}.mn.2015.d{day:03d}.nc4"
    for key, (time, time_bnds) in files.items():
        sector, _, value = key
        nt = time[1].size
        emis = np.full((nt, y.size, x.size), value)
        if sector == "elec_prod":
            # Point-source sectors are NaN except where there are sources
            emis[:] = np.nan
            emis[:, 4, 5] = 100 * value
        ds = xr.Dataset(
            {
                "carbon_emissions": (("time", "y", "x"), emis),
                "time_bnds": time_bnds,
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
        ds.to_netcdf(d / names[key])
    return root


@pytest.fixture
def vulcan_dir(tmp_path):
    """A tiny annual Vulcan v3 archive (2014-2015, all three bounds)."""
    return _write_vulcan(
        tmp_path, "annual", VULCAN_SECTORS + ["total"], ("mn", "lo", "hi")
    )


@pytest.fixture
def vulcan_hourly_dir(tmp_path):
    """A tiny hourly Vulcan v3 archive (2015-01-01 and 02, central estimate)."""
    return _write_vulcan(tmp_path, "hourly", VULCAN_SECTORS)


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
        bbox = (-1.5e6, 4e5, -1.5e6 + 4_000, 4e5 + 3_000)  # data CRS (meters)
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
        # NaN "no emission" cells must not wipe out neighboring point sources
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
    """
    ``scale_by_month`` keeps every sector.

    Those with monthly scale factors are scaled, the rest (enteric fermentation,
    landfills, coal, ...) hold their annual rate in every month.
    """

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
    """
    A tiny EPA v2 archive shaped like the real files.

    It has one annual file per year with ``emi_ch4_<code>_<name>`` variables, and
    monthly scale factors for only some of the sectors.
    """
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


def _edgar_v8_file(path, long_name, year, monthly, value=1.0):
    import pandas as pd

    lat, lon = np.array([40.05, 40.15]), np.array([-111.95, -111.85])
    attrs = {"long_name": long_name, "year": str(year), "units": "kg m-2 s-1"}
    if monthly:
        # Real monthly files label each month by its 15th, as float days since
        # Jan 1 (2000: 14, 45, 74, ...), and have no time bounds
        mid = pd.date_range(f"{year}-01-01", periods=12, freq="MS") + pd.Timedelta(
            "14D"
        )
        days = (mid - pd.Timestamp(f"{year}-01-01")).days.to_numpy(dtype="float32")
        time = ("time", days, {"units": f"days since {year}-01-01 00:00:00"})
        fluxes = (("time", "lat", "lon"), value * np.ones((12, 2, 2)), attrs)
        coords = {"time": time, "lat": lat, "lon": lon}
    else:
        fluxes = (("lat", "lon"), value * np.ones((2, 2)), attrs)
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


# --- The years to read -------------------------------------------------------


def _edgar_years_archive(root, years, unreadable=()):
    """
    EDGAR v7, v8 annual and v8 monthly CH4 files for each of ``years``.

    Every flux in a year's files is that year minus 2000, so a year is told
    from another by its values. A file of each year in ``unreadable`` is not
    NetCDF: reading one raises, so a test that does not read it never opened it.
    """
    for year in [*years, *unreadable]:
        files = [
            (root / "EDGAR/v7/CH4" / code / f"v7.0_FT2021_CH4_{year}_{code}.0.1x0.1.nc")
            for code in ("ENF", "AGS")
        ] + [
            (root / base / code / f"v8.0_FT2022_GHG_CH4_{year}_{code}_flx.nc")
            for base, codes in [
                ("EDGAR/v8/CH4", ("ENF", "AGS")),
                ("EDGAR/v8/monthly/CH4", ("AGRICULTURE", "WASTE")),
            ]
            for code in codes
        ]
        for f in files:
            f.parent.mkdir(parents=True, exist_ok=True)
            if year in unreadable:
                f.write_bytes(b"not a NetCDF file")
            elif "/v7/" in str(f):
                xr.Dataset(
                    {
                        "emi_ch4": (
                            ("lat", "lon"),
                            (year - 2000) * np.ones((2, 2)),
                            {"units": "kg m-2 s-1"},
                        )
                    },
                    coords={"lat": [40.05, 40.15], "lon": [-111.95, -111.85]},
                ).to_netcdf(f)
            else:
                _edgar_v8_file(
                    f, f.parent.name, year, "monthly" in str(f), value=year - 2000
                )
    return root


#: name -> constructor taking the archive and the keywords start and end
EDGAR_YEARS_CASES = {
    "EDGARv7": lambda d, **years: inventories.EDGARv7("CH4", inventory_dir=d, **years),
    "EDGARv8 annual": lambda d, **years: inventories.EDGARv8(
        "CH4", inventory_dir=d, **years
    ),
    "EDGARv8 monthly": lambda d, **years: inventories.EDGARv8(
        "CH4", time_step="monthly", inventory_dir=d, **years
    ),
}


@pytest.mark.parametrize("case", list(EDGAR_YEARS_CASES))
class TestEDGARYears:
    def test_default_reads_every_year(self, tmp_path, case):
        _edgar_years_archive(tmp_path, [2018, 2019, 2020])
        inv = EDGAR_YEARS_CASES[case](tmp_path)
        assert sorted(set(inv.data.indexes["time"].year)) == [2018, 2019, 2020]

    def test_opens_only_the_years_asked(self, tmp_path, case):
        # The unreadable files are the years before and after: reading either
        # raises, so these files were not opened
        _edgar_years_archive(tmp_path, [2018, 2019, 2020], unreadable=[2017, 2021])
        inv = EDGAR_YEARS_CASES[case](tmp_path, start=2018, end=2020)
        assert sorted(set(inv.data.indexes["time"].year)) == [2018, 2019, 2020]
        with pytest.raises(OSError):
            EDGAR_YEARS_CASES[case](tmp_path)

    def test_one_year(self, tmp_path, case):
        _edgar_years_archive(tmp_path, [2018, 2019, 2020], unreadable=[2017, 2021])
        inv = EDGAR_YEARS_CASES[case](tmp_path, start=2019, end=2019)
        assert set(inv.data.indexes["time"].year) == {2019}
        # each sector holds that year's value only
        for sector in inv.data.data_vars:
            assert set(inv.data[sector].pint.dequantify().values.ravel()) == {19.0}

    def test_same_data_as_the_years_of_a_full_read(self, tmp_path, case):
        _edgar_years_archive(tmp_path, [2018, 2019, 2020])
        full = EDGAR_YEARS_CASES[case](tmp_path)
        some = EDGAR_YEARS_CASES[case](tmp_path, start=2019, end=2020)
        want = full.data.sel(time=slice("2019-01-01", "2020-12-31"))
        assert set(some.data.data_vars) == set(want.data_vars)
        assert some.data.indexes["time"].equals(want.indexes["time"])
        for sector in want.data_vars:
            np.testing.assert_array_equal(
                some.data[sector].pint.dequantify().values,
                want[sector].pint.dequantify().values,
            )

    def test_open_ended(self, tmp_path, case):
        _edgar_years_archive(tmp_path, [2018, 2019, 2020])
        since = EDGAR_YEARS_CASES[case](tmp_path, start=2019)
        until = EDGAR_YEARS_CASES[case](tmp_path, end=2019)
        assert sorted(set(since.data.indexes["time"].year)) == [2019, 2020]
        assert sorted(set(until.data.indexes["time"].year)) == [2018, 2019]

    def test_a_year_without_files_is_logged(self, tmp_path, case, caplog):
        _edgar_years_archive(tmp_path, [2018, 2019, 2020])
        with caplog.at_level("WARNING", logger="lair.inventories"):
            inv = EDGAR_YEARS_CASES[case](tmp_path, start=2019, end=2022)
        assert sorted(set(inv.data.indexes["time"].year)) == [2019, 2020]
        assert "no files for 2021 to 2022" in caplog.text

    def test_no_warning_when_every_year_has_files(self, tmp_path, case, caplog):
        _edgar_years_archive(tmp_path, [2018, 2019, 2020])
        with caplog.at_level("WARNING", logger="lair.inventories"):
            EDGAR_YEARS_CASES[case](tmp_path, start=2018, end=2019)
            EDGAR_YEARS_CASES[case](tmp_path)
        assert caplog.text == ""

    def test_a_gap_between_years_is_logged(self, tmp_path, case, caplog):
        _edgar_years_archive(tmp_path, [2017, 2020])
        with caplog.at_level("WARNING", logger="lair.inventories"):
            EDGAR_YEARS_CASES[case](tmp_path, start=2017, end=2020)
        assert "no files for 2018 to 2019" in caplog.text

    def test_no_file_for_the_years_raises(self, tmp_path, case):
        _edgar_years_archive(tmp_path, [2018, 2019, 2020])
        with pytest.raises(FileNotFoundError, match=r"start=2021.*years with files"):
            EDGAR_YEARS_CASES[case](tmp_path, start=2021, end=2025)
        with pytest.raises(FileNotFoundError, match="end=2010"):
            EDGAR_YEARS_CASES[case](tmp_path, end=2010)

    def test_start_after_end_raises(self, tmp_path, case):
        _edgar_years_archive(tmp_path, [2018, 2019, 2020])
        with pytest.raises(FileNotFoundError, match="start=2020, end=2018"):
            EDGAR_YEARS_CASES[case](tmp_path, start=2020, end=2018)

    @pytest.mark.parametrize("year", ["2019", 2019.0, "2019-01-01"])
    def test_a_year_is_an_integer(self, tmp_path, case, year):
        _edgar_years_archive(tmp_path, [2018, 2019, 2020])
        with pytest.raises(TypeError, match="start is a year"):
            EDGAR_YEARS_CASES[case](tmp_path, start=year)
        with pytest.raises(TypeError, match="end is a year"):
            EDGAR_YEARS_CASES[case](tmp_path, end=year)

    def test_numpy_integers_are_years(self, tmp_path, case):
        _edgar_years_archive(tmp_path, [2018, 2019, 2020])
        inv = EDGAR_YEARS_CASES[case](
            tmp_path, start=np.int64(2019), end=np.int32(2019)
        )
        assert set(inv.data.indexes["time"].year) == {2019}


class TestEDGARYearHelpers:
    def test_year_is_not_the_release_tag(self):
        for name, year in [
            ("v8.0_FT2022_GHG_CH4_1970_AGS_flx.nc", "1970"),
            ("v8.0_FT2022_GHG_CH4_2000_PRO_COAL_flx.nc", "2000"),
            ("v7.0_FT2021_CH4_2021_TNR_Aviation_CDS.0.1x0.1.nc", "2021"),
        ]:
            assert inventories._EDGAR_YEAR.search(name).group(1) == year

    @pytest.mark.parametrize(
        "years, text",
        [
            ([2020], "2020"),
            ([2020, 2021], "2020 to 2021"),
            ([1970, 1971, 1972, 2020], "1970 to 1972, 2020"),
            ([2001, 2003, 2004, 2008], "2001, 2003 to 2004, 2008"),
            ([], ""),
        ],
    )
    def test_year_spans(self, years, text):
        assert inventories._year_spans(years) == text


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

    def test_totals_of_mixed_units_raise(self):
        # kg + g added as plain numbers gave 2 kg instead of 1.001 kg
        inv = inventories.Inventory(self._ds("g m-2 s-1"), pollutant="CH4")
        for use in (
            lambda: inv.total_emissions,
            lambda: inv.get_units(),
            lambda: inv.absolute_emissions,
            lambda: inv.integrate(),
        ):
            with pytest.raises(ValueError, match="different units"):
                use()

    def test_message_names_each_sectors_units(self):
        inv = inventories.Inventory(self._ds("g m-2 s-1"), pollutant="CH4")
        with pytest.raises(ValueError, match=r"a: kg/m\*\*2/s, b: g/m\*\*2/s"):
            inv.get_units()

    def test_convert_units_then_totals(self):
        pytest.importorskip("xesmf")
        inv = inventories.Inventory(self._ds("g m-2 s-1"), pollutant="CH4")
        inv = inv.convert_units("kg m-2 s-1")
        assert inv.get_units() == ("kg", "m**2", "s")
        total = inv.total_emissions
        assert np.unique(total.values).tolist() == pytest.approx([1.001])
        assert total.attrs["units"] == "kg/m**2/s"
        assert float(inv.integrate().isel(time=0)) > 0

    def test_standardize_units_then_totals(self):
        inv = inventories.Inventory(
            self._ds("g m-2 s-1"), pollutant="CH4", standardize_units=True
        )
        total = inv.total_emissions
        assert total.attrs["units"] == "kg/km**2/d"
        # 1 kg + 1 g per m2 per s, in kg per km2 per day
        assert np.unique(total.values).tolist() == pytest.approx([1.001 * 1e6 * 86400])


@pytest.fixture
def wetcharts_dir(tmp_path):
    """
    A tiny WetCHARTs v1.3.1 file.

    It has int32 model codes, mid-month times and NaN outside wetlands (most of the real
    grid).
    """
    import pandas as pd

    lat = np.arange(40.25, 42, 0.5)
    lon = np.arange(-112.75, -111, 0.5)
    # Like the real files: int days since 2001-01-01, `time` at mid-month
    # ("middle of each month") and bounds with units of their own
    units = "days since 2001-01-01 00:00:00"
    edges = pd.date_range("2010-01-01", periods=13, freq="MS")
    edges = (edges - pd.Timestamp("2001-01-01")).days.to_numpy(dtype="int32")
    bnds = np.stack([edges[:-1], edges[1:]], axis=1)
    mid = bnds.mean(axis=1).astype("int32")
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
            "time_bnds": (("time", "nv"), bnds, {"units": units}),
            "crs": ((), "a"),
        },
        coords={
            "model": np.array([1913, 1914], dtype="int32"),
            "time": (
                "time",
                mid,
                {"units": units, "bounds": "time_bnds", "calendar": "standard"},
            ),
            "lat": lat,
            "lon": lon,
        },
    )
    d = tmp_path / "WetCHARTs" / "v1.3.1"
    d.mkdir(parents=True)
    ds.to_netcdf(d / "WetCHARTs_v1_3_1_2010.nc")
    return tmp_path


class TestWetCHARTs:
    @pytest.mark.parametrize("model", [1914, "1914"], ids=["int", "str"])
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


# --- Time labels -------------------------------------------------------------
# Every inventory labels each time step by the START of its period (GitHub
# issue #37): annual -> Jan 1, monthly -> the 1st, daily -> 00:00, hourly ->
# the top of the hour. Providers differ (EDGAR monthly at the 15th, Vulcan and
# WetCHARTs at the middle of each step), so sel(time="2020-01-01") only works
# for all of them if lair relabels.


@pytest.fixture
def edgar_dir(tmp_path):
    """EDGAR v7 annual, v8 annual and v8 monthly CH4 files for 2020."""
    d7 = tmp_path / "EDGAR" / "v7" / "CH4" / "ENF"
    d7.mkdir(parents=True)
    xr.Dataset(
        {"emi_ch4": (("lat", "lon"), np.ones((2, 2)), {"units": "kg m-2 s-1"})},
        coords={"lat": [40.05, 40.15], "lon": [-111.95, -111.85]},
    ).to_netcdf(d7 / "v7.0_FT2021_CH4_2020_ENF.0.1x0.1.nc")
    d8 = tmp_path / "EDGAR" / "v8"
    f = d8 / "CH4" / "ENF" / "v8.0_FT2022_GHG_CH4_2020_ENF_flx.nc"
    _edgar_v8_file(f, "Enteric fermentation", 2020, monthly=False)
    for code, name in [
        ("FUEL_EXPLOITATION", "Fuel exploitation"),
        ("AGRICULTURE", "Agriculture"),
    ]:
        f = d8 / "monthly" / "CH4" / code / f"v8.0_FT2022_GHG_CH4_2020_{code}_flx.nc"
        _edgar_v8_file(f, name, 2020, monthly=True)
    return tmp_path


@pytest.fixture
def epa_v1_dir(tmp_path):
    """
    EPA v1 (2012) annual, monthly and daily files.

    Like the real ones, the monthly and daily files number their steps 1..n with units
    of "months".
    """
    lat, lon = np.array([40.05, 40.15]), np.array([-111.95, -111.85])
    d = tmp_path / "EPA" / "v1"
    d.mkdir(parents=True)
    var = "emissions_1A_Combustion_Mobile"
    xr.Dataset(
        {var: (("lat", "lon"), np.ones((2, 2)))}, coords={"lat": lat, "lon": lon}
    ).to_netcdf(d / "GEPA_Annual.nc")
    for name, n in [("Monthly", 12), ("Daily", 366)]:
        time = ("time", np.arange(1, n + 1, dtype="float32"), {"units": "months"})
        xr.Dataset(
            {var: (("time", "lat", "lon"), np.ones((n, 2, 2)))},
            coords={"time": time, "lat": lat, "lon": lon},
        ).to_netcdf(d / f"GEPA_{name}.nc")
    return tmp_path


@pytest.fixture
def gfei_dir(tmp_path):
    """GFEI v2 (2019) files: one `emis_ch4` per file, the year as an attribute."""
    d = tmp_path / "GFEI" / "v2"
    d.mkdir(parents=True)
    lat = ("lat", [40.05, 40.15], {"units": "degrees_north"})
    lon = ("lon", [-111.95, -111.85], {"units": "degrees_east"})
    for sector in ["Coal", "Gas_Production", "Total_Fuel_Exploitation"]:
        xr.Dataset(
            {"emis_ch4": (("lat", "lon"), np.ones((2, 2)))},
            coords={"lat": lat, "lon": lon},
            attrs={"year": "2019"},
        ).to_netcdf(d / f"Global_Fuel_Exploitation_Inventory_v2_2019_{sector}.nc")
    return tmp_path


#: name -> (archive fixture, constructor)
INVENTORY_CASES = {
    "EDGARv7": ("edgar_dir", lambda d: inventories.EDGARv7("CH4", inventory_dir=d)),
    "EDGARv8 annual": (
        "edgar_dir",
        lambda d: inventories.EDGARv8("CH4", inventory_dir=d),
    ),
    "EDGARv8 monthly": (
        "edgar_dir",
        lambda d: inventories.EDGARv8("CH4", time_step="monthly", inventory_dir=d),
    ),
    "EPAv1 annual": ("epa_v1_dir", lambda d: inventories.EPAv1(inventory_dir=d)),
    "EPAv1 monthly": (
        "epa_v1_dir",
        lambda d: inventories.EPAv1("Monthly", inventory_dir=d),
    ),
    "EPAv1 daily": (
        "epa_v1_dir",
        lambda d: inventories.EPAv1("Daily", inventory_dir=d),
    ),
    "EPAv2 annual": ("epa_v2_dir", lambda d: inventories.EPAv2(inventory_dir=d)),
    "EPAv2 monthly": (
        "epa_v2_dir",
        lambda d: inventories.EPAv2(scale_by_month=True, inventory_dir=d),
    ),
    "GFEIv2": ("gfei_dir", lambda d: inventories.GFEIv2(inventory_dir=d)),
    "Vulcan annual": ("vulcan_dir", lambda d: inventories.Vulcan(inventory_dir=d)),
    "Vulcan hourly": (
        "vulcan_hourly_dir",
        lambda d: inventories.Vulcan("hourly", inventory_dir=d),
    ),
    "WetCHARTs": ("wetcharts_dir", lambda d: inventories.WetCHARTs(inventory_dir=d)),
}

#: pandas period alias of each time step
_PERIOD = {"annual": "Y", "monthly": "M", "daily": "D", "hourly": "h"}


def _load(request, case):
    fixture, make = INVENTORY_CASES[case]
    return make(request.getfixturevalue(fixture))


class TestTimeLabels:
    @pytest.mark.parametrize("case", list(INVENTORY_CASES))
    def test_labels_are_period_starts(self, request, case):
        inv = _load(request, case)
        t = inv.data.indexes["time"]
        starts = t.to_period(_PERIOD[inv.time_step]).to_timestamp()
        assert t.equals(starts), t[:3]

    @pytest.mark.parametrize(
        "case, day",
        [
            ("EDGARv8 monthly", "2020-01-01"),
            ("EPAv1 monthly", "2012-01-01"),
            ("EPAv2 monthly", "2018-01-01"),
            ("WetCHARTs", "2010-01-01"),
        ],
    )
    def test_sel_first_of_month(self, request, case, day):
        inv = _load(request, case)
        assert inv.time_step == "monthly"
        jan = inv.data.sel(time=day)
        assert "time" not in jan.dims  # an exact match, not a partial slice
        assert jan.time.values == np.datetime64(day)

    def test_sel_top_of_hour(self, vulcan_hourly_dir):
        v = inventories.Vulcan("hourly", inventory_dir=vulcan_hourly_dir)
        assert v.data.sizes["time"] == 48
        one_am = v.data.sel(time=np.datetime64("2015-01-02T01:00"))
        assert "time" not in one_am.dims

    @pytest.mark.parametrize(
        "case, provider_offset",
        [
            ("EDGARv8 monthly", "14D"),  # the 15th of each month
            ("Vulcan hourly", "30min"),  # the half hour
            ("WetCHARTs", "15D"),  # mid-month
        ],
    )
    def test_totals_do_not_depend_on_labels(self, request, case, provider_offset):
        # Relabeling is only a relabeling: absolute_emissions and integrate
        # give bit-identical totals with the provider's mid-step labels.
        import pandas as pd

        inv = _load(request, case)
        provider = inv.copy()
        provider._data = inv._data.assign_coords(
            time=inv._data.indexes["time"] + pd.Timedelta(provider_offset)
        )
        np.testing.assert_array_equal(
            provider.integrate().values, inv.integrate().values
        )
        for var in inv.data.data_vars:
            np.testing.assert_array_equal(
                provider.absolute_emissions[var].values,
                inv.absolute_emissions[var].values,
            )


# --- Reading files with the base class -------------------------------------


def _write_flux_file(path, name, value):
    """A (time, lat, lon) flux of ``value`` in one variable, for 2020 and 2021."""
    import pandas as pd

    xr.Dataset(
        {name: (("time", "lat", "lon"), np.full((2, 2, 2), value))},
        coords={
            "time": pd.date_range("2020-01-01", periods=2, freq="YS"),
            "lat": [40.05, 40.15],
            "lon": [-111.95, -111.85],
        },
    ).to_netcdf(path)


class TestInventoryFromFiles:
    def test_a_folder_of_files_is_read_together(self, tmp_path):
        _write_flux_file(tmp_path / "energy.nc", "energy", 1.0)
        _write_flux_file(tmp_path / "waste.nc", "waste", 4.0)
        inv = inventories.Inventory(
            str(tmp_path), pollutant="CH4", src_units="kg/m**2/s"
        )
        assert inv.path == str(tmp_path)
        assert set(inv.data.data_vars) == {"energy", "waste"}
        assert (inv.data["energy"] == 1.0).all()
        assert (inv.total_emissions == 5.0).all()

    def test_one_file(self, tmp_path):
        _write_flux_file(tmp_path / "energy.nc", "energy", 3.0)
        inv = inventories.Inventory(
            str(tmp_path / "energy.nc"), pollutant="CH4", src_units="kg/m**2/s"
        )
        assert list(inv.data.data_vars) == ["energy"]
        assert (inv.data["energy"] == 3.0).all()

    def test_a_folder_without_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No Inventory files found"):
            inventories.Inventory(str(tmp_path), pollutant="CH4", src_units="kg/m**2/s")

    def test_data_that_is_not_a_path_or_dataset_raises(self):
        with pytest.raises(ValueError, match="path to a file or an xarray Dataset"):
            inventories.Inventory(123, pollutant="CH4", src_units="kg/m**2/s")

    def test_a_dataset_has_no_files(self, inventory):
        assert inventory.path is None
        with pytest.raises(ValueError, match="built from a Dataset and has no files"):
            inventory._file_root()


class TestInventoryAccessors:
    def test_get_units_as_quantities(self, inventory):
        from lair import units

        quantity, area, time = inventory.get_units(pint=True)
        assert (quantity, area, time) == (units("kg"), units("m**2"), units("s"))

    def test_quantify_has_pint_units(self, inventory):
        from lair import units

        quantified = inventory.quantify()
        assert quantified["energy"].pint.units == units.Unit("kg/m**2/s")
        assert float(quantified["agriculture"].pint.magnitude.max()) == 2.0

    @pytest.fixture
    def plotted(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    @staticmethod
    def _steps():
        # energy 1 in 2020 and 5 in 2021; agriculture is twice energy
        import pandas as pd

        energy = np.stack([np.full((2, 2), 1.0), np.full((2, 2), 5.0)])
        ds = xr.Dataset(
            {
                "energy": (("time", "lat", "lon"), energy),
                "agriculture": (("time", "lat", "lon"), 2 * energy),
            },
            coords={
                "time": pd.date_range("2020-01-01", periods=2, freq="YS"),
                "lat": [40.0, 41.0],
                "lon": [-112.0, -111.0],
            },
        )
        for var in ds.data_vars:
            ds[var].attrs["units"] = "kg/m**2/s"
        return inventories.Inventory(ds, pollutant="CH4", src_units="kg/m**2/s")

    @staticmethod
    def _drawn(ax):
        return np.asarray(ax.collections[0].get_array()).ravel()

    @pytest.mark.usefixtures("plotted")
    @pytest.mark.parametrize(
        "time, sector, expected",
        [
            (-1, None, 15.0),  # the last step: 5 + 10
            (0, None, 3.0),  # the first: 1 + 2
            ("2021-01-01", "agriculture", 10.0),  # a label, one sector
            ("mean", "energy", 3.0),  # (1 + 5) / 2
        ],
    )
    def test_plot_selects_the_step_and_sector(self, time, sector, expected):
        ax = self._steps().plot(time=time, sector=sector)
        np.testing.assert_allclose(self._drawn(ax), expected)

    @pytest.mark.usefixtures("plotted")
    def test_plot_draws_on_the_axes_it_is_given(self):
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})[1]
        assert self._steps().plot(ax=ax, time=0) is ax


# --- Time steps the base class knows ------------------------------------------


class TestCalendarSteps:
    """absolute_emissions of a flux per second is the flux times the step's seconds."""

    @staticmethod
    def _seconds(inv):
        absolute = inv.absolute_emissions["energy"]
        per_m2 = absolute / (inv.gridcell_area * 1e6)
        return per_m2.isel(lat=0, lon=0).values

    def test_quarters_have_the_days_of_their_months(self):
        # January to March 2019 is 31 + 28 + 31 = 90 days, April to June 30 + 31 + 30 = 91
        inv = _flux_of_one("quarterly", "QS", "kg m-2 s-1")
        assert inv.data.indexes["time"].month.tolist() == [1, 4]
        np.testing.assert_allclose(self._seconds(inv), [90 * 86400, 91 * 86400])

    def test_a_quarterly_leap_year_quarter_has_91_days(self):
        # January to March 2020: 31 + 29 + 31
        inv = _flux_of_one("quarterly", "QS", "kg m-2 s-1", start="2020-01-01")
        np.testing.assert_allclose(self._seconds(inv)[0], 91 * 86400)

    def test_biweekly_is_fourteen_days(self):
        inv = _flux_of_one("biweekly", "14D", "kg m-2 s-1")
        np.testing.assert_allclose(self._seconds(inv), [14 * 86400] * 2)

    def test_a_step_that_is_not_known_raises(self):
        with pytest.raises(ValueError, match="Time step weekly not supported"):
            _ = _flux_of_one("weekly", "7D", "kg m-2 s-1").absolute_emissions

    def test_converting_a_quarterly_flux_uses_pints_year(self):
        # No pint unit is one quarter, so nothing is per step: 1 kg m-2 s-1 is
        # 1e-3 Mg / 1e-6 km2 = 1000 Mg km-2 s-1, and a pint year is 365.25 days
        inv = _flux_of_one("quarterly", "QS", "kg m-2 s-1")
        converted = inv.convert_units("Mg/km**2/yr")
        np.testing.assert_allclose(_values(converted), 1000 * 365.25 * 86400)


def test_bounds_that_were_not_decoded_raise():
    # The bounds of a file opened with decode_times=False are numbers
    ds = xr.Dataset(
        {"time_bnds": (("time", "nv"), np.array([[0, 31], [31, 59]]))},
        coords={"time": [15, 45]},
    )
    with pytest.raises(ValueError, match="'time_bnds' was not decoded to datetimes"):
        inventories._label_period_starts(ds)


class TestMultiModelLabels:
    def test_a_model_is_selected_by_its_label(self, tmp_path):
        # Models named by text (WetCHARTs numbers them): 'b' is three times 'a'
        import pandas as pd

        values = np.ones((2, 1, 2, 2))
        values[1] = 3.0
        ds = xr.Dataset(
            {"wetlands": (("model", "time", "lat", "lon"), values)},
            coords={
                "model": ["a", "b"],
                "time": pd.date_range("2020-01-01", periods=1, freq="YS"),
                "lat": [40.05, 40.15],
                "lon": [-111.95, -111.85],
            },
        )
        ds["wetlands"].attrs["units"] = "kg/m**2/s"
        ds.to_netcdf(tmp_path / "models.nc")
        inv = inventories.MultiModelInventory(
            str(tmp_path / "models.nc"),
            pollutant="CH4",
            src_units="kg/m**2/s",
            model="b",
        )
        assert "model" not in inv.data.dims
        assert (inv.data["wetlands"] == 3.0).all()
        assert inv.multimodel_data.sizes["model"] == 2


# --- Loaders ----------------------------------------------------------------


class TestEPALoaders:
    def test_a_time_step_without_a_reader_raises(self, epa_v1_dir):
        import shutil

        folder = epa_v1_dir / "EPA" / "v1"
        shutil.copy(folder / "GEPA_Annual.nc", folder / "GEPA_Hourly.nc")
        with pytest.raises(ValueError, match="Time step hourly not supported"):
            inventories.EPAv1("Hourly", inventory_dir=epa_v1_dir)

    @pytest.mark.parametrize(
        "name, prefix, expected",
        [
            (
                "emi_ch4_1A_Combustion_Stationary",
                "emi_ch4",
                ("1A", "Combustion_Stationary"),
            ),
            ("emi_ch4_5A1_Landfills_MSW", "emi_ch4", ("5A1", "Landfills_MSW")),
            ("emi_ch4_total", "emi_ch4", (None, None)),
            ("lat", "emi_ch4", (None, None)),
        ],
    )
    def test_ipcc_code_and_short_name_from_a_variable_name(
        self, name, prefix, expected
    ):
        epa = _epa_v2(express=False)
        assert epa._extract_ipcc_code_and_short_name(name, prefix) == expected


@pytest.fixture
def gfei_v1_dir(tmp_path):
    """
    GFEI v1 files: two sectors, their relative and geometric deviations, and the total.

    The sectors have a value of their own in each file, so a mix-up shows. The files
    for "All" sectors are there too: they are not a sector.
    """
    d = tmp_path / "GFEI" / "v1"
    d.mkdir(parents=True)
    grid = {"lat": [40.05, 40.15], "lon": [-111.95, -111.85]}
    prefix = "Global_Fuel_Exploitation_Inventory"

    def write(suffix, variable, value):
        xr.Dataset(
            {variable: (("lat", "lon"), np.full((2, 2), value))},
            coords=grid,
            attrs={"year": "2016"},
        ).to_netcdf(d / f"{prefix}_{suffix}.nc")

    for sector, emis, rsd, gsd in [("Coal", 10.0, 0.5, 1.5), ("Gas", 20.0, 0.25, 1.25)]:
        write(sector, "emis_ch4", emis)
        write(f"{sector}_rsd", "rsd", rsd)
        write(f"{sector}_gsd", "gsd", gsd)
    write("Total_Fuel_Exploitation", "emis_ch4", 30.0)
    write("All_rsd", "rsd", 0.99)
    write("All_gsd", "gsd", 9.99)
    return tmp_path


class TestGFEIv1:
    def test_sectors_without_the_total_or_the_deviation_files(self, gfei_v1_dir):
        gfei = inventories.GFEIv1(inventory_dir=gfei_v1_dir)
        assert set(gfei.data.data_vars) == {"Coal", "Gas"}
        assert float(gfei.data["Coal"].max()) == 10.0
        assert float(gfei.data["Gas"].max()) == 20.0

    @pytest.mark.parametrize(
        "kind, expected",
        [
            ("relative", {"Coal": 0.5, "Gas": 0.25}),
            ("geometric", {"Coal": 1.5, "Gas": 1.25}),
        ],
    )
    def test_standard_deviations_of_each_sector(self, gfei_v1_dir, kind, expected):
        gfei = inventories.GFEIv1(inventory_dir=gfei_v1_dir)
        sd = gfei.get_standard_deviations(kind)
        # The files for all sectors together are not a sector
        assert set(sd.data_vars) == set(expected)
        for sector, value in expected.items():
            assert (sd[sector] == value).all()
            assert sd[sector].attrs["standard_name"] == f"{kind}_standard_deviation"
        assert sd.indexes["time"].year.tolist() == [2016]


class TestVulcanErrors:
    def test_alaska_is_refused(self, vulcan_dir):
        with pytest.raises(ValueError, match="Alaska region not supported"):
            inventories.Vulcan(region="AK", inventory_dir=vulcan_dir)

    def test_uncertainties_are_annual_only(self, vulcan_hourly_dir):
        v = inventories.Vulcan("hourly", inventory_dir=vulcan_hourly_dir)
        with pytest.raises(ValueError, match="only available for annual data"):
            v.get_uncertainties("lower")
