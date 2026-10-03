"""Tests for lair.noaa (NOAA greenhouse-gas data: CarbonTracker + GML).

No network: download() calls are checked with ftp_download stubbed out. The
pure logic is tested directly, the field sampler against a tiny synthetic
CarbonTracker-like grid, and sample()/background()/.molefractions/.data against
small synthetic files shaped like the real releases.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lair.noaa import CarbonTracker, CarbonTrackerCH4, CarbonTrackerCO2, GMLData


class TestVersionDispatch:
    @pytest.mark.parametrize(
        "version, specie",
        [
            ("CT-CH4-2025", "ch4"),
            ("CT-CH4-2023", "ch4"),
            ("ct-ch4-test", "ch4"),  # case-insensitive
            ("CT2022", "co2"),
            ("CT-NRT.v2023", "co2"),
        ],
    )
    def test_get_specie_from_version(self, version, specie):
        assert CarbonTracker.get_specie_from_version(version) == specie

    def test_from_version_returns_ch4_subclass(self):
        ct = CarbonTracker.from_version(
            "CT-CH4-2025", carbon_tracker_directory="/tmp/ct"
        )
        assert isinstance(ct, CarbonTrackerCH4)
        assert ct.specie == "ch4"

    def test_from_version_returns_co2_subclass(self):
        ct = CarbonTracker.from_version("CT2019B", carbon_tracker_directory="/tmp/ct")
        assert isinstance(ct, CarbonTrackerCO2)
        assert ct.directory.as_posix() == "/tmp/ct/co2/CT2019B"

    def test_base_class_needs_a_specie(self):
        with pytest.raises(TypeError, match="from_version"):
            CarbonTracker("CT2019B", carbon_tracker_directory="/tmp/ct")


class TestCarbonTrackerPaths:
    def test_directory_from_env(self, monkeypatch):
        monkeypatch.setenv("LAIR_CARBONTRACKER_DIR", "/env/ct")
        ct = CarbonTrackerCH4(version="CT-CH4-2025")
        assert ct.directory.as_posix() == "/env/ct/ch4/CT-CH4-2025"

    def test_directory_unset_raises(self, monkeypatch):
        monkeypatch.delenv("LAIR_CARBONTRACKER_DIR", raising=False)
        with pytest.raises(ValueError, match="LAIR_CARBONTRACKER_DIR"):
            CarbonTrackerCH4(version="CT-CH4-2025")

    def test_directory_layout(self):
        ct = CarbonTrackerCH4(
            version="CT-CH4-2025", carbon_tracker_directory="/data/ct"
        )
        assert ct.directory.as_posix() == "/data/ct/ch4/CT-CH4-2025"
        assert (
            ct.molefractions_dir.as_posix() == "/data/ct/ch4/CT-CH4-2025/molefractions"
        )

    def test_repr_and_str(self):
        ct = CarbonTrackerCH4(
            version="CT-CH4-2025", carbon_tracker_directory="/data/ct"
        )
        assert "CarbonTrackerCH4" in repr(ct)
        assert "CT-CH4-2025" in repr(ct)
        assert str(ct) == "CarbonTrackerCH4(CT-CH4-2025)"

    def test_molefraction_file_for_date(self, tmp_path):
        ct = CarbonTrackerCH4(version="CT-CH4-2025", carbon_tracker_directory=tmp_path)
        month = ct.molefractions_dir / "2020" / "01"
        month.mkdir(parents=True)
        target = month / "CT-CH4-2025.molefrac_glb3x2_2020-01-15.nc"
        target.touch()
        assert ct._molefraction_file_for_date("2020-01-15") == target
        # No file for an unrelated date.
        assert ct._molefraction_file_for_date("2020-02-01") is None


class TestPreprocessMolefractions:
    def test_builds_time_from_components(self):
        ds = xr.Dataset(
            {
                "time_components": (
                    ("time", "n"),
                    np.array([[2020, 1, 1, 0, 0, 0], [2020, 1, 2, 0, 0, 0]]),
                )
            },
            coords={"time": [0, 1]},
        )
        out = CarbonTrackerCH4._preprocess_molefractions(ds)
        assert np.issubdtype(out["time"].dtype, np.datetime64)
        assert "time_components" not in out
        assert pd.Timestamp(out["time"].values[0]) == pd.Timestamp("2020-01-01")

    def test_passthrough_when_already_datetime(self):
        ds = xr.Dataset(coords={"time": pd.to_datetime(["2025-01-01", "2025-01-02"])})
        out = CarbonTrackerCH4._preprocess_molefractions(ds)
        assert np.issubdtype(out["time"].dtype, np.datetime64)


class TestCalcMolefractionsPressure:
    def test_hybrid_sigma_pressure(self):
        mf = xr.Dataset(
            {
                "at": ("level", np.array([0.0, 100.0])),
                "bt": ("level", np.array([1.0, 0.5])),
                "surf_pressure": ((), 100000.0),
            }
        )
        out = CarbonTrackerCH4.calc_molefractions_pressure(mf)
        # P = (at + bt * surf_pressure) / 100  (Pa -> hPa)
        assert out["P"].values.tolist() == pytest.approx([1000.0, 501.0])
        assert out["P"].attrs["units"] == "hPa"


class TestSampleField:
    """_sample_field places each point in the gph layer enclosing its z_asl."""

    @staticmethod
    def _synthetic_ct() -> xr.Dataset:
        times = pd.to_datetime(["2020-01-01", "2020-01-02"])
        lat = np.array([40.0, 41.0])
        lon = np.array([-112.0, -111.0])
        level = np.array([1, 2])
        boundary = np.array([0, 1, 2])
        # gph boundaries at 0, 100, 1000 m everywhere -> layers [0,100), [100,1000)
        gph = np.broadcast_to(
            np.array([0.0, 100.0, 1000.0])[None, :, None, None], (2, 3, 2, 2)
        ).copy()
        ch4 = np.empty((2, 2, 2, 2))
        ch4[:, 0, :, :] = 1900.0  # level 1
        ch4[:, 1, :, :] = 1850.0  # level 2
        return xr.Dataset(
            {
                "gph": (("time", "boundary", "latitude", "longitude"), gph),
                "ch4": (("time", "level", "latitude", "longitude"), ch4),
            },
            coords={
                "time": times,
                "latitude": lat,
                "longitude": lon,
                "level": level,
                "boundary": boundary,
            },
        )

    def test_selects_enclosing_level(self):
        ds = self._synthetic_ct()
        points = pd.DataFrame(
            {
                "datetime": ["2020-01-01 00:00", "2020-01-02 00:00"],
                "lati": [40.0, 41.0],
                "long": [-112.0, -111.0],
                "zagl": [10.0, 200.0],  # -> level 1, level 2
                "indx": [1, 2],
            }
        )
        out = CarbonTracker._sample_field(points, ds, "ch4")
        assert out["ct_ch4_ppb"].tolist() == [1900.0, 1850.0]
        assert out["ct_level"].tolist() == [1.0, 2.0]
        assert "indx" in out.columns  # passthrough columns preserved

    def test_points_outside_grid_are_nan(self):
        ds = self._synthetic_ct()  # cells centred on 40-41 N, 112-111 W
        points = pd.DataFrame(
            {
                "datetime": ["2020-01-01", "2020-01-01"],
                "lati": [40.2, 45.0],
                "long": [-112.0, -112.0],
                "zagl": [10.0, 10.0],
            }
        )
        out = CarbonTracker._sample_field(points, ds, "ch4")
        assert out["ct_ch4_ppb"].iloc[0] == 1900.0
        assert np.isnan(out["ct_ch4_ppb"].iloc[1])
        assert np.isnan(out["ct_level"].iloc[1])

    def test_units_in_column_name(self):
        ds = self._synthetic_ct().rename({"ch4": "co2"})
        points = pd.DataFrame(
            {
                "datetime": ["2020-01-01"],
                "lati": [40.0],
                "long": [-112.0],
                "zagl": [10.0],
            }
        )
        out = CarbonTracker._sample_field(points, ds, "co2", units="ppm")
        assert "ct_co2_ppm" in out.columns


class TestCarbonTrackerCO2:
    def test_file_lookup_flat_layout_and_grid(self, tmp_path):
        ct = CarbonTrackerCO2("CT2019B", carbon_tracker_directory=tmp_path)
        d = ct.molefractions_dir
        assert d.as_posix().endswith("co2/CT2019B/molefractions/co2_total")
        d.mkdir(parents=True)
        for grid in ("glb3x2", "nam1x1"):
            (d / f"CT2019B.molefrac_{grid}_2015-06-01.nc").touch()
        assert (
            ct._molefraction_file_for_date("2015-06-01").name
            == "CT2019B.molefrac_nam1x1_2015-06-01.nc"
        )
        glb = CarbonTrackerCO2(
            "CT2019B", carbon_tracker_directory=tmp_path, grid="glb3x2"
        )
        assert "glb3x2" in glb._molefraction_file_for_date("2015-06-01").name
        assert ct._molefraction_file_for_date("2015-06-02") is None

    def test_background_is_already_ppm(self):
        class _FakeCO2(CarbonTrackerCO2):
            def sample(self, points):
                return pd.DataFrame({"ct_co2_ppm": [400.0, 402.0]})

        out = _FakeCO2("CT2019B", carbon_tracker_directory="/tmp/ct").background(
            pd.DataFrame({"x": [1]})
        )
        assert out["background_ppm"].iloc[0] == pytest.approx(401.0)


class TestDownload:
    def test_sub_dirs_none_downloads_whole_version(self, monkeypatch):
        import lair.noaa as noaa

        calls = []
        monkeypatch.setattr(
            noaa,
            "ftp_download",
            lambda host, paths, *args, **kwargs: calls.append(paths),
        )
        ct = CarbonTrackerCH4(carbon_tracker_directory="/tmp/ct")
        ct.download(sub_dirs=None)
        assert calls == [[f"/products/carbontracker/{ct.specie}/{ct.version}"]]


class TestSample:
    def test_empty_points(self):
        ct = CarbonTrackerCH4(carbon_tracker_directory="/tmp/ct")
        empty = pd.DataFrame(columns=["datetime", "lati", "long", "zagl"])
        assert ct.sample(empty).empty

    def test_drops_points_without_molefraction_files(self, tmp_path):
        # molefractions_dir exists but holds no files -> all points dropped.
        ct = CarbonTrackerCH4(carbon_tracker_directory=tmp_path)
        ct.molefractions_dir.mkdir(parents=True)
        points = pd.DataFrame(
            {
                "datetime": ["2020-01-01"],
                "lati": [40.0],
                "long": [-112.0],
                "zagl": [10.0],
            }
        )
        assert ct.sample(points).empty


class TestBackground:
    """background() averages the sampled field; output is ppm (field is ppb)."""

    def test_no_molefraction_files_gives_nan(self, tmp_path):
        ct = CarbonTrackerCH4(carbon_tracker_directory=tmp_path)
        ct.molefractions_dir.mkdir(parents=True)
        points = pd.DataFrame(
            {
                "datetime": ["2020-01-01"],
                "lati": [40.0],
                "long": [-112.0],
                "zagl": [10.0],
            }
        )
        out = ct.background(points)
        assert np.isnan(out["background_ppm"].iloc[0])
        assert out["n_endpoints"].iloc[0] == 0

    class _FakeCT(CarbonTrackerCH4):
        def sample(self, points):
            return pd.DataFrame(
                {"ct_ch4_ppb": [1900.0, 2100.0, 2000.0], "run_time": ["a", "a", "b"]}
            )

    def test_overall_summary(self):
        ct = self._FakeCT(carbon_tracker_directory="/tmp/ct")
        out = ct.background(pd.DataFrame({"x": [1]}))
        assert out["background_ppm"].iloc[0] == pytest.approx(2.0)
        assert out["sigma_ppm"].iloc[0] == pytest.approx(0.1)
        assert out["n_endpoints"].iloc[0] == 3

    def test_grouped_summary(self):
        ct = self._FakeCT(carbon_tracker_directory="/tmp/ct")
        out = ct.background(pd.DataFrame({"x": [1]}), by="run_time")
        assert out.loc["a", "background_ppm"] == pytest.approx(2.0)
        assert out.loc["a", "n_endpoints"] == 2
        assert out.loc["b", "n_endpoints"] == 1


class TestGMLData:
    def test_filename_and_extension_pandas(self):
        g = GMLData(
            "co2", "spo", gml_dir="/data/gml"
        )  # surface/flask/1/ccgg/event, pandas
        assert g.ext == "txt"
        assert g.filename == "co2_spo_surface-flask_1_ccgg_event.txt"

    def test_filename_and_extension_xarray(self):
        g = GMLData("ch4", "mlo", driver="xarray", gml_dir="/data/gml")
        assert g.ext == "nc"
        assert g.filename == "ch4_mlo_surface-flask_1_ccgg_event.nc"

    def test_gml_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("LAIR_GML_DIR", "/env/gml")
        assert GMLData("ch4", "mlo").directory.as_posix() == "/env/gml/ch4/flask"

    def test_gml_dir_unset_raises(self, monkeypatch):
        monkeypatch.delenv("LAIR_GML_DIR", raising=False)
        with pytest.raises(ValueError, match="LAIR_GML_DIR"):
            GMLData("ch4", "mlo")

    def test_directory_and_filepath(self):
        g = GMLData("ch4", "mlo", gml_dir="/data/gml")
        assert g.directory.as_posix() == "/data/gml/ch4/flask"
        assert g.filepath.as_posix() == "/data/gml/ch4/flask/" + g.filename

    def test_repr_str(self):
        g = GMLData("ch4", "mlo", gml_dir="/data/gml")
        assert "GMLData" in repr(g)
        assert str(g) == "NOAA GML Data(ch4, mlo, flask)"

    def test_data_pandas_driver(self, tmp_path):
        g = GMLData("ch4", "xxx", gml_dir=str(tmp_path))
        g.directory.mkdir(parents=True, exist_ok=True)
        g.filepath.write_text(
            "datetime value qcflag\n"
            "2020-01-01T00:00:00Z 1900.0 ...\n"
            "2020-01-02T00:00:00Z 1910.0 .X.\n"
        )
        data = g.data
        assert len(data) == 2
        assert data.index.name == "datetime"


class TestGMLMonthly:
    def test_reads_monthly_means(self, tmp_path):
        g = GMLData("ch4", "uta", frequency="month", gml_dir=str(tmp_path))
        g.directory.mkdir(parents=True, exist_ok=True)
        g.filepath.write_text(
            "# number_of_header_lines: 3\n"
            "# comment line\n"
            "# data_fields: site year month value\n"
            "UTA 1993  5  1788.16\n"
            "UTA 1993  6  1781.70\n"
        )
        data = g.data
        assert list(data.columns) == ["site", "year", "month", "value"]
        assert data.index.tolist() == [
            pd.Timestamp("1993-05-01"),
            pd.Timestamp("1993-06-01"),
        ]
        assert data["value"].iloc[1] == pytest.approx(1781.70)


class TestApplyQAQC:
    @pytest.fixture
    def data(self):
        return pd.DataFrame(
            {"value": [1, 2, 3, 4], "qcflag": ["...", ".X.", "...", "A.."]}
        )

    def test_keeps_only_good_by_default(self, data):
        assert GMLData.apply_qaqc(data)["value"].tolist() == [1, 3]

    def test_allows_extra_flags(self, data):
        assert GMLData.apply_qaqc(data, flags="A..")["value"].tolist() == [1, 3, 4]

    def test_invalid_driver_raises(self, data):
        with pytest.raises(ValueError, match="Invalid driver"):
            GMLData.apply_qaqc(data, driver="polars")

    def test_xarray_driver(self):
        ds = xr.Dataset(
            {
                "value": ("obs", [1.0, 2.0, 3.0]),
                "qcflag": ("obs", ["...", ".X.", "..."]),
            }
        )
        out = GMLData.apply_qaqc(ds, driver="xarray")
        assert out["value"].values.tolist() == [1.0, 3.0]

    def test_xarray_driver_netcdf_char_flags(self, tmp_path):
        # GML/ObsPack netCDF files store qcflag as a char array (obs, nchar) with
        # no _Encoding attribute, so xarray decodes it to bytes (b'...').
        netCDF4 = pytest.importorskip("netCDF4")
        g = GMLData("ch4", "xxx", driver="xarray", gml_dir=str(tmp_path))
        g.directory.mkdir(parents=True, exist_ok=True)
        with netCDF4.Dataset(g.filepath, "w") as nc:
            nc.createDimension("obs", 3)
            nc.createDimension("string_of_3chars", 3)
            time = nc.createVariable("time", "f8", ("obs",))
            time.units = "seconds since 1970-01-01"
            time[:] = [0.0, 60.0, 120.0]
            nc.createVariable("value", "f8", ("obs",))[:] = [1.0, 2.0, 3.0]
            chars = np.array([list(f) for f in ["...", ".X.", "..."]], dtype="S1")
            nc.createVariable("qcflag", "S1", ("obs", "string_of_3chars"))[:] = chars
        assert g.data["qcflag"].dtype.kind == "S"
        out = GMLData.apply_qaqc(g.data, driver="xarray")
        assert out["value"].values.tolist() == [1.0, 3.0]
        out = GMLData.apply_qaqc(g.data, flags=".X.", driver="xarray")
        assert out["value"].values.tolist() == [1.0, 2.0, 3.0]


# --- reading molefraction files end to end -----------------------------------
#
# Tiny daily files shaped like the real releases: CT-CH4 (older versions) has a
# non-datetime ``time`` plus ``time_components``; CT2019B CO2 decodes time
# natively and carries ``time_components``/``decimal_date`` extras, one file per
# grid in a flat ``co2_total/`` directory.

_LEVELS = np.array([1, 2])
_GPH_BOUNDS = np.array([0.0, 100.0, 1000.0])  # layers [0, 100), [100, 1000) m


def _molefraction_day(day, values, variable="ch4", lat=(40.0, 41.0), ch4_time=True):
    """One day (00Z and 12Z) of a field equal to ``values[level]`` everywhere."""
    times = pd.to_datetime([f"{day} 00:00", f"{day} 12:00"])
    lat = np.asarray(lat)
    lon = np.array([-112.0, -111.0])
    shape = (2, 2, lat.size, lon.size)
    field = np.empty(shape)
    for i, value in enumerate(values):
        field[:, i] = value
    gph = np.broadcast_to(
        _GPH_BOUNDS[None, :, None, None], (2, 3, lat.size, lon.size)
    ).copy()
    components = np.array([[t.year, t.month, t.day, t.hour, 0, 0] for t in times])
    ds = xr.Dataset(
        {
            "gph": (("time", "boundary", "latitude", "longitude"), gph),
            variable: (("time", "level", "latitude", "longitude"), field),
            "time_components": (("time", "ncomp"), components),
        },
        coords={
            "latitude": lat,
            "longitude": lon,
            "level": _LEVELS,
            "boundary": np.arange(3),
        },
    )
    if ch4_time:
        # Older CT-CH4: time is a plain number, the real time is in components
        ds = ds.assign_coords(time=[0.0, 0.5])
    else:
        ds = ds.assign_coords(time=times)
        ds["decimal_date"] = ("time", [2015.4, 2015.41])
    return ds


@pytest.fixture
def ct_ch4(tmp_path):
    """CT-CH4 with files for 2020-01-01 and 2020-01-02 (none for 01-05)."""
    ct = CarbonTrackerCH4(
        version="CT-CH4-2023",
        carbon_tracker_directory=tmp_path,
        cache=False,
        parallel_parse=False,
    )
    month = ct.molefractions_dir / "2020" / "01"
    month.mkdir(parents=True)
    for day, values in (
        ("2020-01-01", (1900.0, 1850.0)),
        ("2020-01-02", (1950.0, 1800.0)),
    ):
        _molefraction_day(day, values).to_netcdf(
            month / f"CT-CH4-2023.molefrac_glb3x2_{day}.nc"
        )
    return ct


@pytest.fixture
def ct_co2(tmp_path):
    """CT2019B with nam1x1 and glb3x2 files for 2015-06-01."""
    ct = CarbonTrackerCO2(
        "CT2019B", carbon_tracker_directory=tmp_path, cache=False, parallel_parse=False
    )
    ct.molefractions_dir.mkdir(parents=True)
    day = "2015-06-01"
    _molefraction_day(day, (400.0, 395.0), "co2", ch4_time=False).to_netcdf(
        ct.molefractions_dir / f"CT2019B.molefrac_nam1x1_{day}.nc"
    )
    # The global grid holds different values on different latitudes
    _molefraction_day(
        day, (410.0, 405.0), "co2", lat=(-30.0, 30.0), ch4_time=False
    ).to_netcdf(ct.molefractions_dir / f"CT2019B.molefrac_glb3x2_{day}.nc")
    return ct


_POINTS = pd.DataFrame(
    {
        "datetime": ["2020-01-01 11:00", "2020-01-02 01:00", "2020-01-05 00:00"],
        "lati": [40.1, 40.9, 40.0],
        "long": [-112.0, -111.1, -112.0],
        "zagl": [10.0, 500.0, 10.0],
        "receptor": ["a", "a", "b"],
    }
)


class TestSampleFromFiles:
    def test_ch4_values_times_and_levels(self, ct_ch4):
        out = ct_ch4.sample(_POINTS)
        # 2020-01-05 has no file -> dropped
        assert len(out) == 2
        assert out["receptor"].tolist() == ["a", "a"]
        assert out["ct_ch4_ppb"].tolist() == [1900.0, 1800.0]
        assert out["ct_level"].tolist() == [1.0, 2.0]
        # Nearest model time/cell, with time rebuilt from time_components
        assert out["ct_time"].tolist() == [
            pd.Timestamp("2020-01-01 12:00"),
            pd.Timestamp("2020-01-02 00:00"),
        ]
        assert out["ct_latitude"].tolist() == [40.0, 41.0]
        assert out["ct_longitude"].tolist() == [-112.0, -111.0]

    def test_background_from_files_in_ppm(self, ct_ch4):
        out = ct_ch4.background(_POINTS, by="receptor")
        # receptor a: 1900 and 1800 ppb -> 1.85 ppm; b has no file
        assert out.loc["a", "background_ppm"] == pytest.approx(1.85)
        assert out.loc["a", "n_endpoints"] == 2
        assert "b" not in out.index

    def test_co2_reads_only_its_grid(self, ct_co2):
        points = pd.DataFrame(
            {
                "datetime": ["2015-06-01 00:00", "2015-06-01 12:00"],
                "lati": [40.0, 41.0],
                "long": [-112.0, -111.0],
                "zagl": [10.0, 200.0],
            }
        )
        out = ct_co2.sample(points)
        assert out["ct_co2_ppm"].tolist() == [400.0, 395.0]
        assert ct_co2.background(points)["background_ppm"].iloc[0] == pytest.approx(
            397.5
        )


class TestMolefractionsProperty:
    def test_ch4_opens_every_file_with_decoded_time(self, ct_ch4):
        ds = ct_ch4.molefractions
        assert pd.DatetimeIndex(ds.time.values).tolist() == list(
            pd.to_datetime(
                [
                    "2020-01-01 00:00",
                    "2020-01-01 12:00",
                    "2020-01-02 00:00",
                    "2020-01-02 12:00",
                ]
            )
        )
        assert "time_components" not in ds
        assert ds["ch4"].sel(level=1).isel(latitude=0, longitude=0).values.tolist() == [
            1900.0,
            1900.0,
            1950.0,
            1950.0,
        ]

    def test_ch4_cache_reuses_the_opened_dataset(self, ct_ch4, tmp_path, monkeypatch):
        import lair.config
        import lair.noaa as noaa

        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(lair.config, "CACHE_DIR", str(cache_dir))
        ct_ch4.cache = True
        first = ct_ch4.molefractions
        cache_file = (
            cache_dir / "carbontracker" / "ch4" / "CT-CH4-2023" / "molefractions.pkl"
        )
        assert cache_file.is_file()

        # A new object with the same files must not reopen them
        def fail(*args, **kwargs):
            raise AssertionError("open_mfdataset called despite the cache")

        monkeypatch.setattr(noaa.xr, "open_mfdataset", fail)
        again = CarbonTrackerCH4(
            version="CT-CH4-2023",
            carbon_tracker_directory=tmp_path,
            parallel_parse=False,
        )
        xr.testing.assert_identical(again.molefractions, first)

    def test_co2_opens_only_its_grid(self, ct_co2):
        ds = ct_co2.molefractions
        assert ds.latitude.values.tolist() == [40.0, 41.0]  # nam1x1, not glb3x2
        assert "time_components" not in ds and "decimal_date" not in ds
        assert float(ds["co2"].sel(level=2).max()) == 395.0

    def test_co2_cache_file_names_the_grid(self, ct_co2, tmp_path, monkeypatch):
        import lair.config

        monkeypatch.setattr(lair.config, "CACHE_DIR", str(tmp_path / "cache"))
        ct_co2.cache = True
        ct_co2.molefractions
        assert (
            tmp_path / "cache/carbontracker/co2/CT2019B/molefractions_nam1x1.pkl"
        ).is_file()

    def test_co2_without_files_raises(self, tmp_path):
        ct = CarbonTrackerCO2("CT2019B", carbon_tracker_directory=tmp_path)
        with pytest.raises(FileNotFoundError, match="nam1x1"):
            ct.molefractions


class TestDownloadRequests:
    @pytest.fixture
    def calls(self, monkeypatch):
        import lair.noaa as noaa

        calls = []
        monkeypatch.setattr(
            noaa,
            "ftp_download",
            lambda *args, **kwargs: calls.append((args, kwargs)),
        )
        return calls

    def test_default_sub_dirs_and_pattern(self, calls, tmp_path):
        ct = CarbonTrackerCO2("CT2019B", carbon_tracker_directory=tmp_path)
        ct.download(pattern="*2015-06*")
        version = "/products/carbontracker/co2/CT2019B"
        assert calls == [
            (
                (
                    "ftp.gml.noaa.gov",
                    [f"{version}/fluxes", f"{version}/molefractions"],
                    str(tmp_path / "co2" / "CT2019B"),
                ),
                {"prefix": version, "pattern": "*2015-06*"},
            )
        ]

    def test_gml_download_path(self, calls, tmp_path):
        g = GMLData("ch4", "uta", sample_type="pfp", gml_dir=str(tmp_path))
        assert g.download() == str(g.filepath)
        assert calls == [
            (
                (
                    "ftp.gml.noaa.gov",
                    "/data/trace_gases/ch4/pfp/surface/txt/"
                    "ch4_uta_surface-pfp_1_ccgg_event.txt",
                    str(tmp_path / "ch4" / "pfp"),
                ),
                {},
            )
        ]


class TestGMLDataReading:
    def test_event_file_sorted_naive_utc(self, tmp_path):
        g = GMLData("ch4", "xxx", gml_dir=str(tmp_path))
        g.directory.mkdir(parents=True)
        g.filepath.write_text(
            "# header comment\n"
            "datetime value qcflag\n"
            "2020-01-02T07:00:00Z 1910.0 ...\n"
            "2020-01-01T00:00:00Z 1900.0 .X.\n"
        )
        data = g.data
        assert data.index.tz is None
        assert data.index.tolist() == [
            pd.Timestamp("2020-01-01 00:00"),
            pd.Timestamp("2020-01-02 07:00"),
        ]
        assert data["value"].tolist() == [1900.0, 1910.0]

    def test_monthly_without_data_fields_raises(self, tmp_path):
        g = GMLData("ch4", "uta", frequency="month", gml_dir=str(tmp_path))
        g.directory.mkdir(parents=True)
        g.filepath.write_text("# no field list here\nUTA 1993 5 1788.16\n")
        with pytest.raises(ValueError, match="data_fields"):
            g.data
