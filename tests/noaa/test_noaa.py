"""Tests for lair.noaa (NOAA greenhouse-gas data: CarbonTracker + GML).

Network/disk paths (download, .data, .molefractions, full sample() over real
files) are not exercised here. The pure logic is tested directly, and the field
sampler is checked against a tiny synthetic CarbonTracker-like grid.

NOTE: lair.noaa is partially WIP (the CarbonTrackerCO2 branch is a stub) and is
excluded from the ruff/pyrefly gate; these tests cover the implemented CH4/GML
surface.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lair.noaa import CarbonTracker, CarbonTrackerCH4, GMLData


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
        ct = CarbonTracker.from_version("CT-CH4-2025", carbon_tracker_directory="/tmp/ct")
        assert isinstance(ct, CarbonTrackerCH4)
        assert ct.specie == "ch4"

    def test_from_version_co2_not_implemented(self):
        with pytest.raises(ValueError, match="not yet implemented"):
            CarbonTracker.from_version("CT2022")


class TestCarbonTrackerPaths:
    def test_directory_layout(self):
        ct = CarbonTrackerCH4(version="CT-CH4-2025", carbon_tracker_directory="/data/ct")
        assert ct.directory.as_posix() == "/data/ct/ch4/CT-CH4-2025"
        assert ct.molefractions_dir.as_posix() == "/data/ct/ch4/CT-CH4-2025/molefractions"

    def test_repr_and_str(self):
        ct = CarbonTrackerCH4(version="CT-CH4-2025", carbon_tracker_directory="/data/ct")
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
            {"time_components": (("time", "n"), np.array([[2020, 1, 1, 0, 0, 0],
                                                          [2020, 1, 2, 0, 0, 0]]))},
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
            coords={"time": times, "latitude": lat, "longitude": lon,
                    "level": level, "boundary": boundary},
        )

    def test_selects_enclosing_level(self):
        ds = self._synthetic_ct()
        points = pd.DataFrame(
            {
                "time": ["2020-01-01 00:00", "2020-01-02 00:00"],
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


class TestDownload:
    def test_sub_dirs_none_downloads_whole_version(self, monkeypatch):
        import lair.noaa as noaa

        calls = []
        monkeypatch.setattr(noaa, "ftp_download",
                            lambda host, paths, *args, **kwargs: calls.append(paths))
        ct = CarbonTrackerCH4(carbon_tracker_directory="/tmp/ct")
        ct.download(sub_dirs=None)
        assert calls == [[f"/products/carbontracker/{ct.specie}/{ct.version}"]]


class TestSample:
    def test_empty_points(self):
        ct = CarbonTrackerCH4(carbon_tracker_directory="/tmp/ct")
        empty = pd.DataFrame(columns=["time", "lati", "long", "zagl"])
        assert ct.sample(empty).empty

    def test_drops_points_without_molefraction_files(self, tmp_path):
        # molefractions_dir exists but holds no files -> all points dropped.
        ct = CarbonTrackerCH4(carbon_tracker_directory=tmp_path)
        ct.molefractions_dir.mkdir(parents=True)
        points = pd.DataFrame(
            {"time": ["2020-01-01"], "lati": [40.0], "long": [-112.0], "zagl": [10.0]}
        )
        assert ct.sample(points).empty


class TestBackground:
    """background() averages the sampled field; output is ppm (field is ppb)."""

    def test_no_molefraction_files_gives_nan(self, tmp_path):
        ct = CarbonTrackerCH4(carbon_tracker_directory=tmp_path)
        ct.molefractions_dir.mkdir(parents=True)
        points = pd.DataFrame(
            {"time": ["2020-01-01"], "lati": [40.0], "long": [-112.0], "zagl": [10.0]}
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
        g = GMLData("co2", "spo")  # defaults: surface/flask/1/ccgg/event, pandas
        assert g.ext == "txt"
        assert g.filename == "co2_spo_surface-flask_1_ccgg_event.txt"

    def test_filename_and_extension_xarray(self):
        g = GMLData("ch4", "mlo", driver="xarray")
        assert g.ext == "nc"
        assert g.filename == "ch4_mlo_surface-flask_1_ccgg_event.nc"

    def test_directory_and_filepath(self):
        g = GMLData("ch4", "mlo", gml_dir="/data/gml")
        assert g.directory.as_posix() == "/data/gml/ch4/flask"
        assert g.filepath.as_posix() == "/data/gml/ch4/flask/" + g.filename

    def test_repr_str(self):
        g = GMLData("ch4", "mlo")
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
