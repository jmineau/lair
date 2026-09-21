"""Tests for lair.soundings.

Requires the `requests` extra (requests/siphon) — skipped when not installed.
Parsing, interpolation and get_soundings are tested offline against small
synthetic CSVs shaped like saved Wyoming responses.
"""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

# exc_type=ImportError covers both ModuleNotFoundError and the plain ImportError
# that lair._optional raises for missing extras.
soundings = pytest.importorskip(
    "lair.soundings", reason="requires requests/siphon", exc_type=ImportError
)


def test_importable():
    """lair.soundings imports when its optional deps are available."""
    assert soundings is not None


def _write_sounding(dirpath, station, time, top=3000.0, with_pw=True):
    """Write a small CSV shaped like a saved siphon/Wyoming response."""
    heights = np.arange(1289.0, top + 1, 100.0)
    df = pd.DataFrame(
        {
            "pressure": np.linspace(870.0, 700.0, heights.size),
            "height": heights,
            "temperature": np.linspace(0.0, -15.0, heights.size),
            "dewpoint": np.linspace(-5.0, -20.0, heights.size),
            "direction": 180.0,
            "speed": 5.0,
            "u_wind": 0.0,
            "v_wind": 5.0,
            "station": station,
            "station_number": 72572,
            "time": time,
            "latitude": 40.77,
            "longitude": -111.95,
            "elevation": 1289.0,
        }
    )
    if with_pw:
        df["pw"] = 10.0
    path = dirpath / f"{station}_{time:%Y%m%d%H}.csv"
    df.to_csv(path, index=False)
    return path


class TestInterpolate:
    def test_levels_above_sounding_top_are_nan(self, tmp_path):
        path = _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01"), top=3000.0)
        ds = soundings.Sounding(str(path)).interpolate(
            start=1300, stop=5000, interval=100
        )
        T = ds.temperature.isel(time=0).to_series()
        assert T.loc[:2900].notna().all()
        assert T.loc[3000:].isna().all()  # no extrapolation past the top (2989 m)

    def test_missing_pw_is_nan(self, tmp_path):
        path = _write_sounding(
            tmp_path, "SLC", pd.Timestamp("2024-01-01"), with_pw=False
        )
        ds = soundings.Sounding(str(path)).interpolate()
        assert np.isnan(ds.pw.item())


class TestGetSoundings:
    def test_start_is_inclusive_and_other_files_skipped(self, tmp_path):
        times = pd.date_range("2024-01-01 00:00", periods=3, freq="12h")
        for t in times:
            _write_sounding(tmp_path, "SLC", t)
        (tmp_path / "README.txt").write_text("not a sounding")
        ds = soundings.get_soundings(
            "SLC",
            start=times[0].to_pydatetime(),
            end=times[-1].to_pydatetime(),
            sounding_dir=str(tmp_path),
        )
        assert ds.sizes["time"] == 3

    def test_months_filter_applies_to_existing_files(self, tmp_path):
        for t in ["2024-01-15 00:00", "2024-02-15 00:00", "2024-03-15 00:00"]:
            _write_sounding(tmp_path, "SLC", pd.Timestamp(t))
        ds = soundings.get_soundings("SLC", sounding_dir=str(tmp_path), months=[1, 3])
        assert sorted(pd.DatetimeIndex(ds.time.values).month) == [1, 3]

    def test_string_start_end(self, tmp_path):
        for t in pd.date_range("2024-01-01 00:00", periods=3, freq="12h"):
            _write_sounding(tmp_path, "SLC", t)
        ds = soundings.get_soundings(
            "SLC",
            start="2024-01-01 12:00",
            end="2024-01-02",
            sounding_dir=str(tmp_path),
        )
        assert ds.sizes["time"] == 2

    def test_missing_dir_without_dates_raises(self, tmp_path):
        with pytest.raises(ValueError, match="start and end"):
            soundings.get_soundings("SLC", sounding_dir=str(tmp_path / "missing"))

    def test_default_dir_is_env_root_plus_station(self, tmp_path, monkeypatch):
        (tmp_path / "SLC").mkdir()
        _write_sounding(tmp_path / "SLC", "SLC", pd.Timestamp("2024-01-01"))
        monkeypatch.setenv("LAIR_SOUNDING_DIR", str(tmp_path))
        ds = soundings.get_soundings("SLC")
        assert ds.sizes["time"] == 1

    def test_unset_env_raises(self, monkeypatch):
        monkeypatch.delenv("LAIR_SOUNDING_DIR", raising=False)
        with pytest.raises(ValueError, match="LAIR_SOUNDING_DIR"):
            soundings.get_soundings("SLC")

    def test_nothing_in_range_raises(self, tmp_path):
        _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01"))
        with pytest.raises(ValueError, match="No soundings found"):
            soundings.get_soundings(
                "SLC", start=dt.datetime(2025, 1, 1), sounding_dir=str(tmp_path)
            )


# Live Wyoming upper-air fetches (download_sounding/download_soundings) would
# need @pytest.mark.network and are not exercised here.
