"""
Tests for lair.soundings.

Requires the `requests` extra (requests/siphon) — skipped when not installed.
Parsing, interpolation and get_soundings are tested offline against small
synthetic CSVs shaped like saved Wyoming responses. The download path
(download_sounding/download_soundings, incl. their retry/skip rules) runs with
siphon's WyomingUpperAir replaced by a fake; live fetches are not exercised.
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

    def test_target_on_bottom_and_top_levels_takes_raw_values(self, tmp_path):
        # Raw levels 1289, 1389, ..., 2989 m; the grid hits the bottom and top
        # levels exactly (#45: the top one came back NaN)
        path = _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01"), top=3000.0)
        snd = soundings.Sounding(str(path))
        ds = snd.interpolate(start=1289, stop=2990, interval=100)
        T = ds.temperature.isel(time=0).to_series()
        raw = snd.data.set_index("height")["temperature"]
        assert T.index.tolist() == raw.index.tolist()
        np.testing.assert_allclose(T.to_numpy(), raw.to_numpy())
        assert T.loc[2989] == pytest.approx(-15.0)
        assert T.loc[1289] == pytest.approx(0.0)

    def test_duplicate_raw_heights(self, tmp_path):
        # A repeated raw level must not duplicate or drop the target level
        path = _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01"), top=3000.0)
        df = pd.read_csv(path)
        df = pd.concat([df, df.iloc[[-1]]]).sort_values("height")
        df.to_csv(path, index=False)
        ds = soundings.Sounding(str(path)).interpolate(
            start=1289, stop=2990, interval=100
        )
        T = ds.temperature.isel(time=0).to_series()
        assert T.index.is_unique
        assert T.loc[2989] == pytest.approx(-15.0)

    def test_duplicate_raw_heights_keep_the_later_row(self, tmp_path):
        # A level repeated with different values takes the later row's values
        path = _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01"), top=3000.0)
        df = pd.read_csv(path)
        repeat = df[df["height"] == 1689.0].assign(temperature=99.0)
        df = pd.concat([df, repeat]).sort_values("height", kind="stable")
        df.to_csv(path, index=False)
        ds = soundings.Sounding(str(path)).interpolate(
            start=1289, stop=2990, interval=100
        )
        T = ds.temperature.isel(time=0).to_series()
        assert T.index.is_unique
        assert T.loc[1689] == pytest.approx(99.0)

    def test_wind_direction_interpolates_across_north(self, tmp_path):
        # 350 deg below and 10 deg above: halfway is ~0/360 deg, not 180 deg
        direction = np.array([350.0, 10.0])
        speed = np.array([10.0, 10.0])
        df = pd.DataFrame(
            {
                "pressure": [870.0, 860.0],
                "height": [1300.0, 1400.0],
                "temperature": [0.0, -1.0],
                "dewpoint": [-5.0, -6.0],
                "direction": direction,
                "speed": speed,
                "u_wind": -speed * np.sin(np.deg2rad(direction)),
                "v_wind": -speed * np.cos(np.deg2rad(direction)),
                "station": "SLC",
                "time": pd.Timestamp("2024-01-01"),
            }
        )
        path = tmp_path / "SLC_2024010100.csv"
        df.to_csv(path, index=False)
        ds = soundings.Sounding(str(path)).interpolate(
            start=1300, stop=1400, interval=50
        )
        wd = ds.direction.isel(time=0).to_series()
        ws = ds.speed.isel(time=0).to_series()
        # Angular distance from north at the midpoint
        assert min(wd.loc[1350] % 360, 360 - wd.loc[1350] % 360) == pytest.approx(
            0.0, abs=1e-6
        )
        assert wd.loc[1300] == pytest.approx(350.0)
        assert ws.loc[1300] == pytest.approx(10.0)
        # Speed is the magnitude of the interpolated vector
        assert ws.loc[1350] == pytest.approx(10.0 * np.cos(np.deg2rad(10.0)))

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

    def test_tz_aware_start_end(self, tmp_path):
        # tz-aware bounds are converted to UTC before comparing with the
        # (naive UTC) file times
        for t in pd.date_range("2024-01-01 00:00", periods=3, freq="12h"):
            _write_sounding(tmp_path, "SLC", t)
        ds = soundings.get_soundings(
            "SLC",
            start=pd.Timestamp("2024-01-01 05:00", tz="America/Denver"),  # 12Z
            end=dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc),
            sounding_dir=str(tmp_path),
        )
        assert list(pd.DatetimeIndex(ds.time.values)) == [
            pd.Timestamp("2024-01-01 12:00"),
            pd.Timestamp("2024-01-02 00:00"),
        ]

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


class TestDownloadSoundings:
    @pytest.fixture
    def requested(self, monkeypatch):
        """Record the times download_soundings asks for, without any network."""
        times = []
        monkeypatch.setattr(
            soundings,
            "download_sounding",
            lambda station, date, dst=None: times.append(pd.Timestamp(date)),
        )
        return times

    def test_times_snap_to_00z_12z(self, requested):
        soundings.download_soundings(
            "SLC", dt.datetime(2024, 1, 1, 6), dt.datetime(2024, 1, 2, 6)
        )
        assert requested == [
            pd.Timestamp("2024-01-01 12:00"),
            pd.Timestamp("2024-01-02 00:00"),
        ]

    def test_bounds_on_synoptic_times_are_inclusive(self, requested):
        soundings.download_soundings("SLC", "2024-01-01 00:00", "2024-01-01 12:00")
        assert requested == [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 12:00"),
        ]

    def test_tz_aware_bounds_are_utc(self, requested):
        # 17:00 MST is 00Z the next day
        soundings.download_soundings(
            "SLC",
            pd.Timestamp("2024-01-01 17:00", tz="America/Denver"),
            pd.Timestamp("2024-01-02 05:00", tz="America/Denver"),
        )
        assert requested == [
            pd.Timestamp("2024-01-02 00:00"),
            pd.Timestamp("2024-01-02 12:00"),
        ]

    def test_months_filter(self, requested):
        soundings.download_soundings(
            "SLC", "2024-01-31 00:00", "2024-02-01 12:00", months=[2]
        )
        assert requested == [
            pd.Timestamp("2024-02-01 00:00"),
            pd.Timestamp("2024-02-01 12:00"),
        ]


def _wyoming_frame(date, station="SLC"):
    """A frame shaped like siphon's ``WyomingUpperAir.request_data`` output."""
    return pd.DataFrame(
        {
            "pressure": [870.0, 850.0, 700.0],
            "height": [1289.0, 1480.0, 3050.0],
            "temperature": [2.0, 0.5, -12.0],
            "dewpoint": [-3.0, -4.0, -20.0],
            "direction": [180.0, 200.0, 250.0],
            "speed": [5.0, 8.0, 20.0],
            "u_wind": [0.0, 2.7, 18.8],
            "v_wind": [5.0, 7.5, 6.8],
            "station": station,
            "station_number": 72572,
            "time": pd.Timestamp(date),
            "latitude": 40.77,
            "longitude": -111.95,
            "elevation": 1289.0,
            "pw": 9.5,
        }
    )


class _FakeWyoming:
    """
    Stand-in for siphon's WyomingUpperAir: records requests, no network.

    ``errors`` maps a request time to a list of exceptions raised by successive
    requests for it, before the data is returned.
    """

    def __init__(self, errors=None):
        self.requests = []
        self.errors = errors or {}

    def request_data(self, time, site_id):
        self.requests.append((pd.Timestamp(time), site_id))
        pending = self.errors.get(pd.Timestamp(time))
        if pending:
            raise pending.pop(0)
        return _wyoming_frame(time, site_id)


class TestDownloadSounding:
    """download_sounding with siphon's Wyoming client stubbed out."""

    @pytest.fixture
    def wyoming(self, monkeypatch):
        fake = _FakeWyoming()
        monkeypatch.setattr(soundings, "WyomingUpperAir", fake)
        return fake

    def test_writes_csv_that_sounding_reads(self, tmp_path, wyoming):
        date = dt.datetime(2024, 1, 1, 12)
        path = soundings.download_sounding("SLC", date, str(tmp_path))

        assert path == str(tmp_path / "SLC_2024010112.csv")
        assert wyoming.requests == [(pd.Timestamp(date), "SLC")]
        snd = soundings.Sounding(path)
        assert snd.station == "SLC"
        assert snd.time == date
        assert snd.pw == pytest.approx(9.5)
        assert snd.data["height"].tolist() == [1289.0, 1480.0, 3050.0]
        # Metadata columns become attributes, not data columns
        assert "station_number" not in snd.data.columns

    def test_existing_file_is_not_requested_again(self, tmp_path, wyoming):
        date = dt.datetime(2024, 1, 1, 12)
        soundings.download_sounding("SLC", date, str(tmp_path))
        soundings.download_sounding("SLC", date, str(tmp_path))
        assert len(wyoming.requests) == 1

    def test_default_dst_is_env_root_plus_station(self, tmp_path, monkeypatch, wyoming):
        monkeypatch.setenv("LAIR_SOUNDING_DIR", str(tmp_path))
        path = soundings.download_sounding("SLC", dt.datetime(2024, 1, 1))
        assert path == str(tmp_path / "SLC" / "SLC_2024010100.csv")
        assert (tmp_path / "SLC" / "SLC_2024010100.csv").is_file()


class TestDownloadSoundingsErrors:
    """How download_soundings treats the errors siphon raises."""

    START, END = "2024-01-01 00:00", "2024-01-02 00:00"  # 00Z, 12Z, 00Z

    def _run(self, tmp_path, monkeypatch, errors):
        fake = _FakeWyoming(errors)
        monkeypatch.setattr(soundings, "WyomingUpperAir", fake)
        sleeps = []
        monkeypatch.setattr(soundings, "sleep", sleeps.append)
        soundings.download_soundings("SLC", self.START, self.END, str(tmp_path))
        return fake, sleeps

    @staticmethod
    def _saved(tmp_path):
        return sorted(p.name for p in tmp_path.iterdir())

    def test_no_data_is_skipped(self, tmp_path, monkeypatch, caplog):
        noon = pd.Timestamp("2024-01-01 12:00")
        errors = {noon: [ValueError("No data available for 2024-01-01 12Z")]}
        _, sleeps = self._run(tmp_path, monkeypatch, errors)
        assert self._saved(tmp_path) == ["SLC_2024010100.csv", "SLC_2024010200.csv"]
        assert sleeps == []
        assert "No data available" in caplog.text

    def test_index_error_is_skipped(self, tmp_path, monkeypatch):
        # siphon raises IndexError on some malformed/empty responses
        noon = pd.Timestamp("2024-01-01 12:00")
        self._run(tmp_path, monkeypatch, {noon: [IndexError("list index")]})
        assert self._saved(tmp_path) == ["SLC_2024010100.csv", "SLC_2024010200.csv"]

    def test_other_value_error_is_raised(self, tmp_path, monkeypatch):
        noon = pd.Timestamp("2024-01-01 12:00")
        with pytest.raises(ValueError, match="bad station"):
            self._run(tmp_path, monkeypatch, {noon: [ValueError("bad station")]})
        # The sounding before the failure was kept
        assert self._saved(tmp_path) == ["SLC_2024010100.csv"]

    def test_server_busy_is_retried(self, tmp_path, monkeypatch):
        import requests

        noon = pd.Timestamp("2024-01-01 12:00")
        busy = requests.exceptions.HTTPError("503: Please try again later.")
        fake, sleeps = self._run(tmp_path, monkeypatch, {noon: [busy, busy]})
        # Retried in place (same time, before moving on), after a pause each time
        assert [t for t, _ in fake.requests] == [
            pd.Timestamp("2024-01-01 00:00"),
            noon,
            noon,
            noon,
            pd.Timestamp("2024-01-02 00:00"),
        ]
        assert sleeps == [2, 2]
        assert len(self._saved(tmp_path)) == 3

    def test_other_http_error_is_raised(self, tmp_path, monkeypatch):
        import requests

        noon = pd.Timestamp("2024-01-01 12:00")
        error = requests.exceptions.HTTPError("404: Not Found")
        with pytest.raises(requests.exceptions.HTTPError, match="404"):
            self._run(tmp_path, monkeypatch, {noon: [error]})


class TestGetSoundingsDownloads:
    def test_empty_dir_downloads_then_reads(self, tmp_path, monkeypatch):
        fake = _FakeWyoming()
        monkeypatch.setattr(soundings, "WyomingUpperAir", fake)
        station_dir = tmp_path / "SLC"  # does not exist yet
        ds = soundings.get_soundings(
            "SLC",
            start="2024-01-01 00:00",
            end="2024-01-01 12:00",
            sounding_dir=str(station_dir),
        )
        assert len(fake.requests) == 2
        assert list(pd.DatetimeIndex(ds.time.values)) == [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-01 12:00"),
        ]
        # The downloaded profile is interpolated onto the default height grid
        T = ds.temperature.isel(time=0).to_series()
        assert T.loc[1289.0] == pytest.approx(2.0)
        assert T.loc[3049.0] == pytest.approx(-12.0 + 1 / 1570 * 12.5, abs=1e-6)
        assert np.isnan(T.loc[3059.0])  # above the top level


class TestGetSoundingsPandas:
    def test_pandas_driver_merges_raw_soundings(self, tmp_path):
        times = [pd.Timestamp("2024-01-01 00:00"), pd.Timestamp("2024-01-01 12:00")]
        for t in times:
            _write_sounding(tmp_path, "SLC", t, top=1500.0)
        df = soundings.get_soundings("SLC", sounding_dir=str(tmp_path), driver="pandas")
        assert isinstance(df, pd.DataFrame)
        # Raw levels 1289, 1389, 1489 m for each sounding, not interpolated
        assert len(df) == 6
        assert sorted(set(df["time"])) == [t.to_pydatetime() for t in times]
        assert (df["station"] == "SLC").all()
        assert (df["station_number"] == 72572).all()
        assert (df["pw"] == 10.0).all()
        assert sorted(df["height"].unique()) == [1289.0, 1389.0, 1489.0]

    def test_invalid_driver_raises(self, tmp_path):
        _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01"))
        with pytest.raises(ValueError, match="Invalid driver"):
            soundings.get_soundings("SLC", sounding_dir=str(tmp_path), driver="zarr")

    def test_unreadable_file_is_skipped(self, tmp_path, caplog):
        _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01 00:00"))
        # Right name, wrong content (no 'time' column)
        (tmp_path / "SLC_2024010112.csv").write_text("a,b\n1,2\n")
        ds = soundings.get_soundings("SLC", sounding_dir=str(tmp_path))
        assert list(pd.DatetimeIndex(ds.time.values)) == [pd.Timestamp("2024-01-01")]
        assert "Error reading file" in caplog.text
        assert "SLC_2024010112.csv" in caplog.text

    def test_end_is_inclusive_and_later_files_skipped(self, tmp_path):
        for t in pd.date_range("2024-01-01 00:00", periods=3, freq="12h"):
            _write_sounding(tmp_path, "SLC", t, top=1500.0)
        df = soundings.get_soundings(
            "SLC", end="2024-01-01 12:00", sounding_dir=str(tmp_path), driver="pandas"
        )
        assert sorted(set(df["time"])) == [
            dt.datetime(2024, 1, 1, 0),
            dt.datetime(2024, 1, 1, 12),
        ]
