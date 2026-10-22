"""
Tests for lair.soundings.

Requires the `requests` extra (requests/siphon) — skipped when not installed.
Parsing, interpolation and get_soundings are tested offline against small
synthetic CSVs shaped like saved Wyoming responses. The download path
(download_sounding/download_soundings, incl. their retry/skip rules) runs with
siphon's WyomingUpperAir replaced by a fake; live fetches are not exercised.
"""

import datetime as dt
import importlib
import sys

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


def test_missing_requests_says_what_to_install(monkeypatch):
    # None in sys.modules makes the import raise ModuleNotFoundError for it
    monkeypatch.delitem(sys.modules, "lair.soundings", raising=False)
    monkeypatch.setitem(sys.modules, "requests", None)
    with pytest.raises(
        ImportError, match="Optional `lair` dependency 'requests'"
    ) as info:
        importlib.import_module("lair.soundings")
    assert not isinstance(info.value, ModuleNotFoundError)
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def _write_sounding(dirpath, station, time, top=3000.0, with_pw=True, elevation=1289.0):
    """Write a small CSV shaped like a saved siphon/Wyoming response."""
    heights = np.arange(elevation, top + 1, 100.0)
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
            "elevation": elevation,
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


class TestInterpolateStart:
    """The levels start at each station's own ground (#117), not at SLC's."""

    def test_slc_default_is_unchanged(self, tmp_path):
        path = _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01"))
        ds = soundings.Sounding(str(path)).interpolate()
        assert ds.height.values[0] == 1289.0
        assert ds.height.values[-1] == 4999.0
        assert ds.sizes["height"] == 372
        assert ds.attrs["notes"] == "Interpolated to 1289-5000m at 10m intervals."

    def test_low_station_keeps_its_lowest_levels(self, tmp_path):
        # A station at 300 m used to lose everything below 1289 m
        path = _write_sounding(
            tmp_path, "OAK", pd.Timestamp("2024-01-01"), elevation=300.0, top=2000.0
        )
        snd = soundings.Sounding(str(path))
        ds = snd.interpolate(interval=100)
        T = ds.temperature.isel(time=0).to_series()
        assert T.index[0] == 300.0
        raw = snd.data.set_index("height")["temperature"]
        assert T.loc[300.0] == pytest.approx(raw.loc[300.0])
        assert T.loc[300.0:2000.0].notna().all()

    def test_high_station_has_no_levels_below_ground(self, tmp_path):
        # A station at 2000 m used to get 72 empty levels from 1289 to 1999 m
        path = _write_sounding(
            tmp_path, "XYZ", pd.Timestamp("2024-01-01"), elevation=2000.0, top=4000.0
        )
        ds = soundings.Sounding(str(path)).interpolate()
        T = ds.temperature.isel(time=0).to_series()
        assert T.index[0] == 2000.0
        assert T.index[-1] == 4990.0
        assert T.loc[:4000.0].notna().all()

    def test_file_without_elevation_starts_at_first_level(self, tmp_path):
        # Soundings downloaded from Wyoming today have no elevation column
        path = _write_sounding(
            tmp_path, "OAK", pd.Timestamp("2024-01-01"), elevation=3.0, top=1000.0
        )
        pd.read_csv(path).drop(columns="elevation").to_csv(path, index=False)
        snd = soundings.Sounding(str(path))
        assert not hasattr(snd, "elevation")
        ds = snd.interpolate(interval=50)
        assert ds.height.values[0] == 3.0
        assert ds.temperature.isel(time=0).to_series().loc[3.0] == pytest.approx(0.0)

    def test_without_elevation_a_wrong_height_does_not_move_the_start(self, tmp_path):
        # One SLC file has a level at 15 m further up: the first level is the
        # ground, not the lowest height
        path = _write_sounding(
            tmp_path, "OAK", pd.Timestamp("2024-01-01"), elevation=300.0, top=1000.0
        )
        df = pd.read_csv(path).drop(columns="elevation")
        df.loc[3, "height"] = 15.0
        df.to_csv(path, index=False)
        ds = soundings.Sounding(str(path)).interpolate(interval=50)
        assert ds.height.values[0] == 300.0

    def test_explicit_start_still_wins(self, tmp_path):
        path = _write_sounding(
            tmp_path, "OAK", pd.Timestamp("2024-01-01"), elevation=300.0
        )
        ds = soundings.Sounding(str(path)).interpolate(start=0, stop=500, interval=50)
        T = ds.temperature.isel(time=0).to_series()
        assert T.index.tolist() == list(np.arange(0.0, 500.0, 50.0))
        assert T.loc[:250.0].isna().all()  # below the ground
        assert T.loc[300.0] == pytest.approx(0.0)

    def test_get_soundings_starts_at_the_station_ground(self, tmp_path):
        times = pd.date_range("2024-01-01 00:00", periods=2, freq="12h")
        for t in times:
            _write_sounding(tmp_path, "OAK", t, elevation=300.0)
        ds = soundings.get_soundings("OAK", sounding_dir=str(tmp_path))
        assert ds.height.values[0] == 300.0
        assert ds.height.values[-1] == 4990.0
        assert ds.temperature.sel(height=300.0).notnull().all()

    def test_get_soundings_shares_one_evenly_spaced_grid(self, tmp_path):
        # Two files whose elevations differ by less than the spacing: one grid
        # from the lower ground, not the union of two grids 3 m apart
        _write_sounding(
            tmp_path, "OAK", pd.Timestamp("2024-01-01 00:00"), elevation=300.0
        )
        _write_sounding(
            tmp_path, "OAK", pd.Timestamp("2024-01-01 12:00"), elevation=303.0
        )
        ds = soundings.get_soundings("OAK", sounding_dir=str(tmp_path))
        assert ds.height.values[0] == 300.0
        np.testing.assert_array_equal(np.diff(ds.height.values), 10.0)
        T = ds.temperature.to_pandas()
        assert np.isnan(T.iloc[1].loc[300.0])  # below the second station's ground
        assert T.iloc[1].loc[310.0:2900.0].notna().all()


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

    def test_misspelled_argument_raises(self, tmp_path):
        # get_soundings once took **kwargs and ignored them: month=[1] read every month
        _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-15 00:00"))
        with pytest.raises(TypeError, match="month"):
            soundings.get_soundings("SLC", sounding_dir=str(tmp_path), month=[1])

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
            end=dt.datetime(2024, 1, 2, tzinfo=dt.UTC),
            sounding_dir=str(tmp_path),
        )
        assert list(pd.DatetimeIndex(ds.time.values)) == [
            pd.Timestamp("2024-01-01 12:00"),
            pd.Timestamp("2024-01-02 00:00"),
        ]

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="No soundings found"):
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

    ``requests`` holds each (time, station); ``formatted`` the time as siphon
    writes it into the request URL, which the server reads as UTC.
    ``errors`` maps a request time to a list of exceptions raised by successive
    requests for it, before the data is returned.
    """

    def __init__(self, errors=None):
        self.requests = []
        self.formatted = []
        self.errors = errors or {}

    def request_data(self, time, site_id):
        self.requests.append((pd.Timestamp(time), site_id))
        self.formatted.append(f"{time:%Y-%m-%d %H:%M:%S}")
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

    @pytest.mark.parametrize(
        ("date", "utc"),
        [
            pytest.param(dt.datetime(2024, 1, 15, 12), "2024-01-15 12", id="naive"),
            pytest.param(
                dt.datetime(2024, 1, 15, 12, tzinfo=dt.UTC), "2024-01-15 12", id="utc"
            ),
            pytest.param(
                dt.datetime(2024, 1, 15, 5, tzinfo=dt.timezone(dt.timedelta(hours=-7))),
                "2024-01-15 12",
                id="fixed-offset-datetime",
            ),
            pytest.param(
                pd.Timestamp("2024-01-15 05:00", tz="America/Denver"),
                "2024-01-15 12",
                id="mst",
            ),
            pytest.param(
                pd.Timestamp("2024-07-15 06:00", tz="America/Denver"),
                "2024-07-15 12",
                id="mdt",
            ),
            # The local date and the UTC date differ: 17:00 MST is 00 UTC the next day
            pytest.param(
                pd.Timestamp("2024-01-15 17:00", tz="America/Denver"),
                "2024-01-16 00",
                id="utc-date-is-next-day",
            ),
        ],
    )
    def test_time_is_requested_and_named_in_utc(self, tmp_path, wyoming, date, utc):
        # Naive times are UTC; aware ones are converted before the request
        # (Wyoming reads the time as UTC) and before the file is named
        expected = pd.Timestamp(f"{utc}:00")
        path = soundings.download_sounding("SLC", date, str(tmp_path))

        assert wyoming.formatted == [f"{expected:%Y-%m-%d %H:%M:%S}"]
        assert path == str(tmp_path / f"SLC_{expected:%Y%m%d%H}.csv")
        assert soundings.Sounding(path).time == expected.to_pydatetime()

    def test_same_instant_in_two_zones_is_one_file(self, tmp_path, wyoming):
        soundings.download_sounding("SLC", dt.datetime(2024, 1, 15, 12), str(tmp_path))
        soundings.download_sounding(
            "SLC", pd.Timestamp("2024-01-15 05:00", tz="America/Denver"), str(tmp_path)
        )
        assert len(wyoming.requests) == 1
        assert sorted(p.name for p in tmp_path.iterdir()) == ["SLC_2024011512.csv"]


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
    """``get_soundings`` downloads only with ``download=True``."""

    @pytest.fixture
    def wyoming(self, monkeypatch):
        fake = _FakeWyoming()
        monkeypatch.setattr(soundings, "WyomingUpperAir", fake)
        return fake

    def test_empty_dir_downloads_then_reads(self, tmp_path, monkeypatch):
        fake = _FakeWyoming()
        monkeypatch.setattr(soundings, "WyomingUpperAir", fake)
        station_dir = tmp_path / "SLC"  # does not exist yet
        ds = soundings.get_soundings(
            "SLC",
            start="2024-01-01 00:00",
            end="2024-01-01 12:00",
            sounding_dir=str(station_dir),
            download=True,
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

    @pytest.mark.parametrize("make_dir", [False, True], ids=["missing", "empty"])
    def test_default_downloads_nothing(self, tmp_path, wyoming, make_dir):
        # The folder is missing or empty: the old default downloaded
        station_dir = tmp_path / "SLC"
        if make_dir:
            station_dir.mkdir()
        with pytest.raises(ValueError, match="No soundings found") as info:
            soundings.get_soundings(
                "SLC",
                start="2024-01-01 00:00",
                end="2024-01-01 12:00",
                sounding_dir=str(station_dir),
            )
        assert wyoming.requests == []
        assert station_dir.exists() is make_dir  # reading creates no folder
        # The error names the folder and how to download
        assert str(station_dir) in str(info.value)
        assert "download=True" in str(info.value)

    def test_default_downloads_nothing_for_a_period_the_dir_lacks(
        self, tmp_path, wyoming
    ):
        _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01 00:00"))
        with pytest.raises(ValueError, match="download=True"):
            soundings.get_soundings(
                "SLC",
                start="2024-02-01",
                end="2024-02-02",
                sounding_dir=str(tmp_path),
            )
        assert wyoming.requests == []
        assert [p.name for p in tmp_path.iterdir()] == ["SLC_2024010100.csv"]

    def test_default_reads_what_is_there_without_downloading(self, tmp_path, wyoming):
        # A gap inside the period is not filled
        _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01 00:00"))
        ds = soundings.get_soundings(
            "SLC", start="2024-01-01", end="2024-01-02", sounding_dir=str(tmp_path)
        )
        assert ds.sizes["time"] == 1
        assert wyoming.requests == []

    def test_download_fills_the_missing_soundings(self, tmp_path, wyoming):
        # The folder holds one of the three soundings in the period
        _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01 12:00"))
        ds = soundings.get_soundings(
            "SLC",
            start="2024-01-01 00:00",
            end="2024-01-02 00:00",
            sounding_dir=str(tmp_path),
            download=True,
        )
        assert [t for t, _ in wyoming.requests] == [
            pd.Timestamp("2024-01-01 00:00"),
            pd.Timestamp("2024-01-02 00:00"),
        ]
        assert ds.sizes["time"] == 3

    def test_download_respects_months(self, tmp_path, wyoming):
        ds = soundings.get_soundings(
            "SLC",
            start="2024-01-31 12:00",
            end="2024-02-01 00:00",
            sounding_dir=str(tmp_path),
            months=[2],
            download=True,
        )
        assert [t for t, _ in wyoming.requests] == [pd.Timestamp("2024-02-01 00:00")]
        assert ds.sizes["time"] == 1

    @pytest.mark.parametrize(
        "dates", [{}, {"start": "2024-01-01"}, {"end": "2024-01-02"}]
    )
    def test_download_needs_start_and_end(self, tmp_path, wyoming, dates):
        _write_sounding(tmp_path, "SLC", pd.Timestamp("2024-01-01 00:00"))
        with pytest.raises(ValueError, match="start and end"):
            soundings.get_soundings(
                "SLC", sounding_dir=str(tmp_path), download=True, **dates
            )
        assert wyoming.requests == []

    def test_nothing_from_wyoming_raises_without_the_hint(self, tmp_path, monkeypatch):
        # Wyoming has no sounding for the period: asking again would not help
        error = ValueError("No data available for 2024-01-01 00Z")
        fake = _FakeWyoming({pd.Timestamp("2024-01-01 00:00"): [error]})
        monkeypatch.setattr(soundings, "WyomingUpperAir", fake)
        with pytest.raises(ValueError, match="No soundings found") as info:
            soundings.get_soundings(
                "SLC",
                start="2024-01-01 00:00",
                end="2024-01-01 00:00",
                sounding_dir=str(tmp_path),
                download=True,
            )
        assert "download=True" not in str(info.value)


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
