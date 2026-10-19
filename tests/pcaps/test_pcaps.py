"""
Tests for lair.pcaps (valley heat deficit + PCAP detection).

The PCAP event/masking helpers are pure pandas. ``valleyheatdeficit`` is checked
against an independent trapezoid integration of a synthetic hydrostatic
sounding.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lair import pcaps

# Same values as lair.constants (SI)
RD, CP, G = 287.05, 1005.0, 9.81


def _sounding(times, interval=10.0):
    """
    Synthetic dry sounding shaped like soundings.Sounding.interpolate output.

    A surface inversion (-5 C at 1290 m warming to 3 C at 1700 m) under a
    -6.5 K/km lapse rate, with pressure built hydrostatically so the
    hypsometric layer temperatures are exact.
    """
    z = np.arange(1290.0, 2500.0 + 1, interval)
    T = np.where(z < 1700, 268.15 + 8 * (z - 1290) / 410, 276.15 - 6.5e-3 * (z - 1700))
    p = np.empty_like(z)
    p[0] = 87000.0
    for i in range(1, z.size):
        p[i] = p[i - 1] * np.exp(-G * interval / (RD * 0.5 * (T[i] + T[i - 1])))
    n = len(times)
    ds = xr.Dataset(
        {
            "temperature": (("time", "height"), np.tile(T - 273.15, (n, 1))),
            "pressure": (("time", "height"), np.tile(p / 100, (n, 1))),
        },
        coords={"time": times, "height": z},
    )
    ds.attrs.update(elevation=1290.0, interpolation_interval=interval)
    return ds, z, T, p


def _reference_vhd(z, T, p, top=2200.0):
    """Independent VHD [MJ/m2]: trapezoid of cp * rho * (theta_top - theta)."""
    keep = z <= top
    z, T, p = z[keep], T[keep], p[keep]
    theta = T * (1e5 / p) ** (RD / CP)
    rho = p / (RD * T)
    f = CP * rho * (theta[-1] - theta)
    return float(np.sum((f[1:] + f[:-1]) / 2 * np.diff(z))) / 1e6


class TestValleyHeatDeficit:
    def test_matches_independent_trapezoid(self):
        times = pd.date_range("2024-01-01", periods=2, freq="12h")
        ds, z, T, p = _sounding(times)
        vhd = pcaps.valleyheatdeficit(ds)
        expected = _reference_vhd(z, T, p)
        assert vhd.name == "VHD_MJ_m2"
        assert vhd.index.name == "Time_UTC"
        assert len(vhd) == 2
        np.testing.assert_allclose(vhd.to_numpy(), expected, rtol=1e-3)

    def test_single_sounding(self):
        # One time step must not be interpolated along time (-> all NaN -> 0)
        ds, z, T, p = _sounding(pd.DatetimeIndex(["2024-01-01"]))
        vhd = pcaps.valleyheatdeficit(ds)
        assert vhd.iloc[0] == pytest.approx(_reference_vhd(z, T, p), rel=1e-3)

    def test_missing_sounding_is_nan_not_zero(self):
        times = pd.date_range("2024-01-01", periods=2, freq="12h")
        ds, z, T, p = _sounding(times)
        ds["temperature"][1] = np.nan
        ds["pressure"][1] = np.nan
        vhd = pcaps.valleyheatdeficit(ds)
        assert vhd.iloc[0] == pytest.approx(_reference_vhd(z, T, p), rel=1e-3)
        assert np.isnan(vhd.iloc[1])

    def test_all_missing_single_sounding_is_nan(self):
        ds, *_ = _sounding(pd.DatetimeIndex(["2024-01-01"]))
        ds["temperature"][:] = np.nan
        ds["pressure"][:] = np.nan
        assert np.isnan(pcaps.valleyheatdeficit(ds).iloc[0])


@pytest.fixture
def vhd():
    # 12-hourly VHD with one 3-sounding run above the threshold (idx 2-4) and
    # one 2-sounding run (idx 6-7) that is too short to be an event.
    idx = pd.date_range("2024-01-01", periods=10, freq="12h")
    return pd.Series(
        [1, 1, 6, 7, 8, 1, 6, 6, 1, 1], index=idx, name="VHD_MJ_m2", dtype=float
    )


def _twelve_hourly(values, start="2024-01-01 00:00"):
    """12-hourly VHD from start; None leaves that sounding out of the series."""
    idx = pd.date_range(start, periods=len(values), freq="12h")
    s = pd.Series([np.nan if v is None else v for v in values], index=idx)
    return s[[v is not None for v in values]]


def _events(*pairs):
    """Expected events from (first, last) sounding times."""
    end = pd.Timedelta(hours=11, minutes=59, seconds=59)
    return [(pd.Timestamp(a), pd.Timestamp(b) + end) for a, b in pairs]


def _as_pairs(events):
    return list(zip(events["start"], events["end"], strict=True))


class TestDeterminePcapEvents:
    def test_finds_runs_meeting_min_periods(self, vhd):
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=3)
        assert _as_pairs(events) == _events(("2024-01-02 00:00", "2024-01-03 00:00"))
        assert events.index.name == "event_id"

    def test_short_runs_excluded(self, vhd):
        # With min_periods=5 even the 3-sounding run is too short -> no events.
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=5)
        assert len(events) == 0

    def test_no_events_keeps_start_end_columns(self, vhd):
        # Callers index events['start'] / events['end'] even when there are none
        events = pcaps.determine_pcap_events(vhd, threshold=100.0)
        assert events.empty
        assert list(events.columns) == ["start", "end"]

    def test_empty_series(self):
        events = pcaps.determine_pcap_events(
            pd.Series([], index=pd.DatetimeIndex([]), dtype=float), threshold=4.04
        )
        assert events.empty
        assert list(events.columns) == ["start", "end"]

    def test_gap_of_two_soundings_ends_the_event(self):
        # Rows were counted, so the two runs either side of the missing
        # soundings merged into one event (#112)
        vhd = _twelve_hourly([5, 5, 5, None, None, 5, 5, 5])
        events = pcaps.determine_pcap_events(vhd, threshold=4.04)
        assert _as_pairs(events) == _events(
            ("2024-01-01 00:00", "2024-01-02 00:00"),
            ("2024-01-03 12:00", "2024-01-04 12:00"),
        )

    def test_long_gap_ends_the_event(self):
        # A season with no soundings between two runs above the threshold
        vhd = pd.concat(
            [_twelve_hourly([5, 5, 5]), _twelve_hourly([5, 5, 5], start="2024-09-01")]
        )
        events = pcaps.determine_pcap_events(vhd, threshold=4.04)
        assert _as_pairs(events) == _events(
            ("2024-01-01 00:00", "2024-01-02 00:00"),
            ("2024-09-01 00:00", "2024-09-02 00:00"),
        )

    def test_one_missing_sounding_is_bridged(self):
        # One missing sounding between two above the threshold is taken as above
        vhd = _twelve_hourly([5, 5, 5, None, 5, 5, 5])
        events = pcaps.determine_pcap_events(vhd, threshold=4.04)
        assert _as_pairs(events) == _events(("2024-01-01 00:00", "2024-01-04 00:00"))

    def test_one_nan_sounding_is_bridged(self):
        # Rows were counted, so the NaN split five soundings above the
        # threshold into two runs of two, and neither was an event (#112)
        vhd = _twelve_hourly([1, 5, 5, np.nan, 5, 5, 1])
        events = pcaps.determine_pcap_events(vhd, threshold=4.04)
        assert _as_pairs(events) == _events(("2024-01-01 12:00", "2024-01-03 12:00"))

    def test_missing_and_nan_are_the_same(self):
        with_nan = _twelve_hourly([5, 5, np.nan, np.nan, 5, 5, 5, np.nan, 5])
        missing = _twelve_hourly([5, 5, None, None, 5, 5, 5, None, 5])
        pd.testing.assert_frame_equal(
            pcaps.determine_pcap_events(with_nan, threshold=4.04),
            pcaps.determine_pcap_events(missing, threshold=4.04),
        )

    def test_bridged_sounding_counts_toward_min_periods(self):
        vhd = _twelve_hourly([1, 5, np.nan, 5, 1])
        events = pcaps.determine_pcap_events(vhd, threshold=4.04, min_periods=3)
        assert _as_pairs(events) == _events(("2024-01-01 12:00", "2024-01-02 12:00"))
        assert pcaps.determine_pcap_events(vhd, threshold=4.04, min_periods=4).empty

    def test_gap_next_to_a_sounding_below_ends_the_event(self):
        vhd = _twelve_hourly([5, 5, 5, np.nan, 1, 5, 5, 5])
        events = pcaps.determine_pcap_events(vhd, threshold=4.04)
        assert _as_pairs(events) == _events(
            ("2024-01-01 00:00", "2024-01-02 00:00"),
            ("2024-01-03 12:00", "2024-01-04 12:00"),
        )

    def test_threshold_is_exclusive(self):
        vhd = _twelve_hourly([4.04, 4.04, 4.04])
        assert pcaps.determine_pcap_events(vhd, threshold=4.04).empty

    def test_order_of_the_series_does_not_matter(self, vhd):
        pd.testing.assert_frame_equal(
            pcaps.determine_pcap_events(vhd.iloc[::-1], threshold=5.0),
            pcaps.determine_pcap_events(vhd, threshold=5.0),
        )

    def test_time_off_the_12_hour_grid_raises(self):
        vhd = _twelve_hourly([5, 5, 5])
        vhd.index = vhd.index.insert(1, pd.Timestamp("2024-01-01 06:00")).delete(-1)
        with pytest.raises(ValueError, match="12-hourly: 2024-01-01 06:00:00"):
            pcaps.determine_pcap_events(vhd, threshold=4.04)

    def test_repeated_time_raises(self):
        vhd = _twelve_hourly([5, 5, 5])
        vhd.index = vhd.index[[0, 1, 1]]
        with pytest.raises(ValueError, match="more than one value at 2024-01-01 12:00"):
            pcaps.determine_pcap_events(vhd, threshold=4.04)

    def test_tz_aware_series_keeps_its_zone(self):
        vhd = _twelve_hourly([5, 5, 5])
        vhd.index = vhd.index.tz_localize("UTC").tz_convert("America/Denver")
        events = pcaps.determine_pcap_events(vhd, threshold=4.04)
        assert events.iloc[0]["start"] == pd.Timestamp("2024-01-01", tz="UTC")
        assert str(events.iloc[0]["start"].tz) == "America/Denver"


class TestBuildPcapMask:
    def test_marks_in_event_timestamps(self, vhd):
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=3)
        mask = pcaps.build_pcap_mask(vhd.index, events)
        assert mask.dtype == bool
        # The event window covers idx 2..4: idx 5 is 12 hours after idx 4.
        assert int(mask.sum()) == 3
        assert mask.iloc[0] == False  # noqa: E712 - explicit bool check
        assert mask.iloc[2] == True  # noqa: E712

    def test_empty_events_all_false(self, vhd):
        empty = pcaps.determine_pcap_events(vhd, threshold=100.0, min_periods=3)
        mask = pcaps.build_pcap_mask(vhd.index, empty)
        assert not mask.any()


class TestTimezones:
    def test_tz_aware_index_is_compared_in_utc(self, vhd):
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=3)
        # 2024-01-02 00:00 UTC is 2024-01-01 17:00 in Denver (UTC-7)
        local = pd.DatetimeIndex(
            ["2024-01-01 17:00", "2024-01-01 16:00"], tz="America/Denver"
        )
        mask = pcaps.build_pcap_mask(local, events)
        assert mask.tolist() == [True, False]


class TestFilterPcapEvents:
    def test_drops_in_event_rows(self, vhd):
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=3)
        df = pd.DataFrame({"x": range(len(vhd))}, index=vhd.index)
        filtered = pcaps.filter_pcap_events(df, events)
        # 3 in-event rows dropped, 7 remain.
        assert len(filtered) == 7
        assert filtered.index.tolist() == vhd.index.delete([2, 3, 4]).tolist()


# TODO: valleyheatdeficit() — needs a pint-quantified sounding xr.Dataset
# (temperature/pressure/elevation/interpolation_interval); add a small fixture.
