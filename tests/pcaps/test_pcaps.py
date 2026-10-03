"""Tests for lair.pcaps (valley heat deficit + PCAP detection).

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
    """Synthetic dry sounding shaped like soundings.Sounding.interpolate output.

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
    # Hourly VHD with one >= 3-hr run above threshold (idx 2-4) and one short
    # 2-hr run (idx 6-7) that should NOT qualify as an event.
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    return pd.Series(
        [1, 1, 6, 7, 8, 1, 6, 6, 1, 1], index=idx, name="VHD_MJ_m2", dtype=float
    )


class TestDeterminePcapEvents:
    def test_finds_runs_meeting_min_periods(self, vhd):
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=3)
        assert len(events) == 1
        assert events.iloc[0]["start"] == pd.Timestamp("2024-01-01 02:00")
        # End extends ~12 h past the last in-event timestamp (inclusive window).
        assert events.iloc[0]["end"] == pd.Timestamp("2024-01-01 04:00") + pd.Timedelta(
            hours=11, minutes=59, seconds=59
        )

    def test_short_runs_excluded(self, vhd):
        # With min_periods=5 even the 3-hr run is too short -> no events.
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=5)
        assert len(events) == 0

    def test_no_events_keeps_start_end_columns(self, vhd):
        # Callers index events['start'] / events['end'] even when there are none
        events = pcaps.determine_pcap_events(vhd, threshold=100.0)
        assert events.empty
        assert list(events.columns) == ["start", "end"]


class TestBuildPcapMask:
    def test_marks_in_event_timestamps(self, vhd):
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=3)
        mask = pcaps.build_pcap_mask(vhd.index, events)
        assert mask.dtype == bool
        # The event window covers idx 2..9 within this 10-hour index.
        assert int(mask.sum()) == 8
        assert mask.iloc[0] == False  # noqa: E712 - explicit bool check
        assert mask.iloc[2] == True  # noqa: E712

    def test_empty_events_all_false(self, vhd):
        empty = pcaps.determine_pcap_events(vhd, threshold=100.0, min_periods=3)
        mask = pcaps.build_pcap_mask(vhd.index, empty)
        assert not mask.any()


class TestTimezones:
    def test_tz_aware_index_is_compared_in_utc(self, vhd):
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=3)
        # 2024-01-01 02:00 UTC is 2023-12-31 19:00 in Denver (UTC-7)
        local = pd.DatetimeIndex(
            ["2023-12-31 19:00", "2023-12-31 18:00"], tz="America/Denver"
        )
        mask = pcaps.build_pcap_mask(local, events)
        assert mask.tolist() == [True, False]


class TestFilterPcapEvents:
    def test_drops_in_event_rows(self, vhd):
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=3)
        df = pd.DataFrame({"x": range(len(vhd))}, index=vhd.index)
        filtered = pcaps.filter_pcap_events(df, events)
        # 8 in-event rows dropped, 2 remain.
        assert len(filtered) == 2
        assert filtered.index.tolist() == [vhd.index[0], vhd.index[1]]


# TODO: valleyheatdeficit() — needs a pint-quantified sounding xr.Dataset
# (temperature/pressure/elevation/interpolation_interval); add a small fixture.
