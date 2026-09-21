"""Tests for lair.pcaps (valley heat deficit + PCAP detection).

The PCAP event/masking helpers are pure pandas and tested here. The headline
``valleyheatdeficit`` integrator needs a pint-quantified sounding Dataset and is
left as a TODO.
"""

import pandas as pd
import pytest

from lair import pcaps


@pytest.fixture
def vhd():
    # Hourly VHD with one >= 3-hr run above threshold (idx 2-4) and one short
    # 2-hr run (idx 6-7) that should NOT qualify as an event.
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    return pd.Series([1, 1, 6, 7, 8, 1, 6, 6, 1, 1], index=idx,
                     name="VHD_MJ_m2", dtype=float)


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
        assert mask.iloc[2] == True   # noqa: E712

    def test_empty_events_all_false(self, vhd):
        empty = pcaps.determine_pcap_events(vhd, threshold=100.0, min_periods=3)
        mask = pcaps.build_pcap_mask(vhd.index, empty)
        assert not mask.any()


class TestTimezones:
    def test_tz_aware_index_is_compared_in_utc(self, vhd):
        events = pcaps.determine_pcap_events(vhd, threshold=5.0, min_periods=3)
        # 2024-01-01 02:00 UTC is 2023-12-31 19:00 in Denver (UTC-7)
        local = pd.DatetimeIndex(["2023-12-31 19:00", "2023-12-31 18:00"],
                                 tz="America/Denver")
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
