"""Tests for lair.clock (time/date utilities)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from lair import clock


class TestTimeRangeParsing:
    def test_year_string_expands_to_full_year(self):
        tr = clock.TimeRange("2024")
        assert tr.start == dt.datetime(2024, 1, 1)
        assert tr.stop == dt.datetime(2025, 1, 1)

    def test_year_month_string(self):
        tr = clock.TimeRange("2024-07")
        assert tr.start == dt.datetime(2024, 7, 1)
        assert tr.stop == dt.datetime(2024, 8, 1)

    def test_parse_iso_inclusive_adds_one_day(self):
        # The stop setter parses date strings inclusively (end of that day).
        assert clock.TimeRange.parse_iso("2024-01-02") == dt.datetime(2024, 1, 2)
        assert clock.TimeRange.parse_iso("2024-01-02", inclusive=True) == dt.datetime(
            2024, 1, 3
        )

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            clock.TimeRange.parse_iso("not-a-date")


class TestTimeRangeMembership:
    def test_contains_within_bounds(self):
        tr = clock.TimeRange(start=dt.datetime(2024, 1, 1), stop=dt.datetime(2024, 1, 3))
        assert dt.datetime(2024, 1, 2) in tr
        assert dt.datetime(2024, 2, 1) not in tr

    def test_open_ended_ranges(self):
        after = clock.TimeRange(start=dt.datetime(2024, 1, 1))
        assert dt.datetime(2030, 1, 1) in after
        assert dt.datetime(2000, 1, 1) not in after

    def test_empty_range_contains_everything(self):
        tr = clock.TimeRange()
        assert dt.datetime(1900, 1, 1) in tr

    def test_total_seconds_requires_both_ends(self):
        with pytest.raises(ValueError):
            clock.TimeRange(start=dt.datetime(2024, 1, 1)).total_seconds

    def test_iter_yields_start_stop(self):
        start, stop = dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)
        assert list(clock.TimeRange(start=start, stop=stop)) == [start, stop]


class TestLeapYearSeconds:
    def test_leap_year(self):
        assert clock.TimeRange("2024").total_seconds == 366 * 86400

    def test_common_year(self):
        assert clock.TimeRange("2023").total_seconds == 365 * 86400


class TestDecimalDate:
    def test_start_of_year_is_integer(self):
        assert clock.dt2decimalDate(dt.datetime(2024, 1, 1)) == pytest.approx(2024.0)

    def test_midyear_round_trip(self):
        d = dt.datetime(2024, 7, 2, 12)
        decimal = clock.dt2decimalDate(d)
        assert 2024.0 < decimal < 2025.0
        # Round-trip is accurate to within a second.
        recovered = clock.decimalDate2dt(decimal)
        assert abs((recovered - d).total_seconds()) < 1.0


class TestTimer:
    def test_context_manager_returns_timer(self):
        with clock.Timer(logger=None) as t:
            assert isinstance(t, clock.Timer)

    def test_stop_without_start_raises(self):
        timer = clock.Timer(logger=None)
        with pytest.raises(clock.Timer.TimerError):
            timer.stop()

    def test_double_start_raises(self):
        timer = clock.Timer(logger=None)
        timer.start()
        with pytest.raises(clock.Timer.TimerError):
            timer.start()

    def test_elapsed_is_nonnegative(self):
        timer = clock.Timer(logger=None)
        timer.start()
        assert timer.stop() >= 0.0


class TestTimezones:
    def test_utc_to_mst_offset(self, sample_datetimes):
        # MST is UTC-7 (no DST).
        converted = clock.UTC2MST([dt.datetime(2024, 1, 1, 18)])
        assert converted[0].hour == 11
        assert str(converted[0].tzinfo) == "MST"

    def test_localize_drops_tzinfo(self):
        converted = clock.UTC2MST([dt.datetime(2024, 1, 1, 18)], localize=True)
        assert converted[0].tzinfo is None
        assert converted[0].hour == 11


class TestTimeMatrices:
    def test_difference_matrix_shape_and_diagonal(self, sample_datetimes):
        m = clock.time_difference_matrix(sample_datetimes)
        n = len(sample_datetimes)
        assert m.shape == (n, n)
        # Distance from each time to itself is zero.
        assert all(m[i, i] == pd.Timedelta(0) for i in range(n))

    def test_difference_matrix_absolute(self, sample_datetimes):
        m = clock.time_difference_matrix(sample_datetimes, absolute=True)
        assert (m >= pd.Timedelta(0)).all()

    def test_decay_matrix_in_unit_interval(self, sample_datetimes):
        decay = clock.time_decay_matrix(sample_datetimes, decay="1h")
        assert isinstance(decay, np.ndarray)
        assert np.all((decay > 0) & (decay <= 1))
        # Self-decay (zero lag) is exactly 1.
        assert np.allclose(np.diag(decay), 1.0)


class TestIntervalHelpers:
    def test_regular_times_to_intervals_monthly(self):
        times = pd.to_datetime(["2024-01-01", "2024-02-01"])
        intervals = clock.regular_times_to_intervals(times, "monthly")
        assert isinstance(intervals, pd.IntervalIndex)
        assert intervals[0].left == pd.Timestamp("2024-01-01")
        assert intervals[0].right == pd.Timestamp("2024-02-01")

    def test_regular_times_to_intervals_invalid_step(self):
        with pytest.raises(ValueError):
            clock.regular_times_to_intervals(pd.to_datetime(["2024-01-01"]), "weekly")

    def test_periodindex_to_binedges(self):
        pi = pd.period_range("2024-01", periods=3, freq="M")
        edges = clock.periodindex_to_binedges(pi)
        # n periods -> n+1 edges.
        assert len(edges) == len(pi) + 1
        assert edges[0] == pd.Timestamp("2024-01-01")


class TestConvertTimezonesPandas:
    def test_dataframe_index_converted(self):
        df = pd.DataFrame(
            {"v": [1, 2]},
            index=pd.to_datetime(["2024-01-01 18:00", "2024-01-01 19:00"]),
        )
        out = clock.convert_timezones(df, totz="MST", fromtz="UTC", driver="pandas")
        assert list(out.index.hour) == [11, 12]  # UTC-7
        assert str(out.index.tz) == "MST"

    def test_series_of_datetimes_converted(self):
        s = pd.Series(pd.to_datetime(["2024-01-01 18:00", "2024-01-01 19:00"]))
        out = clock.convert_timezones(s, totz="MST", fromtz="UTC", driver="pandas")
        assert list(out.dt.hour) == [11, 12]

    def test_localize_drops_tz(self):
        df = pd.DataFrame(
            {"v": [1]}, index=pd.to_datetime(["2024-01-01 18:00"])
        )
        out = clock.convert_timezones(
            df, totz="MST", fromtz="UTC", localize=True, driver="pandas"
        )
        assert out.index.tz is None

    def test_invalid_driver_raises(self):
        with pytest.raises(ValueError):
            clock.convert_timezones([], totz="MST", driver="bogus")


class TestAggregation:
    def test_diurnal_collapses_to_hours(self):
        df = pd.DataFrame(
            {"v": range(48)}, index=pd.date_range("2024-01-01", periods=48, freq="h")
        )
        out = clock.diurnal(df)  # default freq must parse on pandas >= 3.0
        assert len(out) == 24  # two days collapse onto 24 unique hours

    def test_seasonal_indexes_by_season_and_year(self):
        df = pd.DataFrame(
            {"v": range(12)}, index=pd.date_range("2024-01-31", periods=12, freq="ME")
        )
        out = clock.seasonal(df)
        assert "season" in out.index.names


class TestTimerAccumulation:
    def test_named_timer_accumulates(self):
        clock.Timer(name="acc", logger=None).reset_timers()
        with clock.Timer(name="acc", logger=None):
            pass
        assert "acc" in clock.Timer.timers
        assert clock.Timer.timers["acc"] >= 0.0


def test_datetime_accessor_passthrough_without_dt():
    # A plain list has no `.dt` accessor -> returned unchanged.
    obj = [1, 2, 3]
    assert clock.datetime_accessor(obj) is obj
