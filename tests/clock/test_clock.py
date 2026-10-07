"""Tests for lair.clock (time/date utilities)."""

import datetime as dt
from zoneinfo import ZoneInfo

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

    @pytest.mark.parametrize("bad", ["2024-1", "2024-1-5", "20241", "2024-01-15junk"])
    def test_unpadded_or_trailing_input_raises(self, bad):
        # These used to parse as hour 1 of 2024 (or silently drop the tail)
        with pytest.raises(ValueError):
            clock.TimeRange.parse_iso(bad)

    @pytest.mark.parametrize(
        "string, start, stop",
        [
            ("2024-01-15T12", (2024, 1, 15, 12), (2024, 1, 15, 13)),
            ("2024011512", (2024, 1, 15, 12), (2024, 1, 15, 13)),
            ("2024-01-15T12:30", (2024, 1, 15, 12, 30), (2024, 1, 15, 12, 31)),
            ("2024-01-15 12:30:05", (2024, 1, 15, 12, 30, 5), (2024, 1, 15, 12, 30, 6)),
            (
                "2024-01-15T12:30:05Z",
                (2024, 1, 15, 12, 30, 5),
                (2024, 1, 15, 12, 30, 6),
            ),
            (
                "2024-01-15T12:30:05.5",
                (2024, 1, 15, 12, 30, 5, 500000),
                (2024, 1, 15, 12, 30, 5, 600000),
            ),
        ],
    )
    def test_sub_day_precision(self, string, start, stop):
        assert clock.TimeRange.parse_iso(string) == dt.datetime(*start)
        assert clock.TimeRange.parse_iso(string, inclusive=True) == dt.datetime(*stop)


class TestTimeRangeMembership:
    def test_contains_within_bounds(self):
        tr = clock.TimeRange(
            start=dt.datetime(2024, 1, 1), stop=dt.datetime(2024, 1, 3)
        )
        assert dt.datetime(2024, 1, 2) in tr
        assert dt.datetime(2024, 2, 1) not in tr

    def test_stop_is_exclusive(self):
        # The stop of a string range is the start of the next period
        assert dt.datetime(2024, 12, 31, 23, 59) in clock.TimeRange("2024")
        assert dt.datetime(2025, 1, 1) not in clock.TimeRange("2024")
        assert dt.datetime(2024, 1, 16) not in clock.TimeRange("2024-01-15")
        assert dt.datetime(2024, 1, 1) not in clock.TimeRange(stop="2023")

    def test_open_ended_ranges(self):
        after = clock.TimeRange(start=dt.datetime(2024, 1, 1))
        assert dt.datetime(2030, 1, 1) in after
        assert dt.datetime(2000, 1, 1) not in after

    def test_empty_range_contains_everything(self):
        tr = clock.TimeRange()
        assert dt.datetime(1900, 1, 1) in tr

    def test_total_seconds_requires_both_ends(self):
        with pytest.raises(ValueError):
            _ = clock.TimeRange(start=dt.datetime(2024, 1, 1)).total_seconds

    def test_iter_yields_start_stop(self):
        start, stop = dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)
        assert list(clock.TimeRange(start=start, stop=stop)) == [start, stop]

    @pytest.mark.parametrize(
        "kwargs", [{"start": "2024-01-01"}, {"stop": "2024-02-01"}]
    )
    def test_time_range_with_start_or_stop_raises(self, kwargs):
        # A ValueError, not an assert (asserts vanish under python -O)
        with pytest.raises(ValueError, match="time_range"):
            clock.TimeRange("2024", **kwargs)


class TestTimeRangeInputs:
    def test_copy_of_a_time_range(self):
        original = clock.TimeRange("2024-03")
        copy = clock.TimeRange(original)
        assert (copy.start, copy.stop) == (original.start, original.stop)

    @pytest.mark.parametrize(
        "time_range",
        [["2024-01", "2024-02"], ("2024-01", "2024-02"), slice("2024-01", "2024-02")],
        ids=["list", "tuple", "slice"],
    )
    def test_pair_of_strings_includes_the_stop_period(self, time_range):
        tr = clock.TimeRange(time_range)
        assert tr.start == dt.datetime(2024, 1, 1)
        assert tr.stop == dt.datetime(2024, 3, 1)  # all of February

    def test_datetime64_bounds(self):
        tr = clock.TimeRange(
            start=np.datetime64("2024-01-01T06:00"), stop=np.datetime64("2024-01-02")
        )
        assert tr.start == dt.datetime(2024, 1, 1, 6)
        assert tr.stop == dt.datetime(2024, 1, 2)
        assert isinstance(tr.start, dt.datetime)
        assert tr.total_seconds == 18 * 3600

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"time_range": 2024}, "time_range"),
            ({"time_range": ["2024", "2025", "2026"]}, "time_range"),
            ({"start": 2024}, "start"),
            ({"stop": 2024.5}, "stop"),
        ],
    )
    def test_unrecognized_formats_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            clock.TimeRange(**kwargs)

    @pytest.mark.parametrize(
        "kwargs, text",
        [
            ({}, "Entire Observation Period"),
            ({"stop": "2023"}, "Before 2024-01-01 00:00:00"),
            ({"start": "2024"}, "After 2024-01-01 00:00:00"),
            (
                {"time_range": "2024-02"},
                "2024-02-01 00:00:00 to 2024-03-01 00:00:00",
            ),
        ],
    )
    def test_str(self, kwargs, text):
        assert str(clock.TimeRange(**kwargs)) == text


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

    @pytest.mark.parametrize(
        "d",
        [
            dt.datetime(2023, 3, 1),
            dt.datetime(2024, 12, 31),
            dt.datetime(2020, 7, 4, 6),
        ],
    )
    def test_round_trip_is_exact(self, d):
        # Midnight used to come back as 23:59:59.999998 of the previous day
        assert clock.decimalDate2dt(clock.dt2decimalDate(d)) == d

    def test_aware_timestamp_is_converted_to_utc(self):
        # 12:00 MDT is 18:00 UTC; pandas 2 (pytz) and 3 (zoneinfo) used to
        # disagree here (wall clock vs elapsed-since-local-Jan-1)
        t = pd.Timestamp("2024-06-01 12:00", tz="America/Denver")
        expected = 2024 + (152 + 18 / 24) / 366
        assert clock.dt2decimalDate(t) == pytest.approx(expected, abs=1e-12)
        assert clock.dt2decimalDate(t) == clock.dt2decimalDate(
            dt.datetime(2024, 6, 1, 18)
        )

    def test_aware_datetime_is_converted_to_utc(self):
        d = dt.datetime(2024, 6, 1, 12, tzinfo=ZoneInfo("America/Denver"))
        assert clock.dt2decimalDate(d) == clock.dt2decimalDate(
            dt.datetime(2024, 6, 1, 18)
        )

    def test_aware_year_is_the_utc_year(self):
        # 20:00 MST on Dec 31 is already 03:00 UTC on Jan 1
        d = dt.datetime(2024, 12, 31, 20, tzinfo=ZoneInfo("America/Denver"))
        assert clock.dt2decimalDate(d) == pytest.approx(2025 + 3 / 24 / 365, abs=1e-12)

    def test_fall_back_instants_are_distinct_and_ordered(self):
        # 01:30 happens twice on 2024-11-03 in Denver (fold=0 MDT, fold=1 MST)
        first = dt.datetime(2024, 11, 3, 1, 30, tzinfo=ZoneInfo("America/Denver"))
        second = first.replace(fold=1)
        hour = 3600 / clock.TimeRange("2024").total_seconds
        assert clock.dt2decimalDate(second) - clock.dt2decimalDate(first) == (
            pytest.approx(hour, rel=1e-6)
        )

    def test_aware_round_trip_gives_naive_utc(self):
        t = pd.Timestamp("2024-06-01 12:00", tz="America/Denver")
        assert clock.decimalDate2dt(clock.dt2decimalDate(t)) == dt.datetime(
            2024, 6, 1, 18
        )


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

    def test_logger_gets_the_formatted_elapsed_time(self):
        messages = []
        with clock.Timer(text="took {:.1f} s", logger=messages.append) as t:
            pass
        assert t._start_time is None  # stopped on exit
        assert len(messages) == 1
        assert messages[0].startswith("took ") and messages[0].endswith(" s")
        assert float(messages[0].split()[1]) >= 0.0

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

    def test_naive_without_fromtz_raises(self):
        # Used to silently assume the machine's local timezone
        with pytest.raises(ValueError, match="fromtz"):
            clock.convert_timezones([dt.datetime(2024, 1, 1, 18)], totz="UTC")

    def test_aware_without_fromtz_is_fine(self):
        d = dt.datetime(2024, 1, 1, 18, tzinfo=dt.UTC)
        assert clock.convert_timezones([d], totz="MST")[0].hour == 11

    def test_mixed_list_naive_elements_use_fromtz(self):
        aware = dt.datetime(2024, 1, 1, 18, tzinfo=dt.UTC)
        # Not the machine's zone, so local-time fallback can't pass by accident
        naive = dt.datetime(2024, 1, 1, 18)  # EST -> 23:00 UTC
        out = clock.convert_timezones(
            [naive, aware], totz="UTC", fromtz="America/New_York", localize=True
        )
        assert out == [dt.datetime(2024, 1, 1, 23), dt.datetime(2024, 1, 1, 18)]

    def test_ambiguous_nat_gives_nat(self):
        out = clock.MTN2UTC([dt.datetime(2024, 11, 3, 1, 30)], ambiguous="NaT")
        assert out[0] is pd.NaT

    def test_ambiguous_array_is_per_element(self):
        times = [dt.datetime(2024, 11, 3, 1, 30)] * 2
        out = clock.MTN2UTC(times, ambiguous=[True, False], localize=True)
        assert out == [dt.datetime(2024, 11, 3, 7, 30), dt.datetime(2024, 11, 3, 8, 30)]


def _mtn2utc(times, driver, **kwargs):
    """Run MTN2UTC with either driver and return naive UTC times as a list."""
    if driver == "pandas":
        times = pd.Series(pd.to_datetime(times))
    return list(clock.MTN2UTC(times, driver=driver, localize=True, **kwargs))


@pytest.mark.parametrize("driver", [None, "pandas"])
class TestDST:
    """Both drivers treat DST transitions the same way (pandas semantics)."""

    fall_back = dt.datetime(2024, 11, 3, 1, 30)  # happens twice
    spring_forward = dt.datetime(2024, 3, 10, 2, 30)  # never happens

    def test_ambiguous_raises_by_default(self, driver):
        with pytest.raises(ValueError):
            _mtn2utc([self.fall_back], driver)

    @pytest.mark.parametrize(
        "ambiguous, hour",
        [(True, 7), (False, 8)],  # True = DST (MDT, UTC-6)
    )
    def test_ambiguous_flag_picks_the_instant(self, driver, ambiguous, hour):
        out = _mtn2utc([self.fall_back], driver, ambiguous=ambiguous)
        assert out == [dt.datetime(2024, 11, 3, hour, 30)]

    def test_nonexistent_raises_by_default(self, driver):
        with pytest.raises(ValueError):
            _mtn2utc([self.spring_forward], driver)

    def test_nonexistent_shift_forward(self, driver):
        # Shifted to 03:00 MDT = 09:00 UTC
        out = _mtn2utc([self.spring_forward], driver, nonexistent="shift_forward")
        assert out == [dt.datetime(2024, 3, 10, 9)]

    def test_unaffected_times_unchanged(self, driver):
        out = _mtn2utc(
            [dt.datetime(2024, 6, 1, 12), dt.datetime(2024, 1, 1, 12)], driver
        )
        assert out == [dt.datetime(2024, 6, 1, 18), dt.datetime(2024, 1, 1, 19)]


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
        df = pd.DataFrame({"v": [1]}, index=pd.to_datetime(["2024-01-01 18:00"]))
        out = clock.convert_timezones(
            df, totz="MST", fromtz="UTC", localize=True, driver="pandas"
        )
        assert out.index.tz is None

    def test_naive_without_fromtz_raises(self):
        s = pd.Series(pd.to_datetime(["2024-01-01 18:00"]))
        with pytest.raises(ValueError, match="fromtz"):
            clock.convert_timezones(s, totz="MST", driver="pandas")

    def test_invalid_driver_raises(self):
        with pytest.raises(ValueError):
            clock.convert_timezones([], totz="MST", driver="bogus")

    def test_list_with_pandas_driver_raises(self):
        with pytest.raises(ValueError, match="DataFrame or Series"):
            clock.convert_timezones(
                [dt.datetime(2024, 1, 1)], totz="MST", fromtz="UTC", driver="pandas"
            )

    def test_unknown_fromtz_is_not_reported_as_dst(self):
        # Only DST errors are re-raised as ValueError; anything else (here an
        # unknown zone, a KeyError under both pytz and zoneinfo) passes through
        s = pd.Series(pd.to_datetime(["2024-01-01 18:00"]))
        with pytest.raises(KeyError, match="Not/AZone"):
            clock.convert_timezones(s, totz="MST", fromtz="Not/AZone", driver="pandas")


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

    def test_seasonal_labels_djf_by_january_year(self):
        # Two years of monthly values equal to the month number
        idx = pd.date_range("2023-01-31", "2024-12-31", freq="ME")
        df = pd.DataFrame({"v": idx.month.astype(float)}, index=idx)
        out = clock.seasonal(df)["v"]

        assert out.index.names == ["season", "year"]
        # DJF 2024 = Dec 2023 + Jan 2024 + Feb 2024
        assert out["DJF", 2024] == pytest.approx((12 + 1 + 2) / 3)
        # Partial edge seasons are kept: Jan-Feb 2023 and Dec 2024 only
        assert out["DJF", 2023] == pytest.approx((1 + 2) / 2)
        assert out["DJF", 2025] == pytest.approx(12)
        # Other seasons keep their calendar year
        assert out["MAM", 2024] == pytest.approx((3 + 4 + 5) / 3)
        assert out["SON", 2023] == pytest.approx((9 + 10 + 11) / 3)
        assert sorted(out.xs("JJA").index) == [2023, 2024]


class TestTimerAccumulation:
    def test_named_timer_accumulates(self):
        clock.Timer(name="acc", logger=None).reset_timers()
        with clock.Timer(name="acc", logger=None):
            pass
        assert "acc" in clock.Timer.timers
        assert clock.Timer.timers["acc"] >= 0.0

    def test_existing_timer_survives_reset(self):
        t = clock.Timer(name="survivor", logger=None)
        t.reset_timers()
        with t:  # used to raise KeyError: the name was gone from the new dict
            pass
        assert clock.Timer.timers["survivor"] >= 0.0


def test_datetime_accessor_passthrough_without_dt():
    # A plain list has no `.dt` accessor -> returned unchanged.
    obj = [1, 2, 3]
    assert clock.datetime_accessor(obj) is obj
