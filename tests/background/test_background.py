"""
Tests for lair.background.

Scaffold only (see tests/README.md). Importing lair.background requires the
NOAA GML CCG filter (lair/_ccg_filter.py, fetched on first `import lair` unless
LAIR_SKIP_CCG_DOWNLOAD is set) and the `science` extra (scipy). When either is
missing, this module is skipped.
"""

import pytest

background = pytest.importorskip(
    "lair.background",
    reason="requires the CCG filter (_ccg_filter.py) and scipy",
    exc_type=ImportError,
)


def test_importable():
    """lair.background imports when the CCG filter and scipy are available."""
    assert background is not None


class TestRollingBaseline:
    def test_shape_and_low_quantile(self):
        import numpy as np
        import pandas as pd

        idx = pd.date_range("2024-01-01", periods=72, freq="h")
        signal = pd.Series(np.arange(72, dtype=float), index=idx)
        baseline = background.rolling_baseline(signal, window="24h", q=0.1)
        assert len(baseline) == len(signal)
        # A low quantile of a rising signal stays at/below its overall mean.
        assert baseline.dropna().max() <= signal.max()


def test_phase_shift_corrected_baseline_returns_series():
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2020-01-01 00:00", "2020-01-01 01:00", freq="min")
    signal = pd.Series(np.random.default_rng(0).normal(100, 5, len(idx)), index=idx)
    out = background.phase_shift_corrected_baseline(signal, n=60, q=0.1)
    assert isinstance(out, pd.Series)
    assert len(out) > 0


class TestPhaseShiftIndex:
    """phase_shift_corrected_baseline returns the baseline at the input's times (#116)."""

    def test_slower_than_1hz_keeps_the_input_index(self):
        import numpy as np
        import pandas as pd

        idx = pd.date_range("2024-06-01 12:00", periods=60, freq="10s")
        x = pd.Series(np.linspace(2.0, 2.1, 60), index=idx)
        b = background.phase_shift_corrected_baseline(x)
        assert b.index.equals(x.index)
        assert b.notna().all()
        assert (x - b).index.equals(x.index)

    def test_values_are_the_one_second_baseline_at_those_times(self):
        import numpy as np
        import pandas as pd

        idx = pd.date_range("2024-06-01 12:00", periods=60, freq="10s")
        x = pd.Series(np.linspace(2.0, 2.1, 60), index=idx)
        hz = x.asfreq("s")
        # on 1 Hz input the index is already the grid
        expected = background.phase_shift_corrected_baseline(hz, n=61)
        out = background.phase_shift_corrected_baseline(x, n=61)
        assert np.allclose(out, expected.reindex(idx))

    def test_two_days_with_a_gap(self):
        import numpy as np
        import pandas as pd

        day = pd.date_range("2024-06-01 12:00", periods=30, freq="5s")
        idx = day.append(day + pd.Timedelta("2D"))
        x = pd.Series(np.arange(60, dtype=float), index=idx)
        b = background.phase_shift_corrected_baseline(x, n=11)
        assert b.index.equals(idx)
        assert b.notna().all()


class TestThoning:
    @pytest.fixture
    def co2(self):
        import numpy as np
        import pandas as pd

        idx = pd.date_range("2020-01-01", periods=400, freq="D")
        t = np.arange(400)
        return pd.Series(
            400 + 0.005 * t + 3 * np.sin(2 * np.pi * t / 365.25), index=idx, name="co2"
        )

    def test_nan_rows_keep_original_index(self, co2):
        co2.iloc[10] = float("nan")
        out = background.thoning(co2, debug=False)
        assert out.index.equals(co2.index)
        assert out.notna().all()

    def test_unsorted_input_matches_sorted(self, co2):
        import pandas as pd

        expected = background.thoning(co2, debug=False)
        shuffled = background.thoning(co2.iloc[::-1], debug=False)
        pd.testing.assert_series_equal(shuffled.sort_index(), expected)

    def test_filter_debug_follows_log_level(self, co2, monkeypatch, caplog):
        import logging

        seen = []

        def fake_filter(xp, yp, **kwargs):
            seen.append(kwargs["debug"])

        monkeypatch.setattr(background, "ccgFilter", fake_filter)
        background.thoning_filter(co2)
        caplog.set_level(logging.DEBUG, logger="lair.background")
        background.thoning_filter(co2)
        assert seen == [False, True]


class TestSampleInterval:
    """``sampleinterval`` (days) when the data is more often than daily (#94)."""

    @staticmethod
    def captured_kwargs(monkeypatch, data, **kwargs):
        seen = {}

        def fake_filter(xp, yp, **kw):
            seen.update(kw)

        monkeypatch.setattr(background, "ccgFilter", fake_filter)
        background.thoning_filter(data, **kwargs)
        return seen

    @pytest.mark.parametrize(("freq", "per_day"), [("h", 24), ("30min", 48)])
    def test_subdaily_default(self, freq, per_day):
        """The issue's repro: hourly and 30-minute data no longer divide by zero."""
        import numpy as np
        import pandas as pd

        idx = pd.date_range("2024-01-01", periods=per_day * 60, freq=freq)
        y = pd.Series(420 + np.sin(np.arange(len(idx)) / 24), index=idx, name="CO2")
        out = background.thoning(y)
        assert out.index.equals(idx)
        assert out.notna().all()
        # The median step, in days of 1/365 decimal year (2024 has 366 days)
        filt = background.thoning_filter(y)
        assert filt.sampleinterval == pytest.approx(365 / 366 / per_day)

    def test_daily_in_leap_year(self):
        """Daily steps within a leap year are 1/366 year, under NOAA's threshold."""
        import pandas as pd

        idx = pd.date_range("2024-01-01", "2024-12-31", freq="D")
        y = pd.Series(400.0, index=idx) + pd.Series(range(len(idx)), index=idx) / 365
        filt = background.thoning_filter(y)
        assert filt.sampleinterval == pytest.approx(365 / 366)

    def test_hourly_curve_matches_daily(self):
        """
        The hourly default follows the daily curve, and filters a diurnal cycle.

        A grid of one day on hourly data would sample midnight every day and
        put the 5 ppm diurnal cycle into the curve.
        """
        import numpy as np
        import pandas as pd

        from lair.clock import dt2decimalDate

        def signal(index):
            t = np.array([dt2decimalDate(x) for x in index])
            return 410 + 2.5 * (t - 2023) + 3 * np.sin(2 * np.pi * t)

        daily_idx = pd.date_range("2023-01-01", "2023-12-31", freq="D")
        hourly_idx = pd.date_range("2023-01-01", "2023-12-31", freq="h")
        diurnal = 5 * np.cos(2 * np.pi * hourly_idx.hour / 24)
        daily = background.thoning(pd.Series(signal(daily_idx), index=daily_idx))
        hourly = background.thoning(
            pd.Series(signal(hourly_idx) + diurnal, index=hourly_idx),
            smooth_time=daily_idx,
        )
        inner = (daily_idx >= "2023-03-01") & (daily_idx < "2023-11-01")
        assert np.abs(hourly - daily)[inner].max() < 0.01

    def test_daily_default_passes_nothing(self, monkeypatch):
        """Data with steps of a day or more keeps NOAA's own estimate."""
        import pandas as pd

        for freq in ["D", "7D", "MS"]:
            idx = pd.date_range("2021-01-01", "2023-12-31", freq=freq)
            data = pd.Series(400.0, index=idx)
            assert "sampleinterval" not in self.captured_kwargs(monkeypatch, data)

    def test_subdaily_with_gap_keeps_noaa_estimate(self, monkeypatch):
        """Hourly data with a gap of a day or more did not fail; it is unchanged."""
        import pandas as pd

        idx = pd.date_range("2021-01-01", "2021-03-31 23:00", freq="h")
        idx = idx[(idx < "2021-02-01") | (idx >= "2021-02-04")]
        data = pd.Series(400.0, index=idx)
        assert "sampleinterval" not in self.captured_kwargs(monkeypatch, data)

    @pytest.mark.parametrize("value", [1, 0.25, 1 / 24])
    def test_explicit_value_passed_through(self, monkeypatch, value):
        import pandas as pd

        idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
        data = pd.Series(400.0, index=idx)
        seen = self.captured_kwargs(monkeypatch, data, sampleinterval=value)
        assert seen["sampleinterval"] is value

    def test_one_time_raises(self):
        import pandas as pd

        data = pd.Series([400.0, 401.0], index=pd.to_datetime(["2024-01-01"] * 2))
        with pytest.raises(ValueError, match="two or more distinct times"):
            background.thoning_filter(data)


def test_thoning_filter_rejects_nan():
    import pandas as pd

    data = pd.Series(
        [400.0, float("nan"), 401.0], index=pd.date_range("2020-01-01", periods=3)
    )
    with pytest.raises(ValueError, match="NaN"):
        background.thoning_filter(data)
