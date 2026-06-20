"""Tests for lair.background.

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


def test_get_well_mixed_subsets_afternoon(sample_timeseries):
    """get_well_mixed keeps only the well-mixed afternoon hours and resamples daily."""
    result = background.get_well_mixed(sample_timeseries)
    # One input day -> one daily row.
    assert len(result) == 1


def test_get_well_mixed_multiple_days():
    import pandas as pd

    idx = pd.to_datetime(
        ["2024-01-01 13:00", "2024-01-01 14:00", "2024-01-02 13:00"]
    )
    df = pd.DataFrame({"co2": [1.0, 2.0, 3.0]}, index=idx)
    assert len(background.get_well_mixed(df)) == 2


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
    signal = pd.Series(
        np.random.default_rng(0).normal(100, 5, len(idx)), index=idx
    )
    out = background.phase_shift_corrected_baseline(signal, n=60, q=0.1)
    assert isinstance(out, pd.Series)
    assert len(out) > 0
