"""Shared pytest configuration and fixtures for the lair test suite.

Keep this file *minimal*. Anything specific to a single module belongs in that
module's own ``tests/<module>/conftest.py`` so it travels with the module when
the module is extracted into its own package (see ``tests/README.md``).
"""

import os

import pandas as pd
import pytest

# Make `import lair` hermetic: never reach for the NOAA GML FTP download during
# tests. If lair/_ccg_filter.py is absent (fresh checkout / CI), importing lair
# would otherwise try to fetch it over FTP. This only takes effect when the file
# is missing; when present (the usual case on CHPC) the real module is used.
os.environ.setdefault("LAIR_SKIP_CCG_DOWNLOAD", "1")


@pytest.fixture
def sample_timeseries() -> pd.Series:
    """An hourly pandas Series with a tz-naive DatetimeIndex (one full day)."""
    index = pd.date_range("2024-01-01 00:00", "2024-01-01 23:00", freq="1h")
    return pd.Series(range(len(index)), index=index, name="value", dtype="float64")
