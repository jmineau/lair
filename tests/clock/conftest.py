"""Fixtures local to the lair.clock tests.

This conftest lives *inside* the clock test directory on purpose: it travels
with the module if lair.clock is ever extracted into its own package. Don't
move clock-specific fixtures up to the top-level tests/conftest.py.
"""

import datetime as dt

import pytest


@pytest.fixture
def sample_datetimes() -> list[dt.datetime]:
    """A short ascending list of naive datetimes spanning a few hours."""
    base = dt.datetime(2024, 1, 1, 0, 0)
    return [base + dt.timedelta(hours=h) for h in (0, 1, 3, 6)]
