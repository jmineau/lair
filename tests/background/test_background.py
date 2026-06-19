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


# TODO: add behavior tests for lair.background, e.g.:
#   - rolling_baseline / phase_shift_corrected_baseline on a synthetic signal
#   - thoning_filter / thoning round-trip (CCG filter); mark `network` if it
#     needs the live filter download
