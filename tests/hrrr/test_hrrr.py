"""Tests for lair.hrrr.

Scaffold only (see tests/README.md). Requires the `requests`/`formats`/`geo`
extras (boto3/numcodecs/cartopy) — skipped when not installed. Live HRRR access
should be marked `network`.
"""

import pytest

# exc_type=ImportError: lair._optional re-raises missing extras as a plain
# ImportError (not ModuleNotFoundError), which importorskip ignores by default.
hrrr = pytest.importorskip(
    "lair.hrrr", reason="requires boto3/numcodecs/cartopy", exc_type=ImportError
)


def test_importable():
    """lair.hrrr imports when its optional deps are available."""
    assert hrrr is not None


# TODO: add behavior tests for lair.hrrr, e.g.:
#   - filename/key construction for a given time (pure, offline)
#   - point sampling against a tiny cached grid
#   - @pytest.mark.network for live zarr/MesoWest fetches
