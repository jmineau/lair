"""Tests for lair.geo.

Scaffold only (see tests/README.md). Requires the `geo` extra
(cartopy/shapely/pyproj/rasterio/rioxarray) — skipped when not installed.
"""

import pytest

# exc_type=ImportError: lair._optional re-raises missing extras as a plain
# ImportError (not ModuleNotFoundError), which importorskip ignores by default.
geo = pytest.importorskip(
    "lair.geo", reason="requires the `geo` extra", exc_type=ImportError
)


def test_importable():
    """lair.geo imports when the geo extra is available."""
    assert geo is not None


# TODO: add behavior tests for lair.geo, e.g.:
#   - wrap_lons / round_latlon on known coordinates
#   - BaseGrid construction and cell geometry
#   - points_along_line spacing
