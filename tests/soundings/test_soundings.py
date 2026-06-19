"""Tests for lair.soundings.

Scaffold only (see tests/README.md). Requires the `requests` extra
(requests/siphon) — skipped when not installed. Live Wyoming upper-air fetches
should be marked `network`.
"""

import pytest

# exc_type=ImportError covers both ModuleNotFoundError and the plain ImportError
# that lair._optional raises for missing extras.
soundings = pytest.importorskip(
    "lair.soundings", reason="requires requests/siphon", exc_type=ImportError
)


def test_importable():
    """lair.soundings imports when its optional deps are available."""
    assert soundings is not None


# TODO: add behavior tests for lair.soundings, e.g.:
#   - parsing of a saved Wyoming response fixture (offline)
#   - @pytest.mark.network for live fetches
