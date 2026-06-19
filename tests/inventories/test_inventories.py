"""Tests for lair.inventories.

Scaffold only (see tests/README.md). Requires the `geo` extra (imports
lair.geo). Many loaders read CHPC group data — mark those `chpc`.
"""

import pytest

# exc_type=ImportError: lair._optional re-raises missing extras as a plain
# ImportError (not ModuleNotFoundError), which importorskip ignores by default.
inventories = pytest.importorskip(
    "lair.inventories", reason="requires the `geo` extra", exc_type=ImportError
)


def test_importable():
    """lair.inventories imports when the geo extra is available."""
    assert inventories is not None


# TODO: add behavior tests for lair.inventories, e.g.:
#   - unit handling / mass_flux conversions on a synthetic Dataset
#   - @pytest.mark.chpc for loaders that read GROUP_DIR inventories
