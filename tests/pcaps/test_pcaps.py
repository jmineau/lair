"""Tests for lair.pcaps.

Scaffold only — gives lair.pcaps its own self-contained test directory (see
tests/README.md). Replace the import smoke test with real behavior tests.
"""

import pytest

pcaps = pytest.importorskip("lair.pcaps")


def test_importable():
    """lair.pcaps imports (guards against import-time regressions)."""
    assert pcaps is not None


# TODO: add behavior tests for lair.pcaps, e.g.:
#   - valley heat deficit (VHD) on a synthetic sounding
#   - PCAP detection thresholds
