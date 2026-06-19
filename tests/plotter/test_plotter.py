"""Tests for lair.plotter.

Scaffold only — gives lair.plotter its own self-contained test directory (see
tests/README.md). Use a non-interactive backend for any real plotting tests.
"""

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required

import pytest  # noqa: E402

plotter = pytest.importorskip("lair.plotter")


def test_importable():
    """lair.plotter imports (guards against import-time regressions)."""
    assert plotter is not None


# TODO: add behavior tests for lair.plotter, e.g.:
#   - log formatter tick labels
#   - custom legend handlers render without error
