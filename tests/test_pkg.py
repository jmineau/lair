"""Package-level smoke tests for lair.

These guard the public surface set up in ``lair/__init__.py`` and the
import-time side effects documented in AGENTS.md.
"""

import pint
import pytest

import lair


def test_imports():
    """The top-level package imports without error."""
    assert lair is not None


def test_units_registry_is_pint():
    """``lair.units`` is a usable pint registry (the shared application registry)."""
    from lair import units

    assert isinstance(units, pint.ApplicationRegistry)
    # A basic quantity round-trips through the registry.
    q = 1.0 * units("km")
    assert q.to("m").magnitude == pytest.approx(1000.0)


def test_mass_flux_context_registered():
    """The custom ``mass_flux`` context is registered at import (see __init__)."""
    from lair import units

    # Entering the context by name only succeeds if it was registered.
    with units.context("mass_flux"):
        pass


def test_reexports_present():
    """The handful of symbols __init__ explicitly re-exports are available."""
    assert callable(lair.ftp_download)
    assert callable(lair.unzip)
    assert hasattr(lair, "config")


def test_verbose_flag_accessible():
    """Verbosity is a boolean toggle on the config module (no logging setup)."""
    assert isinstance(lair.config.verbose, bool)
    assert callable(lair.config.vprint)
