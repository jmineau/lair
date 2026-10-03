"""How lair.background fails to import without the NOAA CCG filter.

Kept apart from test_background.py, which skips entirely when the filter is not
installed: these tests simulate its absence, so they run either way.
"""

import importlib
import sys

import pytest


@pytest.fixture
def fresh_import(monkeypatch):
    """Import lair.background anew, with lair._ccg_filter not yet imported."""
    import lair  # noqa: F401  (the package must already be importable)

    monkeypatch.delitem(sys.modules, "lair.background", raising=False)
    monkeypatch.delitem(sys.modules, "lair._ccg_filter", raising=False)
    return lambda: importlib.import_module("lair.background")


def test_missing_filter_says_how_to_install_it(fresh_import, monkeypatch):
    # None in sys.modules makes the import raise ModuleNotFoundError for it
    monkeypatch.setitem(sys.modules, "lair._ccg_filter", None)
    with pytest.raises(ImportError, match=r"lair\.setup_ccg_filter\(\)") as info:
        fresh_import()
    assert not isinstance(info.value, ModuleNotFoundError)
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


def test_missing_dependency_of_the_filter_is_not_masked(fresh_import, monkeypatch):
    # The filter is there but needs scipy, which isn't: say so, not "run
    # setup_ccg_filter()"
    class _NeedsScipy:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "lair._ccg_filter":
                raise ModuleNotFoundError("No module named 'scipy'", name="scipy")
            return None

    monkeypatch.setattr(sys, "meta_path", [_NeedsScipy(), *sys.meta_path])
    with pytest.raises(ModuleNotFoundError) as info:
        fresh_import()
    assert info.value.name == "scipy"
