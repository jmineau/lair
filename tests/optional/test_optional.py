"""Tests for lair._optional (optional-dependency import helper).

NOTE: this directory maps to the private module ``lair._optional``. If that
helper is ever promoted/renamed, rename this directory to match.
"""

import sys

import pytest

from lair._optional import import_optional_dependency


def test_imports_available_module():
    # os is always importable; the helper should return the module object.
    mod = import_optional_dependency("os")
    assert mod.__name__ == "os"


def test_missing_module_raises_informative_error():
    with pytest.raises(ImportError, match="Optional `lair` dependency"):
        import_optional_dependency("a_module_that_does_not_exist_xyz")


def test_missing_parent_package_raises_informative_error():
    with pytest.raises(ImportError, match="Optional `lair` dependency") as info:
        import_optional_dependency("a_module_that_does_not_exist_xyz.sub")
    # The original error is chained for debugging
    assert isinstance(info.value.__cause__, ModuleNotFoundError)


@pytest.fixture
def make_package(tmp_path, monkeypatch):
    """Write an importable package whose __init__ runs `body`."""

    def make(name, body):
        pkg = tmp_path / name
        pkg.mkdir()
        (pkg / "__init__.py").write_text(body)
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, name, raising=False)
        return name

    return make


def test_installed_package_missing_its_own_dependency_is_not_reported_as_missing(
    make_package,
):
    # The package is installed; what's missing is *its* dependency. That
    # error must surface as-is, not as "optional dependency not found".
    name = make_package("lair_fake_pkg_a", "import a_dep_that_does_not_exist_xyz\n")
    with pytest.raises(ModuleNotFoundError) as info:
        import_optional_dependency(name)
    assert info.value.name == "a_dep_that_does_not_exist_xyz"


def test_installed_package_with_broken_import_reraises(make_package):
    name = make_package("lair_fake_pkg_b", "from os import no_such_name_xyz\n")
    with pytest.raises(ImportError, match="no_such_name_xyz"):
        import_optional_dependency(name)
