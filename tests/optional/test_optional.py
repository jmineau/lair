"""
Tests for lair._optional (optional-dependency import helper).

NOTE: this directory maps to the private module ``lair._optional``. If that
helper is ever promoted/renamed, rename this directory to match.
"""

import ast
import re
import sys
import tomllib
from pathlib import Path

import pytest

import lair
from lair._optional import _EXTRA_OF, import_optional_dependency

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


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


@pytest.mark.parametrize(("name", "extra"), _EXTRA_OF.items())
def test_missing_package_names_its_extra_and_the_install_command(
    name, extra, monkeypatch
):
    monkeypatch.setitem(sys.modules, name, None)  # makes `import name` fail
    with pytest.raises(ImportError) as info:
        import_optional_dependency(name)
    message = str(info.value)
    assert message.startswith(f"Optional `lair` dependency '{name}' not found.")
    assert f"`{extra}` extra" in message
    # lair is installed from GitHub: `pip install lair` is an unrelated project
    assert f'pip install "lair[{extra}] @ git+https://github.com/jmineau/lair.git"' in (
        message
    )
    assert "uv pip install" in message


def test_missing_submodule_names_the_extra_of_its_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "cartopy", None)
    monkeypatch.delitem(sys.modules, "cartopy.crs", raising=False)
    with pytest.raises(ImportError, match="`geo` extra"):
        import_optional_dependency("cartopy.crs")


def test_package_without_an_extra_is_reported_without_one():
    with pytest.raises(ImportError) as info:
        import_optional_dependency("a_module_that_does_not_exist_xyz")
    assert str(info.value) == (
        "Optional `lair` dependency 'a_module_that_does_not_exist_xyz' not found."
    )


def _name(requirement):
    """Return the package name of a requirement such as ``s3fs>=2023.10.0``."""
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement)
    assert match, requirement
    return match.group().lower()


def _extra_of(extras):
    """
    Return each package's smallest extra, from pyproject's table of extras.

    Smallest means the fewest packages: the least to install. ``lair[...]``
    entries (``complete``) are skipped, so ``complete`` is never named.
    """
    packages = {
        extra: [n for n in map(_name, requirements) if n != "lair"]
        for extra, requirements in extras.items()
    }
    owners = {}
    for extra, names in packages.items():
        for name in names:
            owners.setdefault(name, []).append(extra)
    return {
        name: min(extras_, key=lambda e: len(packages[e]))
        for name, extras_ in owners.items()
    }


def test_smallest_extra_is_chosen_for_a_package_in_several():
    extras = {
        "big": ["a", "b>=2", "c"],
        "small": ["a"],
        "complete": ["lair[big]", "lair[small]"],
    }
    assert _extra_of(extras) == {"a": "small", "b": "big", "c": "big"}


def test_table_matches_pyproject():
    """_EXTRA_OF is exactly the packages in ``[project.optional-dependencies]``."""
    extras = tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]
    assert _extra_of(extras) == _EXTRA_OF


def test_every_call_in_lair_names_a_package_in_the_table():
    calls = []
    for path in sorted(Path(lair.__file__).parent.glob("*.py")):
        if path.name == "_ccg_filter.py":  # NOAA's code, downloaded
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "import_optional_dependency"
            ):
                arg = node.args[0]
                assert isinstance(arg, ast.Constant), f"{path.name}:{node.lineno}"
                calls.append(arg.value.split(".")[0])
    assert calls, "found no calls: the scan is broken"
    missing = set(calls) - set(_EXTRA_OF)
    assert not missing, f"add {missing} to its extra in pyproject.toml and to _EXTRA_OF"
