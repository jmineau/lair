"""Tests for lair._optional (optional-dependency import helper).

NOTE: this directory maps to the private module ``lair._optional``. If that
helper is ever promoted/renamed, rename this directory to match.
"""

import pytest

from lair._optional import import_optional_dependency


def test_imports_available_module():
    # os is always importable; the helper should return the module object.
    mod = import_optional_dependency("os")
    assert mod.__name__ == "os"


def test_missing_module_raises_informative_error():
    with pytest.raises(ImportError, match="Optional `lair` dependency"):
        import_optional_dependency("a_module_that_does_not_exist_xyz")
