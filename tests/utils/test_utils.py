"""Tests for lair.utils (tiny helpers: DotDict, updating_print)."""

import copy

import pytest

from lair.utils import DotDict, updating_print


class TestDotDict:
    def test_attribute_access(self):
        d = DotDict({"a": 1, "b": 2})
        assert d.a == 1
        assert d.b == 2

    def test_nested_dict_wrapped(self):
        d = DotDict({"outer": {"inner": 42}})
        # Nested plain dicts are returned as DotDicts so dotted access chains.
        assert isinstance(d.outer, DotDict)
        assert d.outer.inner == 42

    def test_attribute_assignment(self):
        d = DotDict()
        d.x = 10
        assert d["x"] == 10
        assert d.x == 10

    def test_attribute_deletion(self):
        d = DotDict({"k": 1})
        del d.k
        assert "k" not in d

    def test_is_a_dict(self):
        d = DotDict({"a": 1})
        assert isinstance(d, dict)
        assert d == {"a": 1}

    def test_nested_attribute_write_sticks(self):
        d = DotDict({"a": {"b": 1}})
        d.a.b = 2
        assert d["a"]["b"] == 2
        assert d.a.b == 2

    def test_missing_attribute_raises_attribute_error(self):
        d = DotDict({"a": 1})
        with pytest.raises(AttributeError, match="missing"):
            _ = d.missing
        assert not hasattr(d, "missing")
        assert getattr(d, "missing", "default") == "default"

    def test_deepcopy(self):
        d = DotDict({"a": {"b": [1, 2]}})
        d2 = copy.deepcopy(d)
        assert isinstance(d2, DotDict)
        assert d2 == d
        d2.a.b.append(3)
        assert d.a.b == [1, 2]

    def test_dir_includes_keys(self):
        d = DotDict({"alpha": 1})
        assert "alpha" in dir(d)


def test_updating_print_uses_carriage_return(capsys):
    updating_print("hello")
    captured = capsys.readouterr()
    assert captured.out == "\rhello"
