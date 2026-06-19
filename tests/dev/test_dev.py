"""Tests for lair.dev (development helpers)."""

from lair.dev import public_attrs


class _Sample:
    def __init__(self):
        self.visible = 1
        self._hidden = 2

    def method(self):
        return None

    def _private_method(self):
        return None


def test_public_attrs_excludes_underscored():
    attrs = public_attrs(_Sample())
    assert "visible" in attrs
    assert "method" in attrs
    assert "_hidden" not in attrs
    assert "_private_method" not in attrs
    assert not any(a.startswith("_") for a in attrs)


def test_public_attrs_returns_list():
    assert isinstance(public_attrs(object()), list)
