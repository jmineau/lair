"""Tests for lair.config (data-dir resolution, paths + verbosity toggle).

NOTE: importing lair.config has side effects (creates CACHE_DIR, sets a pandas
option). Verbosity is a plain boolean, not the stdlib logging module.
"""

import os
from pathlib import Path

import pytest

from lair import config


def test_directories_are_strings():
    assert isinstance(config.LAIR_DIR, str)
    assert isinstance(config.CACHE_DIR, str)


def test_no_builtin_chpc_paths():
    # lair resolves data locations from arguments / LAIR_* env vars only
    for name in ("HOME", "GROUP_DIR", "INVENTORY_DIR", "MET_DIR", "SPATIAL_DIR"):
        assert not hasattr(config, name)


class TestGetDataDir:
    def test_explicit_path_wins(self, monkeypatch):
        monkeypatch.setenv("LAIR_TEST_DIR", "/from/env")
        assert config.get_data_dir("LAIR_TEST_DIR", "/explicit") == Path("/explicit")

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("LAIR_TEST_DIR", "/from/env")
        assert config.get_data_dir("LAIR_TEST_DIR") == Path("/from/env")

    def test_unset_raises_naming_the_variable(self, monkeypatch):
        monkeypatch.delenv("LAIR_TEST_DIR", raising=False)
        with pytest.raises(ValueError, match="LAIR_TEST_DIR"):
            config.get_data_dir("LAIR_TEST_DIR")


def test_cache_dir_created_on_import():
    # config creates CACHE_DIR at import time.
    assert os.path.isdir(config.CACHE_DIR)


def test_verbose_is_boolean():
    assert isinstance(config.verbose, bool)


def test_vprint_respects_verbose_flag(capsys):
    original = config.verbose
    try:
        config.verbose = True
        config.vprint("loud")
        assert "loud" in capsys.readouterr().out

        config.verbose = False
        config.vprint("silent")
        assert capsys.readouterr().out == ""
    finally:
        config.verbose = original
