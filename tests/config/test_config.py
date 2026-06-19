"""Tests for lair.config (paths + verbosity toggle).

NOTE: importing lair.config has side effects (creates CACHE_DIR, sets a pandas
option). Verbosity is a plain boolean, not the stdlib logging module.
"""

import os

from lair import config


def test_directories_are_strings():
    assert isinstance(config.HOME, str)
    assert isinstance(config.LAIR_DIR, str)
    assert isinstance(config.CACHE_DIR, str)


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
