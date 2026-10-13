"""Tests for lair.config (data-dir resolution, cache location)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from lair import config


def test_lair_dir_is_a_string():
    assert isinstance(config.LAIR_DIR, str)


def test_no_builtin_chpc_paths():
    # lair resolves data locations from arguments / LAIR_* env vars only. That
    # includes the cache folder: a CACHE_DIR constant would be read at import
    # (see get_cache_dir).
    for name in (
        "HOME",
        "GROUP_DIR",
        "INVENTORY_DIR",
        "MET_DIR",
        "SPATIAL_DIR",
        "CACHE_DIR",
    ):
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


class TestGetCacheDir:
    def test_env_variable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAIR_CACHE_DIR", str(tmp_path / "cache"))
        assert config.get_cache_dir() == tmp_path / "cache"

    def test_variable_is_read_at_each_call(self, tmp_path, monkeypatch):
        # lair is already imported: a variable set now must be seen
        for name in ("first", "second"):
            monkeypatch.setenv("LAIR_CACHE_DIR", str(tmp_path / name))
            assert config.get_cache_dir() == tmp_path / name

    @pytest.mark.parametrize("value", [None, ""])
    def test_default_is_dot_cache_lair_in_home(self, tmp_path, monkeypatch, value):
        # unset and empty both mean the default (get_data_dir does the same)
        if value is None:
            monkeypatch.delenv("LAIR_CACHE_DIR", raising=False)
        else:
            monkeypatch.setenv("LAIR_CACHE_DIR", value)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows
        assert config.get_cache_dir() == tmp_path / ".cache" / "lair"

    def test_makes_no_folder(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LAIR_CACHE_DIR", str(tmp_path / "cache"))
        config.get_cache_dir()
        assert not (tmp_path / "cache").exists()
        monkeypatch.delenv("LAIR_CACHE_DIR")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        config.get_cache_dir()
        assert not (tmp_path / ".cache").exists()


@pytest.fixture(params=["writable", "read-only", "file"])
def home(request, tmp_path):
    """A home directory that is writable, read-only, or not a folder at all."""
    home = tmp_path / "home"
    if request.param == "file":
        # Works for root too: nothing can be made inside a regular file
        home.write_text("not a folder")
        yield home
        return
    home.mkdir()
    if request.param == "read-only":
        home.chmod(0o500)
        try:
            (home / "probe").mkdir()
        except OSError:
            pass
        else:
            home.chmod(0o700)
            pytest.skip("cannot make a read-only folder (running as root?)")
    yield home
    home.chmod(0o700)  # let pytest clean up


def test_import_makes_no_cache_folder(home, tmp_path):
    # `import lair` must not touch the file system for the cache: with a
    # read-only home (a container, a sandbox) it used to fail with a
    # PermissionError, even for code that never reads the cache.
    env = {k: v for k, v in os.environ.items() if k not in ("LAIR_CACHE_DIR", "HOME")}
    env.update(HOME=str(home), USERPROFILE=str(home), LAIR_SKIP_CCG_DOWNLOAD="1")
    result = subprocess.run(
        [sys.executable, "-c", "import lair"],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (home / ".cache").exists()


def test_cache_variable_set_after_import_is_used(tmp_path):
    # The example from the bug report, in a fresh interpreter: the variable is
    # set after `import lair` and `import lair.noaa`, so the cache must go to
    # the new folder and the default one (in the home folder) must stay absent.
    home = tmp_path / "home"
    home.mkdir()
    new = tmp_path / "somewhere" / "else"
    code = (
        "import os, lair, lair.noaa\n"
        "os.environ['LAIR_CACHE_DIR'] = os.environ.pop('NEW_CACHE_DIR')\n"
        "print(lair.noaa._cache_file('carbontracker', 'x.pkl'))\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("LAIR_CACHE_DIR", "HOME")}
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        LAIR_SKIP_CCG_DOWNLOAD="1",
        NEW_CACHE_DIR=str(new),
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == new / "carbontracker" / "x.pkl"
    assert (new / "carbontracker").is_dir()
    assert not (home / ".cache").exists()


def test_import_makes_no_folder_for_lair_cache_dir(tmp_path):
    # A variable set before the import is not a reason to make its folder
    # either: only the code that caches does, on first use.
    cache = tmp_path / "cache"
    env = {k: v for k, v in os.environ.items() if k != "LAIR_CACHE_DIR"}
    env.update(LAIR_CACHE_DIR=str(cache), LAIR_SKIP_CCG_DOWNLOAD="1")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import lair, lair.config, lair.noaa; lair.config.get_cache_dir()",
        ],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not cache.exists()


def test_no_verbose_flag():
    # Verbosity moved to the standard logging module (#22)
    assert not hasattr(config, "verbose")
    assert not hasattr(config, "vprint")
