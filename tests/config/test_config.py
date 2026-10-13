"""Tests for lair.config (data-dir resolution, cache location)."""

import os
import subprocess
import sys
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


def test_no_verbose_flag():
    # Verbosity moved to the standard logging module (#22)
    assert not hasattr(config, "verbose")
    assert not hasattr(config, "vprint")
