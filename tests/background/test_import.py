"""
How lair.background gets, or fails to get, the NOAA CCG filter.

Kept apart from test_background.py, which skips entirely when the filter is not
installed: these tests fake the download and the cache folder, so they run
either way and never reach NOAA.
"""

import hashlib
import importlib
import os
import sys
import zipfile

import pytest

import lair

SOURCE = "def ccgFilter():\n    return 'noaa'\n"
MODULE = "lair._noaa_ccg_filter"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty cache folder, the download allowed, and lair.background unimported."""
    monkeypatch.setenv("LAIR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("LAIR_SKIP_CCG_DOWNLOAD", raising=False)
    monkeypatch.setattr(
        lair, "_CCG_FILTER_SHA256", hashlib.sha256(SOURCE.encode()).hexdigest()
    )
    saved = {k: sys.modules.pop(k, None) for k in ("lair.background", MODULE)}
    yield tmp_path / "cache"
    for name, module in saved.items():
        sys.modules.pop(name, None)
        if module is not None:
            sys.modules[name] = module


def fake_ftp(monkeypatch, source=SOURCE, calls=None):
    """Replace lair.ftp_download with one that writes a zip holding ``source``."""

    def ftp_download(host, paths, download_dir, **kwargs):
        if calls is not None:
            calls.append(host)
        with zipfile.ZipFile(os.path.join(download_dir, "ccg_filter.zip"), "w") as z:
            z.writestr("ccg_filter.py", source)
        return True

    monkeypatch.setattr(lair, "ftp_download", ftp_download)


def import_background():
    return importlib.import_module("lair.background")


def test_first_import_fetches_into_the_cache_folder(cache, monkeypatch):
    calls = []
    fake_ftp(monkeypatch, calls=calls)
    background = import_background()
    assert calls == ["ftp.gml.noaa.gov"]
    assert (cache / "ccg_filter" / "ccg_filter.py").read_text() == SOURCE
    assert background.ccgFilter() == "noaa"
    assert sys.modules[MODULE].__file__ == str(cache / "ccg_filter" / "ccg_filter.py")


def test_later_import_uses_the_cached_file(cache, monkeypatch):
    fake_ftp(monkeypatch)
    import_background()
    for name in ("lair.background", MODULE):  # a new process, as far as lair knows
        del sys.modules[name]
    calls = []
    fake_ftp(monkeypatch, calls=calls)
    assert import_background().ccgFilter() == "noaa"
    assert calls == []


def test_old_copy_in_the_package_folder_is_ignored(cache, monkeypatch):
    # An older lair left lair/_ccg_filter.py beside __init__.py. Whatever is
    # importable under that name is not used: the checked copy is.
    import types

    old = types.ModuleType("lair._ccg_filter")
    old.ccgFilter = lambda: "old copy"
    monkeypatch.setitem(sys.modules, "lair._ccg_filter", old)
    fake_ftp(monkeypatch)
    assert import_background().ccgFilter() == "noaa"


def test_skipped_download_says_how_to_install_it(cache, monkeypatch):
    calls = []
    fake_ftp(monkeypatch, calls=calls)
    monkeypatch.setenv("LAIR_SKIP_CCG_DOWNLOAD", "1")
    with pytest.raises(ImportError, match=r"lair\.setup_ccg_filter\(\)") as info:
        import_background()
    assert not isinstance(info.value, ModuleNotFoundError)
    assert "LAIR_SKIP_CCG_DOWNLOAD" in str(info.value)
    assert calls == []
    assert not cache.exists()


def test_failed_download_says_why(cache, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("FTP unreachable")

    monkeypatch.setattr(lair, "ftp_download", fail)
    with pytest.raises(ImportError, match=r"lair\.setup_ccg_filter\(\)") as info:
        import_background()
    assert "FTP unreachable" in str(info.value)
    assert isinstance(info.value.__cause__, OSError)
    assert not (cache / "ccg_filter" / "ccg_filter.py").exists()


def test_bad_checksum_is_refused_and_not_imported(cache, monkeypatch):
    fake_ftp(monkeypatch, source="def ccgFilter():\n    return 'tampered'\n")
    with pytest.raises(ImportError, match="SHA-256") as info:
        import_background()
    assert isinstance(info.value.__cause__, ValueError)
    assert not (cache / "ccg_filter" / "ccg_filter.py").exists()
    assert MODULE not in sys.modules


def test_cached_file_with_wrong_checksum_is_not_run(cache, monkeypatch):
    # A file edited after it was fetched must not be executed
    (cache / "ccg_filter").mkdir(parents=True)
    bad = cache / "ccg_filter" / "ccg_filter.py"
    bad.write_text("raise SystemExit('ran an unchecked file')\n")
    monkeypatch.setenv("LAIR_SKIP_CCG_DOWNLOAD", "1")
    with pytest.raises(ImportError, match=r"lair\.setup_ccg_filter\(\)"):
        import_background()


def test_missing_dependency_of_the_filter_is_not_masked(cache, monkeypatch):
    # The filter is there but needs scipy, which isn't: say so, not "run
    # setup_ccg_filter()"
    needs_scipy = "import a_package_that_is_not_installed\n"
    monkeypatch.setattr(
        lair, "_CCG_FILTER_SHA256", hashlib.sha256(needs_scipy.encode()).hexdigest()
    )
    fake_ftp(monkeypatch, source=needs_scipy)
    with pytest.raises(ModuleNotFoundError) as info:
        import_background()
    assert info.value.name == "a_package_that_is_not_installed"
    assert MODULE not in sys.modules
