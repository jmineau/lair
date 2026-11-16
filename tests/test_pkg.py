"""
Package-level smoke tests for lair.

These guard the public surface set up in ``lair/__init__.py`` and the
import-time side effects documented in AGENTS.md.
"""

import hashlib
import logging
import os
import subprocess
import sys

import pint
import pytest

import lair


def test_imports():
    """The top-level package imports without error."""
    assert lair is not None


def test_units_registry_is_pint():
    """``lair.units`` is a usable pint registry (the shared application registry)."""
    from lair import units

    assert isinstance(units, pint.ApplicationRegistry)
    # A basic quantity round-trips through the registry.
    q = 1.0 * units("km")
    assert q.to("m").magnitude == pytest.approx(1000.0)


def test_mass_flux_context_registered():
    """The custom ``mass_flux`` context is registered at import (see __init__)."""
    from lair import units

    # Entering the context by name only succeeds if it was registered.
    with units.context("mass_flux"):
        pass


class TestMassFluxContext:
    """The ``mass_flux`` context converts through ``mw``, a molar mass quantity."""

    def test_converts_both_ways_with_a_molar_mass(self):
        from lair import units

        mw = 16.04 * units("g/mol")
        with units.context("mass_flux", mw=mw):
            mass = (1.0 * units("mol/m**2/s")).to("kg/m**2/s")
            moles = (1.0 * units("kg/m**2/s")).to("mol/m**2/s")
        assert mass.magnitude == pytest.approx(0.01604)
        assert moles.magnitude == pytest.approx(1 / 0.01604)

    @pytest.mark.parametrize(
        "src, dst",
        [("mol/m**2/s", "kg/m**2/s"), ("kg/m**2/s", "mol/m**2/s")],
        ids=["moles to mass", "mass to moles"],
    )
    def test_no_mw_raises_and_names_it(self, src, dst):
        from lair import units

        with (
            units.context("mass_flux"),
            pytest.raises(TypeError, match=r"needs mw.*mw=None"),
        ):
            (1.0 * units(src)).to(dst)

    @pytest.mark.parametrize(
        "mw",
        [
            16.04,
            "16.04 g/mol",
            lair.units.Unit("g/mol"),
            16.04 * lair.units("g"),
        ],
        ids=["plain number", "string", "unit without a number", "mass, not molar mass"],
    )
    def test_mw_that_is_not_a_molar_mass_raises_and_names_it(self, mw):
        from lair import units

        with (
            units.context("mass_flux", mw=mw),
            pytest.raises(TypeError, match=r"needs mw.*quantity.*g/mol"),
        ):
            (1.0 * units("mol/m**2/s")).to("kg/m**2/s")


def test_compound_units_print_in_dimension_order():
    """
    A flux unit prints its parts by dimension, not alphabetically.

    ``inventories.Inventory.get_units`` splits a unit on "/" into its quantity,
    area and time parts, so it relies on this order (pint's default sort gives
    ``kg/d/km**2``). It holds on pint 0.24 and 0.25 and on pint >= 0.26, which
    keep ``sort_by_dimensionality`` in different places.
    """
    from lair import units

    assert units.formatter.default_sort_func is lair.sort_by_dimensionality
    assert f"{units('kg/km**2/day').units:~C}" == "kg/km**2/d"


def test_import_fails_when_pint_has_no_sort_by_dimensionality():
    """
    ``import lair`` raises ``ImportError`` when pint has no ``sort_by_dimensionality``.

    lair used to go on with pint's own unit order, and ``Inventory.get_units``
    then mislabeled a flux unit's parts. The function is removed from both
    places pint keeps it, in a new interpreter, after the packages lair imports
    before its own pint import.
    """
    code = (
        "import importlib\n"
        "import cf_xarray.units, pint_xarray\n"
        "for name in ('pint.delegates.formatter',\n"
        "             'pint.delegates.formatter._compound_unit_helpers'):\n"
        "    try:\n"
        "        module = importlib.import_module(name)\n"
        "    except ImportError:\n"
        "        continue\n"
        "    if hasattr(module, 'sort_by_dimensionality'):\n"
        "        del module.sort_by_dimensionality\n"
        "import lair\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "cannot import name 'sort_by_dimensionality'" in result.stderr


def test_sort_by_dimensionality_in_pints_private_module():
    """
    ``import lair`` finds ``sort_by_dimensionality`` where pint 0.24 and 0.25 keep it.

    Those versions have the function only in
    ``pint.delegates.formatter._compound_unit_helpers``; pint >= 0.26 exports it
    from ``pint.delegates.formatter`` too. A new interpreter gets the older
    layout (the function moved to the private module) before lair is imported,
    so the fallback import runs whichever pint is installed.
    """
    code = (
        "import cf_xarray.units, pint_xarray\n"
        "import pint.delegates.formatter as public\n"
        "from pint.delegates.formatter import _compound_unit_helpers as private\n"
        "func = getattr(public, 'sort_by_dimensionality', None)\n"
        "func = func or private.sort_by_dimensionality\n"
        "if hasattr(public, 'sort_by_dimensionality'):\n"
        "    del public.sort_by_dimensionality\n"
        "private.sort_by_dimensionality = func\n"
        "import lair\n"
        "assert lair.sort_by_dimensionality is func\n"
        "assert lair.units.formatter.default_sort_func is func\n"
        "print(f\"{lair.units('kg/km**2/day').units:~C}\")\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "kg/km**2/d"


def test_version_without_installed_metadata():
    """
    ``lair.__version__`` is ``0+unknown`` when lair is not installed.

    That is a source tree on ``sys.path`` with no package metadata. A new
    interpreter makes the metadata lookup fail with ``PackageNotFoundError``,
    as it does there, for lair only, before lair is imported.
    """
    code = (
        "import importlib.metadata as md\n"
        "real_version = md.version\n"
        "def version(name):\n"
        "    if name == 'lair':\n"
        "        raise md.PackageNotFoundError(name)\n"
        "    return real_version(name)\n"
        "md.version = version\n"
        "import lair\n"
        "print(lair.__version__)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0+unknown"


def test_reexports_present():
    """The handful of symbols __init__ explicitly re-exports are available."""
    assert callable(lair.ftp_download)
    assert callable(lair.unzip)
    assert hasattr(lair, "config")


def test_logging_is_left_to_the_application():
    """
    ``lair`` adds no handlers and sets no level on its loggers.

    Without configuration, Python's last-resort handler then shows WARNING
    and above on stderr, and INFO progress messages stay hidden. (A
    NullHandler would silence the warnings too.)
    """
    import logging

    logger = logging.getLogger("lair")
    assert logger.handlers == []
    assert logger.level == logging.NOTSET


def test_version_from_metadata():
    """``lair.__version__`` comes from the installed metadata (setuptools-scm)."""
    assert isinstance(lair.__version__, str) and lair.__version__


def test_import_makes_no_network_connection_and_no_files(tmp_path):
    """
    ``import lair`` fetches nothing, even with the download allowed.

    A fresh interpreter, with every socket connection made to fail and an empty
    cache folder: the import must still succeed, leave the cache folder absent
    and not import ``lair.background`` (which fetches the CCG filter).
    """
    cache = tmp_path / "cache"
    code = (
        "import socket, sys\n"
        "def refuse(*args, **kwargs):\n"
        "    raise AssertionError('import lair opened a network connection')\n"
        "socket.socket.connect = refuse\n"
        "socket.create_connection = refuse\n"
        "import lair\n"
        "assert 'lair.background' not in sys.modules\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "LAIR_SKIP_CCG_DOWNLOAD"}
    env["LAIR_CACHE_DIR"] = str(cache)
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not cache.exists()


class TestSetupCcgFilter:
    """
    ``setup_ccg_filter`` with the NOAA FTP download stubbed out.

    Each test installs into a temporary directory (``directory=`` or
    ``LAIR_CACHE_DIR``), never into the real cache. The checksum constant is
    replaced by the hash of a small stand-in source, so the stand-in passes.
    """

    REMOTE = "user/thoning/ccgcrv/ccg_filter.zip"
    SOURCE = "def ccgFilter():\n    return 'noaa'\n"

    @pytest.fixture(autouse=True)
    def _allow_download(self, monkeypatch):
        monkeypatch.delenv("LAIR_SKIP_CCG_DOWNLOAD", raising=False)
        monkeypatch.setattr(lair, "_CCG_FILTER_SHA256", self._sha(self.SOURCE))

    @staticmethod
    def _sha(text):
        return hashlib.sha256(text.encode()).hexdigest()

    @classmethod
    def _stub_ftp(cls, monkeypatch, members, calls=None, side_effect=None):
        """Replace lair.ftp_download with one that writes a zip of ``members``."""
        import zipfile

        def fake_ftp_download(host, paths, download_dir, **kwargs):
            if calls is not None:
                calls.append((host, paths, download_dir))
            if side_effect is not None:
                side_effect()
            zf = os.path.join(download_dir, os.path.basename(paths))
            with zipfile.ZipFile(zf, "w") as z:
                for name, text in members.items():
                    z.writestr(name, text)
            return True

        monkeypatch.setattr(lair, "ftp_download", fake_ftp_download)

    @staticmethod
    def _target(directory):
        return directory / "ccg_filter.py"

    def test_installs_only_the_filter_module(self, tmp_path, monkeypatch):
        calls = []
        members = {
            "ccg_filter.py": self.SOURCE,
            "ccg_dates.py": "# dates\n",
            "ccgcrv.py": "# crv\n",
        }
        self._stub_ftp(monkeypatch, members, calls)
        path = lair.setup_ccg_filter(directory=tmp_path)
        assert path == self._target(tmp_path)
        assert len(calls) == 1
        assert calls[0][:2] == ("ftp.gml.noaa.gov", self.REMOTE)
        # Downloaded into a temporary directory, not straight into the folder
        assert calls[0][2] != str(tmp_path)
        assert self._target(tmp_path).read_text() == self.SOURCE
        # No zip, companion files or temporary directory left behind
        assert sorted(p.name for p in tmp_path.iterdir()) == ["ccg_filter.py"]

    def test_lands_in_the_cache_folder_by_default(self, tmp_path, monkeypatch):
        cache = tmp_path / "cache"
        monkeypatch.setenv("LAIR_CACHE_DIR", str(cache))
        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE})
        path = lair.setup_ccg_filter()
        assert path == cache / "ccg_filter" / "ccg_filter.py"
        assert path.read_text() == self.SOURCE
        assert [p.name for p in (cache / "ccg_filter").iterdir()] == ["ccg_filter.py"]

    def test_cache_folder_is_read_when_called(self, tmp_path, monkeypatch):
        # LAIR_CACHE_DIR set after `import lair` is honored
        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE})
        for name in ("first", "second"):
            monkeypatch.setenv("LAIR_CACHE_DIR", str(tmp_path / name))
            path = lair.setup_ccg_filter()
            assert path is not None
            assert path.is_relative_to(tmp_path / name)

    @pytest.mark.parametrize(
        "members",
        [
            {"ccg_filter.py": SOURCE},  # companions dropped from the zip
            {  # new files added to the zip
                "ccg_filter.py": SOURCE,
                "ccg_dates.py": "",
                "ccgcrv.py": "",
                "README.txt": "",
                "examples/demo.py": "",
            },
            {"ccgcrv/ccg_filter.py": SOURCE, "ccgcrv/ccgcrv.py": ""},  # in a folder
        ],
        ids=["no-companions", "extra-files", "nested"],
    )
    def test_tolerates_changed_zip_contents(self, tmp_path, monkeypatch, members):
        self._stub_ftp(monkeypatch, members)
        lair.setup_ccg_filter(directory=tmp_path)
        assert self._target(tmp_path).read_text() == self.SOURCE
        assert sorted(p.name for p in tmp_path.iterdir()) == ["ccg_filter.py"]

    def test_bad_checksum_is_refused(self, tmp_path, monkeypatch):
        # What NOAA serves is not the file lair was written against
        other = "def ccgFilter():\n    return 'changed'\n"
        self._stub_ftp(monkeypatch, {"ccg_filter.py": other})
        with pytest.raises(ValueError, match="SHA-256") as info:
            lair.setup_ccg_filter(directory=tmp_path)
        assert self._sha(other) in str(info.value)
        assert self._sha(self.SOURCE) in str(info.value)
        assert list(tmp_path.iterdir()) == []

    def test_real_checksum_refuses_a_stand_in(self, tmp_path, monkeypatch):
        # With lair's own constant, none of these tests' sources would pass
        monkeypatch.undo()
        monkeypatch.delenv("LAIR_SKIP_CCG_DOWNLOAD", raising=False)
        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE})
        assert len(lair._CCG_FILTER_SHA256) == 64
        with pytest.raises(ValueError, match="SHA-256"):
            lair.setup_ccg_filter(directory=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_existing_target_is_kept(self, tmp_path, monkeypatch):
        calls = []
        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE}, calls)
        self._target(tmp_path).write_text(self.SOURCE)
        assert lair.setup_ccg_filter(directory=tmp_path) == self._target(tmp_path)
        assert calls == []

    def test_existing_target_with_wrong_checksum_is_replaced(
        self, tmp_path, monkeypatch, caplog
    ):
        calls = []
        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE}, calls)
        self._target(tmp_path).write_text("# edited or truncated\n")
        with caplog.at_level(logging.WARNING, logger="lair"):
            lair.setup_ccg_filter(directory=tmp_path)
        assert len(calls) == 1
        assert self._target(tmp_path).read_text() == self.SOURCE
        assert "checksum" in caplog.text

    def test_target_written_meanwhile_is_used(self, tmp_path, monkeypatch):
        # Another process (e.g. a SLURM array task) finishes first
        def other_process_wins():
            self._target(tmp_path).write_text(self.SOURCE)

        self._stub_ftp(
            monkeypatch, {"ccg_filter.py": self.SOURCE}, side_effect=other_process_wins
        )
        assert lair.setup_ccg_filter(directory=tmp_path) == self._target(tmp_path)
        assert self._target(tmp_path).read_text() == self.SOURCE
        assert sorted(p.name for p in tmp_path.iterdir()) == ["ccg_filter.py"]

    def test_skip_env_var(self, tmp_path, monkeypatch):
        calls = []
        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE}, calls)
        monkeypatch.setenv("LAIR_SKIP_CCG_DOWNLOAD", "1")
        assert lair.setup_ccg_filter(directory=tmp_path) is None
        assert calls == []
        assert list(tmp_path.iterdir()) == []

    def test_skip_env_var_still_uses_a_good_copy(self, tmp_path, monkeypatch):
        calls = []
        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE}, calls)
        self._target(tmp_path).write_text(self.SOURCE)
        monkeypatch.setenv("LAIR_SKIP_CCG_DOWNLOAD", "0")  # any text skips
        assert lair.setup_ccg_filter(directory=tmp_path) == self._target(tmp_path)
        assert calls == []

    def test_skip_env_var_does_not_use_a_bad_copy(self, tmp_path, monkeypatch, caplog):
        self._target(tmp_path).write_text("# edited\n")
        monkeypatch.setenv("LAIR_SKIP_CCG_DOWNLOAD", "1")
        with caplog.at_level(logging.WARNING, logger="lair"):
            assert lair.setup_ccg_filter(directory=tmp_path) is None
        assert "Not using" in caplog.text
        assert self._target(tmp_path).read_text() == "# edited\n"

    def test_download_failure_leaves_nothing(self, tmp_path, monkeypatch):
        def fail():
            raise OSError("FTP unreachable")

        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE}, side_effect=fail)
        with pytest.raises(OSError, match="FTP unreachable"):
            lair.setup_ccg_filter(directory=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_zip_without_filter_leaves_nothing(self, tmp_path, monkeypatch):
        self._stub_ftp(monkeypatch, {"ccgcrv.py": "", "README.txt": ""})
        with pytest.raises(FileNotFoundError, match="ccg_filter.py"):
            lair.setup_ccg_filter(directory=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_corrupt_zip_leaves_nothing(self, tmp_path, monkeypatch):
        def fake_ftp_download(host, paths, download_dir, **kwargs):
            with open(os.path.join(download_dir, "ccg_filter.zip"), "wb") as f:
                f.write(b"not a zip")

        monkeypatch.setattr(lair, "ftp_download", fake_ftp_download)
        import zipfile

        with pytest.raises(zipfile.BadZipFile):
            lair.setup_ccg_filter(directory=tmp_path)
        assert list(tmp_path.iterdir()) == []

    def test_unwritable_cache_folder_raises(self, tmp_path, monkeypatch):
        # A path that cannot be made (a file is in the way), as in a read-only home
        blocker = tmp_path / "blocker"
        blocker.write_text("")
        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE})
        with pytest.raises(OSError):
            lair.setup_ccg_filter(directory=blocker / "ccg_filter")
