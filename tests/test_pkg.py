"""Package-level smoke tests for lair.

These guard the public surface set up in ``lair/__init__.py`` and the
import-time side effects documented in AGENTS.md.
"""

import os

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


def test_reexports_present():
    """The handful of symbols __init__ explicitly re-exports are available."""
    assert callable(lair.ftp_download)
    assert callable(lair.unzip)
    assert hasattr(lair, "config")


def test_logging_is_left_to_the_application():
    """lair adds no handlers and sets no level on its loggers.

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


class TestSetupCcgFilter:
    """``setup_ccg_filter`` with the NOAA FTP download stubbed out.

    Each test installs into a temporary directory (``lair_dir=``), never into
    the package itself.
    """

    REMOTE = "user/thoning/ccgcrv/ccg_filter.zip"
    SOURCE = "def ccgFilter():\n    return 'noaa'\n"

    @pytest.fixture(autouse=True)
    def _allow_download(self, monkeypatch):
        monkeypatch.delenv("LAIR_SKIP_CCG_DOWNLOAD", raising=False)

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
    def _target(lair_dir):
        return lair_dir / "_ccg_filter.py"

    def test_installs_only_the_filter_module(self, tmp_path, monkeypatch):
        calls = []
        members = {
            "ccg_filter.py": self.SOURCE,
            "ccg_dates.py": "# dates\n",
            "ccgcrv.py": "# crv\n",
        }
        self._stub_ftp(monkeypatch, members, calls)
        lair.setup_ccg_filter(lair_dir=str(tmp_path))
        assert len(calls) == 1
        assert calls[0][:2] == ("ftp.gml.noaa.gov", self.REMOTE)
        # Downloaded into a temporary directory, not straight into lair_dir
        assert calls[0][2] != str(tmp_path)
        assert self._target(tmp_path).read_text() == self.SOURCE
        # No zip, companion files or temporary directory left behind
        assert sorted(p.name for p in tmp_path.iterdir()) == ["_ccg_filter.py"]

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
        lair.setup_ccg_filter(lair_dir=str(tmp_path))
        assert self._target(tmp_path).read_text() == self.SOURCE
        assert sorted(p.name for p in tmp_path.iterdir()) == ["_ccg_filter.py"]

    def test_existing_target_is_kept(self, tmp_path, monkeypatch):
        calls = []
        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE}, calls)
        self._target(tmp_path).write_text("# already installed\n")
        lair.setup_ccg_filter(lair_dir=str(tmp_path))
        assert calls == []
        assert self._target(tmp_path).read_text() == "# already installed\n"

    def test_target_written_meanwhile_is_used(self, tmp_path, monkeypatch):
        # Another process (e.g. a SLURM array task) finishes first
        def other_process_wins():
            self._target(tmp_path).write_text("# from another process\n")

        self._stub_ftp(
            monkeypatch, {"ccg_filter.py": self.SOURCE}, side_effect=other_process_wins
        )
        lair.setup_ccg_filter(lair_dir=str(tmp_path))
        assert self._target(tmp_path).read_text() == "# from another process\n"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["_ccg_filter.py"]

    def test_skip_env_var(self, tmp_path, monkeypatch):
        calls = []
        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE}, calls)
        monkeypatch.setenv("LAIR_SKIP_CCG_DOWNLOAD", "1")
        lair.setup_ccg_filter(lair_dir=str(tmp_path))
        assert calls == []
        assert list(tmp_path.iterdir()) == []

    def test_download_failure_leaves_nothing(self, tmp_path, monkeypatch):
        def fail():
            raise OSError("FTP unreachable")

        self._stub_ftp(monkeypatch, {"ccg_filter.py": self.SOURCE}, side_effect=fail)
        with pytest.raises(OSError, match="FTP unreachable"):
            lair.setup_ccg_filter(lair_dir=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_zip_without_filter_leaves_nothing(self, tmp_path, monkeypatch):
        self._stub_ftp(monkeypatch, {"ccgcrv.py": "", "README.txt": ""})
        with pytest.raises(FileNotFoundError, match="ccg_filter.py"):
            lair.setup_ccg_filter(lair_dir=str(tmp_path))
        assert list(tmp_path.iterdir()) == []

    def test_corrupt_zip_leaves_nothing(self, tmp_path, monkeypatch):
        def fake_ftp_download(host, paths, download_dir, **kwargs):
            with open(os.path.join(download_dir, "ccg_filter.zip"), "wb") as f:
                f.write(b"not a zip")

        monkeypatch.setattr(lair, "ftp_download", fake_ftp_download)
        import zipfile

        with pytest.raises(zipfile.BadZipFile):
            lair.setup_ccg_filter(lair_dir=str(tmp_path))
        assert list(tmp_path.iterdir()) == []
