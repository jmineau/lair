"""Tests for lair.mesowest.

lair is a toolkit, so the SODAR data directory is resolved from an explicit
argument or the ``LAIR_MESOWEST_DIR`` env var (no hard-coded CHPC path). The
data-reading paths (``parse`` / ``read_data``) need real HDF5 archives and are
left for `chpc`-marked tests; the logic here is exercised offline.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr
import pytest

from lair import mesowest
from lair.mesowest import Sodar, resolve_mesowest_dir


def test_importable():
    """lair.mesowest imports (it used to raise NameError at import)."""
    assert mesowest is not None


class TestResolveMesowestDir:
    def test_explicit_arg_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("LAIR_MESOWEST_DIR", "/from/env")
        assert resolve_mesowest_dir("/explicit") == "/explicit"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("LAIR_MESOWEST_DIR", "/from/env")
        assert resolve_mesowest_dir() == "/from/env"

    def test_raises_when_unconfigured(self, monkeypatch):
        monkeypatch.delenv("LAIR_MESOWEST_DIR", raising=False)
        with pytest.raises(ValueError, match="LAIR_MESOWEST_DIR"):
            resolve_mesowest_dir()


def test_sodar_requires_a_directory(monkeypatch):
    """Constructing a Sodar without a configured directory fails clearly."""
    monkeypatch.delenv("LAIR_MESOWEST_DIR", raising=False)
    with pytest.raises(ValueError, match="MesoWest data directory"):
        Sodar("ABC")


class TestGetFiles:
    """get_files() filters by station id, suffix, and time range.

    The fixture filenames follow the archive layout the module assumes
    (``<SID:6 chars><YYYY>_<MM>_sodar.h5`` so ``file[6:13]`` is ``YYYY_MM``).
    """

    def _sodar_over(self, archive_dir) -> Sodar:
        # Bypass __init__ (which reads HDF5 metadata) to test the listing logic.
        sodar = object.__new__(Sodar)
        sodar.SID = "ABCDEF"
        sodar.archive_dir = str(archive_dir)
        return sodar

    def test_filters_by_sid_and_suffix(self, tmp_path):
        for name in [
            "ABCDEF2024_01_sodar.h5",
            "ABCDEF2024_06_sodar.h5",
            "OTHERX2024_01_sodar.h5",   # different SID
            "ABCDEF2024_01_notes.txt",  # wrong suffix
        ]:
            (tmp_path / name).touch()
        files = self._sodar_over(tmp_path).get_files()
        names = [os.path.basename(f) for f in files]
        assert names == ["ABCDEF2024_01_sodar.h5", "ABCDEF2024_06_sodar.h5"]

    def test_filters_by_time_range(self, tmp_path):
        (tmp_path / "ABCDEF2024_01_sodar.h5").touch()
        (tmp_path / "ABCDEF2024_06_sodar.h5").touch()
        files = self._sodar_over(tmp_path).get_files(time_range=("2024-05", "2024-07"))
        assert [os.path.basename(f) for f in files] == ["ABCDEF2024_06_sodar.h5"]


def test_get_winds_at_height():
    """get_winds_at_height selects the level matching a height and renames cols."""
    times = pd.date_range("2024-01-01", periods=3, freq="h")
    ds = xr.Dataset(
        {
            "HEIGHT": (("Time_UTC", "level"), np.tile([10.0, 50.0], (3, 1))),
            "WD": (("Time_UTC", "level"), np.array([[100, 200], [110, 210], [120, 220]], float)),
            "WS": (("Time_UTC", "level"), np.array([[1, 2], [3, 4], [5, 6]], float)),
        },
        coords={"Time_UTC": times, "level": [0, 1]},
    )
    winds = Sodar.get_winds_at_height(ds, 50.0)
    assert list(winds.columns) == ["direction", "speed"]
    assert winds["direction"].tolist() == [200.0, 210.0, 220.0]
    assert winds["speed"].tolist() == [2.0, 4.0, 6.0]


# TODO (@pytest.mark.chpc): parse()/read_data() against a real HDF5 archive
# under LAIR_MESOWEST_DIR (needs the optional `tables` dependency).
