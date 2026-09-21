"""Tests for lair.records (file/dir utilities).

Network helpers (ftp_download, wget_download) are not exercised against real
servers; wget_download is checked with subprocess.run stubbed out, and the pure
filesystem helpers are tested against tmp_path fixtures.
"""

import os
import zipfile

import pytest

from lair import records


class TestUnzip:
    def _make_zip(self, path, members):
        with zipfile.ZipFile(path, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)

    def test_extracts_into_given_dir(self, tmp_path):
        zf = tmp_path / "bundle.zip"
        self._make_zip(zf, {"a.txt": "hello", "sub/b.txt": "world"})
        out = tmp_path / "out"
        out.mkdir()
        records.unzip(str(zf), str(out))
        assert (out / "a.txt").read_text() == "hello"
        assert (out / "sub" / "b.txt").read_text() == "world"

    def test_defaults_to_archive_dir(self, tmp_path):
        zf = tmp_path / "bundle.zip"
        self._make_zip(zf, {"only.txt": "data"})
        records.unzip(str(zf))  # no dir_path -> extract alongside the archive
        assert (tmp_path / "only.txt").read_text() == "data"


class TestListFiles:
    @pytest.fixture
    def tree(self, tmp_path):
        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.csv").write_text("")
        (tmp_path / ".hidden").write_text("")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.txt").write_text("")
        return tmp_path

    def test_lists_visible_entries_excluding_hidden(self, tree):
        # Non-recursive list returns visible entries (incl. subdir names);
        # dotfiles are excluded by default.
        names = records.list_files(tree)
        assert {"a.txt", "b.csv"} <= set(names)
        assert ".hidden" not in names

    def test_pattern_filter(self, tree):
        assert records.list_files(tree, pattern="*.txt") == ["a.txt"]

    def test_all_files_includes_hidden(self, tree):
        assert ".hidden" in records.list_files(tree, all_files=True)

    def test_recursive_full_names(self, tree):
        found = records.list_files(tree, pattern="*.txt", recursive=True, full_names=True)
        base = {os.path.basename(f) for f in found}
        assert base == {"a.txt", "c.txt"}


class TestCacher:
    def test_caches_and_reuses_result(self, tmp_path):
        calls = {"n": 0}

        def square(x):
            calls["n"] += 1
            return x * x

        cache_file = str(tmp_path / "cache.pkl")
        cached = records.Cacher(square, cache_file)

        assert cached(3) == 9
        assert calls["n"] == 1
        # Second call with the same args hits the cache (func not re-run).
        assert cached(3) == 9
        assert calls["n"] == 1
        # Different args -> function runs again.
        assert cached(4) == 16
        assert calls["n"] == 2

    def test_persists_across_instances(self, tmp_path):
        calls = {"n": 0}

        def square(x):
            calls["n"] += 1
            return x * x

        cache_file = str(tmp_path / "cache.pkl")
        records.Cacher(square, cache_file)(5)
        assert calls["n"] == 1
        # A fresh Cacher over the same file reloads the index and reuses results.
        assert records.Cacher(square, cache_file)(5) == 25
        assert calls["n"] == 1

    def test_bare_filename_uses_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cacher = records.Cacher(lambda x: x, "cache.pkl")
        assert cacher.index_file == ".cache.pkl.index"

    def test_requires_pkl_extension(self, tmp_path):
        with pytest.raises(AssertionError):
            records.Cacher(lambda x: x, str(tmp_path / "cache.txt"))


def test_read_kml(tmp_path):
    # Requires the optional fastkml dependency (formats extra).
    pytest.importorskip("fastkml")
    kml_path = tmp_path / "t.kml"
    kml_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<kml xmlns="http://www.opengis.net/kml/2.2">'
        "<Document><name>t</name></Document></kml>"
    )
    k = records.read_kml(str(kml_path))
    assert k is not None


class TestWgetDownload:
    @pytest.fixture
    def calls(self, monkeypatch):
        """Record the commands wget_download would run instead of running them."""
        import subprocess

        calls = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))
        return calls

    def test_single_url_string_is_one_download(self, tmp_path, calls):
        url = "https://example.com/data/file.csv"
        records.wget_download(url, str(tmp_path))
        assert calls == [["wget", "-O", str(tmp_path / "file.csv"), url]]

    def test_prefix_with_leading_slash(self, tmp_path, calls):
        url = "https://example.com/pub/data/file.csv"
        records.wget_download(url, str(tmp_path), prefix="/pub")
        assert calls[0][2] == str(tmp_path / "data" / "file.csv")

    def test_worker_errors_are_raised(self, tmp_path, monkeypatch):
        import subprocess

        def fake_run(cmd, check):
            if cmd[0] == "unzip":
                raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(subprocess.CalledProcessError):
            records.wget_download(["https://example.com/a.zip"], str(tmp_path))
