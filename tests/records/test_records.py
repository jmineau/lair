"""Tests for lair.records (file/dir utilities).

Network helpers (ftp_download, wget_download) are not exercised here; the pure
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
