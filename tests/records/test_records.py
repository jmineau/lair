"""
Tests for lair.records (file/dir utilities).

Network helpers (ftp_download, wget_download) are not exercised against real
servers: wget_download runs with subprocess.run stubbed out and ftp_download
against an in-memory fake of ftplib.FTP. The pure filesystem helpers are tested
against tmp_path fixtures.
"""

import functools
import os
import subprocess
import sys
import threading
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

    def test_ignore_case(self, tree):
        (tree / "D.TXT").write_text("")
        assert records.list_files(tree, pattern="*.txt") == ["a.txt"]
        found = records.list_files(tree, pattern="*.Txt", ignore_case=True)
        assert sorted(found) == ["D.TXT", "a.txt"]  # original case is returned

    def test_recursive_full_names(self, tree):
        found = records.list_files(
            tree, pattern="*.txt", recursive=True, full_names=True
        )
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

    def test_reload_reruns_once_and_keeps_other_entries(self, tmp_path):
        calls = []

        def square(x):
            calls.append(x)
            return x * x

        cache_file = str(tmp_path / "cache.pkl")
        cached = records.Cacher(square, cache_file)
        cached(1)
        cached(2)

        # reload=True re-runs each set of args once, then serves the fresh result.
        reloaded = records.Cacher(square, cache_file, reload=True)
        assert reloaded(1) == 1
        assert reloaded(1) == 1
        assert calls == [1, 2, 1]

        # Entries that were not refreshed are still readable afterwards (#26).
        assert records.Cacher(square, cache_file)(2) == 4
        assert calls == [1, 2, 1]

    def test_reload_result_replaces_old_one(self, tmp_path):
        value = {"v": "old"}
        cache_file = str(tmp_path / "cache.pkl")
        records.Cacher(lambda: value["v"], cache_file)()

        value["v"] = "new"
        assert records.Cacher(lambda: value["v"], cache_file, reload=True)() == "new"
        assert records.Cacher(lambda: value["v"], cache_file)() == "new"

    def test_bare_filename_uses_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cacher = records.Cacher(lambda x: x, "cache.pkl")
        assert cacher.index_file == ".cache.pkl.index"

    def test_requires_pkl_extension(self, tmp_path):
        with pytest.raises(ValueError, match=".pkl"):
            records.Cacher(lambda x: x, str(tmp_path / "cache.txt"))


def run_count(tmp_path, source, *args, name="f", **kwargs):
    """Cache ``source``'s function in a file shared by every call; return (result, ran)."""
    ran = []
    namespace = {"__name__": "edited_module", "ran": ran}
    exec(compile(source, "edited_module.py", "exec"), namespace)
    cached = records.Cacher(namespace[name], str(tmp_path / "cache.pkl"))
    return cached(*args, **kwargs), len(ran)


class TestCacherKeyHoldsTheCode:
    """Editing a function runs it again (#219); an edit with no effect does not."""

    BODY = "def f(x):\n    ran.append(1)\n    return x * {factor}\n"

    def test_edited_body_runs_again(self, tmp_path):
        assert run_count(tmp_path, self.BODY.format(factor=2), 3) == (6, 1)
        assert run_count(tmp_path, self.BODY.format(factor=2), 3) == (6, 0)
        assert run_count(tmp_path, self.BODY.format(factor=3), 3) == (9, 1)

    def test_changed_constant_runs_again(self, tmp_path):
        assert run_count(tmp_path, self.BODY.format(factor=1000), 1) == (1000, 1)
        assert run_count(tmp_path, self.BODY.format(factor=100), 1) == (100, 1)

    def test_changed_name_in_body_runs_again(self, tmp_path):
        source = "def f(x):\n    ran.append(1)\n    return x.{attr}\n"
        assert run_count(tmp_path, source.format(attr="real"), 3) == (3, 1)
        assert run_count(tmp_path, source.format(attr="imag"), 3) == (0, 1)

    def test_comment_blank_lines_and_local_names_keep_the_result(self, tmp_path):
        first = "def f(x):\n    ran.append(1)\n    y = x * 2\n    return y\n"
        second = (
            "\n\n# a comment\ndef f(x):\n    ran.append(1)  # why\n"
            "    z = x * 2\n\n    return z\n"
        )
        assert run_count(tmp_path, first, 3) == (6, 1)
        assert run_count(tmp_path, second, 3) == (6, 0)

    def test_docstring_edit_keeps_the_result(self, tmp_path):
        source = 'def f(x):\n    """{doc}"""\n    ran.append(1)\n    return x\n'
        assert run_count(tmp_path, source.format(doc="One."), 3) == (3, 1)
        assert run_count(tmp_path, source.format(doc="Two, longer."), 3) == (3, 0)

    def test_other_name_or_module_does_not_share_results(self, tmp_path):
        source = "def {name}(x):\n    ran.append(1)\n    return x\n"
        assert run_count(tmp_path, source.format(name="f"), 3, name="f") == (3, 1)
        assert run_count(tmp_path, source.format(name="g"), 3, name="g") == (3, 1)

    def test_function_in_the_arguments_is_keyed_on_its_code(self, tmp_path):
        # the case in the issue: preprocess= edited without a rename
        pre = "def preprocess(x):\n    return x * {factor}\n"
        run = "def f(data, preprocess):\n    ran.append(1)\n    return [preprocess(v) for v in data]\n"

        def go(factor):
            ran = []
            namespace = {"__name__": "edited_module", "ran": ran}
            exec(
                compile(pre.format(factor=factor) + run, "edited_module.py", "exec"),
                namespace,
            )
            cached = records.Cacher(namespace["f"], str(tmp_path / "cache.pkl"))
            return cached([1, 2], preprocess=namespace["preprocess"]), len(ran)

        assert go(2) == ([2, 4], 1)
        assert go(2) == ([2, 4], 0)
        assert go(100) == ([100, 200], 1)

    def test_default_argument_function_and_value(self, tmp_path):
        source = (
            "def helper(x):\n    return x + {offset}\n"
            "def f(x, h=helper, scale={scale}):\n    ran.append(1)\n"
            "    return h(x) * scale\n"
        )
        assert run_count(tmp_path, source.format(offset=1, scale=2), 1) == (4, 1)
        assert run_count(tmp_path, source.format(offset=1, scale=2), 1) == (4, 0)
        assert run_count(tmp_path, source.format(offset=5, scale=2), 1) == (12, 1)
        assert run_count(tmp_path, source.format(offset=5, scale=3), 1) == (18, 1)

    def test_nested_function_edit(self, tmp_path):
        source = (
            "def f(x):\n    ran.append(1)\n"
            "    def inner(v):\n        return v + {offset}\n"
            "    return inner(x)\n"
        )
        assert run_count(tmp_path, source.format(offset=1), 1) == (2, 1)
        assert run_count(tmp_path, source.format(offset=1), 1) == (2, 0)
        assert run_count(tmp_path, source.format(offset=7), 1) == (8, 1)

    def test_decorated_function_edit(self, tmp_path):
        source = (
            "import functools\n"
            "def deco(g):\n"
            "    @functools.wraps(g)\n"
            "    def wrapper(*a, **k):\n        ran.append(1)\n        return g(*a, **k)\n"
            "    return wrapper\n"
            "@deco\ndef f(x):\n    return x * {factor}\n"
        )
        assert run_count(tmp_path, source.format(factor=2), 3) == (6, 1)
        assert run_count(tmp_path, source.format(factor=2), 3) == (6, 0)
        assert run_count(tmp_path, source.format(factor=4), 3) == (12, 1)

    def test_helper_edit_is_not_seen(self, tmp_path):
        # the documented limit: only the function itself is in the key
        source = (
            "def helper(x):\n    return x * {factor}\n"
            "def f(x):\n    ran.append(1)\n    return helper(x)\n"
        )
        assert run_count(tmp_path, source.format(factor=2), 3) == (6, 1)
        assert run_count(tmp_path, source.format(factor=3), 3) == (6, 0)

    def test_recursive_closure_ends(self, tmp_path):
        source = (
            "def f(n):\n    ran.append(1)\n"
            "    def fact(k):\n        return 1 if k < 2 else k * fact(k - 1)\n"
            "    return fact(n)\n"
        )
        assert run_count(tmp_path, source, 4) == (24, 1)
        assert run_count(tmp_path, source, 4) == (24, 0)

    def test_recursive_function_in_the_arguments(self, tmp_path):
        def make():
            def fact(k):
                return 1 if k < 2 else k * fact(k - 1)

            return fact

        cached = records.Cacher(lambda fn: fn(4), str(tmp_path / "other.pkl"))
        assert cached(make()) == 24
        assert cached(make()) == 24

    def test_function_with_an_empty_closure_cell(self, tmp_path):
        def make():
            def g():
                return later

            yield g  # `later` has no value yet
            later = 1

        cached = records.Cacher(lambda fn: "ok", str(tmp_path / "cache.pkl"))
        assert cached(next(make())) == "ok"

    def test_tuple_and_set_constants(self, tmp_path):
        source = (
            "def f(x):\n    ran.append(1)\n"
            "    return x in ('a', 'b') or x in {{'c', 'd', {extra!r}}}\n"
        )
        assert run_count(tmp_path, source.format(extra="e"), "c") == (True, 1)
        assert run_count(tmp_path, source.format(extra="e"), "c") == (True, 0)
        assert run_count(tmp_path, source.format(extra="z"), "c") == (True, 1)

    def test_set_constant_is_stable_across_processes(self, tmp_path):
        # `in {...}` compiles to a frozenset constant, whose order depends on
        # the string hash seed that differs between processes
        script = tmp_path / "job.py"
        script.write_text(
            "import sys\n"
            "from lair.records import Cacher\n"
            "def f(x):\n"
            "    print('ran')\n"
            "    return x in {'north', 'south', 'east', 'west', 'up', 'down'}\n"
            "print(Cacher(f, sys.argv[1])('up'))\n"
        )
        cache_file = str(tmp_path / "cache.pkl")

        def run(seed):
            env = dict(os.environ, PYTHONHASHSEED=str(seed))
            out = subprocess.run(
                [sys.executable, str(script), cache_file],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            return out.stdout.split()

        assert run(1) == ["ran", "True"]
        assert run(2) == ["True"]
        assert run(3) == ["True"]


class Doubler:
    """An instance that pickles, for a method passed to a Cacher."""

    def double(self, x):
        return x * 2


class TestCacherCallables:
    """Callables with no plain code object are still cached."""

    def cached(self, tmp_path, func, *args, **kwargs):
        return records.Cacher(func, str(tmp_path / "cache.pkl"))(*args, **kwargs)

    def test_lambda_builtin_and_class(self, tmp_path):
        assert self.cached(tmp_path, lambda x: x + 1, 1) == 2
        assert self.cached(tmp_path, len, "abc") == 3
        assert self.cached(tmp_path, str, 5) == "5"

    def test_partial_hits_and_depends_on_its_arguments(self, tmp_path):
        calls = []

        def power(x, p):
            calls.append(p)
            return x**p

        cache_file = str(tmp_path / "cache.pkl")
        square = functools.partial(power, p=2)
        assert records.Cacher(square, cache_file)(3) == 9
        assert records.Cacher(square, cache_file)(3) == 9  # also logs without __name__
        assert records.Cacher(functools.partial(power, p=3), cache_file)(3) == 27
        assert calls == [2, 3]

    def test_bound_method(self, tmp_path):
        class Scaler:
            def __init__(self, k):
                self.k = k
                self.calls = 0

            def scale(self, x):
                self.calls += 1
                return x * self.k

        scaler = Scaler(5)
        cache_file = str(tmp_path / "cache.pkl")
        assert records.Cacher(scaler.scale, cache_file)(2) == 10
        assert records.Cacher(scaler.scale, cache_file)(2) == 10
        assert scaler.calls == 1

    def test_bound_method_in_the_arguments(self, tmp_path):
        cached = records.Cacher(lambda fn, x: fn(x), str(tmp_path / "cache.pkl"))
        assert cached(Doubler().double, 4) == 8
        assert cached(Doubler().double, 4) == 8

    def test_callable_object_with_unpicklable_state(self, tmp_path):
        class Doubler:
            def __init__(self):
                self.lock = threading.Lock()  # cannot be pickled
                self.calls = 0

            def __call__(self, x):
                self.calls += 1
                return 2 * x

        doubler = Doubler()
        cache_file = str(tmp_path / "cache.pkl")
        assert records.Cacher(doubler, cache_file)(4) == 8
        assert records.Cacher(doubler, cache_file)(4) == 8
        assert doubler.calls == 1

    def test_function_with_unpicklable_closure_and_default(self, tmp_path):
        lock = threading.Lock()

        def f(x, guard=lock):
            with lock:
                return x

        assert self.cached(tmp_path, f, 1) == 1

    def test_closure_variable_that_changes_does_not_change_the_key(self, tmp_path):
        calls = {"n": 0}  # a dict is only its type in the key

        def f(x):
            calls["n"] += 1
            return x

        cache_file = str(tmp_path / "cache.pkl")
        assert records.Cacher(f, cache_file)(1) == 1
        assert records.Cacher(f, cache_file)(1) == 1
        assert calls["n"] == 1

    def test_closure_number_is_part_of_the_key(self, tmp_path):
        def make(k):
            def f(x):
                return x * k

            return f

        cache_file = str(tmp_path / "cache.pkl")
        assert records.Cacher(make(2), cache_file)(3) == 6
        assert records.Cacher(make(5), cache_file)(3) == 15


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
    # fastkml < 1.0 has a features() method, >= 1.0 a features list
    features = k.features() if callable(k.features) else k.features
    (document,) = list(features)
    assert document.name == "t"


def test_read_kml_with_fastkml_before_1_0(tmp_path, monkeypatch):
    """
    Before 1.0, fastkml has no ``KML.parse``: ``from_string`` fills a new instance.

    The installed fastkml is replaced by a stand-in with the old interface, which
    records the bytes it is given.
    """
    kml_module = pytest.importorskip("fastkml.kml")
    text = '<kml xmlns="http://www.opengis.net/kml/2.2"><name>Cañón</name></kml>'

    class OldKML:
        def from_string(self, data):
            self.data = data

    monkeypatch.setattr(kml_module, "KML", OldKML)
    kml_path = tmp_path / "old.kml"
    kml_path.write_text(text, encoding="utf-8")
    k = records.read_kml(str(kml_path))
    assert isinstance(k, OldKML)
    assert k.data == text.encode("utf-8")


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

    def _fake_wget(self, monkeypatch, fail=()):
        """Stub subprocess.run: wget writes a file, unzip is recorded."""
        import subprocess

        calls = []

        def run(cmd, check):
            calls.append(cmd)
            if cmd[0] == "wget":
                if cmd[3] in fail:
                    raise subprocess.CalledProcessError(8, cmd)
                with open(cmd[2], "w") as f:
                    f.write("downloaded")

        monkeypatch.setattr(subprocess, "run", run)
        return calls

    def test_zip_is_unzipped_and_removed(self, tmp_path, monkeypatch):
        calls = self._fake_wget(monkeypatch)
        records.wget_download("https://example.com/pub/a.zip", str(tmp_path))
        zip_path = str(tmp_path / "a.zip")
        assert calls == [
            ["wget", "-O", zip_path, "https://example.com/pub/a.zip"],
            ["unzip", "-d", str(tmp_path), zip_path],
        ]
        assert not os.path.exists(zip_path)

    def test_unzip_false_keeps_the_zip(self, tmp_path, monkeypatch):
        calls = self._fake_wget(monkeypatch)
        records.wget_download(
            "https://example.com/pub/a.zip", str(tmp_path), unzip=False
        )
        assert [c[0] for c in calls] == ["wget"]
        assert (tmp_path / "a.zip").read_text() == "downloaded"

    def test_failed_download_is_skipped(self, tmp_path, monkeypatch, caplog):
        bad = "https://example.com/pub/bad.zip"
        good = "https://example.com/pub/good.csv"
        calls = self._fake_wget(monkeypatch, fail={bad})
        records.wget_download([bad, good], str(tmp_path))
        # No unzip of the failed zip; the other file still downloads
        assert [c[0] for c in calls] == ["wget", "wget"]
        assert (tmp_path / "good.csv").exists()
        assert "Failed to download" in caplog.text

    def test_empty_prefix_recreates_remote_tree(self, tmp_path, monkeypatch):
        self._fake_wget(monkeypatch)
        records.wget_download(
            "https://example.com/pub/data/2024/f.csv", str(tmp_path), prefix=""
        )
        assert (tmp_path / "pub" / "data" / "2024" / "f.csv").exists()


class _FakeFTP:
    """
    In-memory stand-in for ``ftplib.FTP``.

    ``tree`` maps absolute remote paths to bytes (files) or None (directories,
    whose children are the paths directly beneath them). cwd() into a file or a
    missing path raises error_perm 550, like a real server.
    """

    instances: list = []

    def __init__(self, tree, host):
        self.tree = {"/": None, **tree}
        self.host = host
        self.cwd_path = "/"
        self.logged_in = None
        self.quit_called = False
        self.closed = False
        self.retrieved = []
        _FakeFTP.instances.append(self)

    def login(self, user, passwd):
        self.logged_in = (user, passwd)

    def cwd(self, path):
        import ftplib

        path = "/" + path.strip("/") if path != "/" else "/"
        if self.tree.get(path, b"") is not None:  # a file, or missing
            raise ftplib.error_perm(f"550 {path}: No such directory")
        self.cwd_path = path

    def nlst(self):
        here = self.cwd_path.rstrip("/")
        return sorted(
            p.rsplit("/", 1)[1]
            for p in self.tree
            if p != "/" and p.rsplit("/", 1)[0] == here
        )

    def retrbinary(self, cmd, callback):
        verb, path = cmd.split(" ", 1)
        assert verb == "RETR"
        self.retrieved.append(path)
        callback(self.tree[path])

    def quit(self):
        self.quit_called = True

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # As ftplib.FTP: send QUIT, ignore a server that does not answer, always close.
        try:
            self.quit()
        except (OSError, EOFError):
            pass
        finally:
            self.close()


class TestFtpDownload:
    TREE = {
        "/pub": None,
        "/pub/data": None,
        "/pub/data/2015": None,
        "/pub/data/2015/a_2015-06.nc": b"a",
        "/pub/data/2015/b_2015-07.nc": b"b",
        "/pub/data/2016": None,
        "/pub/data/2016/c_2016-06.nc": b"c",
        "/pub/data/readme.txt": b"r",
    }

    @pytest.fixture
    def ftp(self, monkeypatch):
        import ftplib

        _FakeFTP.instances = []
        monkeypatch.setattr(
            ftplib, "FTP", lambda host: _FakeFTP(self.TREE, host), raising=True
        )
        return _FakeFTP.instances

    @staticmethod
    def _local(root):
        return {
            p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob("*")
            if p.is_file()
        }

    def test_directory_is_mirrored_under_its_name(self, tmp_path, ftp):
        assert records.ftp_download("ftp.example.com", "/pub/data", str(tmp_path))
        assert self._local(tmp_path) == {
            "data/2015/a_2015-06.nc": b"a",
            "data/2015/b_2015-07.nc": b"b",
            "data/2016/c_2016-06.nc": b"c",
            "data/readme.txt": b"r",
        }
        (conn,) = ftp
        assert conn.host == "ftp.example.com"
        assert conn.logged_in == ("anonymous", "anonymous@")
        assert conn.quit_called

    def test_credentials_are_passed_through(self, tmp_path, ftp):
        records.ftp_download(
            "h", "pub/data/readme.txt", str(tmp_path), username="me", password="pw"
        )
        assert ftp[0].logged_in == ("me", "pw")

    def test_single_file_lands_in_download_dir(self, tmp_path, ftp):
        # No leading slash on the remote path is fine: it is taken from root
        records.ftp_download("h", "pub/data/readme.txt", str(tmp_path))
        assert self._local(tmp_path) == {"readme.txt": b"r"}

    def test_prefix_strips_the_common_part(self, tmp_path, ftp):
        records.ftp_download("h", "/pub/data/2015", str(tmp_path), prefix="/pub/data")
        assert set(self._local(tmp_path)) == {
            "2015/a_2015-06.nc",
            "2015/b_2015-07.nc",
        }

    def test_empty_prefix_recreates_the_remote_tree(self, tmp_path, ftp):
        records.ftp_download("h", "/pub/data/2016", str(tmp_path), prefix="")
        assert set(self._local(tmp_path)) == {"pub/data/2016/c_2016-06.nc"}

    def test_pattern_filters_files_not_directories(self, tmp_path, ftp):
        # The directories don't match '*-06*' but are still descended into
        records.ftp_download("h", "/pub/data", str(tmp_path), pattern="*-06*")
        assert set(self._local(tmp_path)) == {
            "data/2015/a_2015-06.nc",
            "data/2016/c_2016-06.nc",
        }
        assert sorted(ftp[0].retrieved) == [
            "/pub/data/2015/a_2015-06.nc",
            "/pub/data/2016/c_2016-06.nc",
        ]

    def test_several_paths_one_connection(self, tmp_path, ftp):
        records.ftp_download(
            "h", ["/pub/data/2015", "/pub/data/2016"], str(tmp_path), prefix="/pub"
        )
        assert set(self._local(tmp_path)) == {
            "data/2015/a_2015-06.nc",
            "data/2015/b_2015-07.nc",
            "data/2016/c_2016-06.nc",
        }
        assert len(ftp) == 1

    def test_other_permission_errors_are_raised(self, tmp_path, monkeypatch):
        import ftplib

        class _Denied(_FakeFTP):
            def cwd(self, path):
                if path != "/":
                    raise ftplib.error_perm("530 Login incorrect.")

        _FakeFTP.instances = []
        monkeypatch.setattr(ftplib, "FTP", lambda host: _Denied(self.TREE, host))
        with pytest.raises(ftplib.error_perm, match="530"):
            records.ftp_download("h", "/pub/data", str(tmp_path))
        assert self._local(tmp_path) == {}
        assert _FakeFTP.instances[-1].closed  # not left open for the GC to warn about

    def test_the_connection_is_closed(self, tmp_path, ftp):
        records.ftp_download("h", "/pub/data/readme.txt", str(tmp_path))
        assert ftp[0].quit_called
        assert ftp[0].closed

    def test_a_quit_the_server_does_not_answer_still_closes(
        self, tmp_path, monkeypatch
    ):
        import ftplib

        class _Dropped(_FakeFTP):
            def quit(self):
                raise EOFError  # the server already closed the control connection

        _FakeFTP.instances = []
        monkeypatch.setattr(ftplib, "FTP", lambda host: _Dropped(self.TREE, host))
        assert records.ftp_download("h", "/pub/data/readme.txt", str(tmp_path))
        assert self._local(tmp_path) == {"readme.txt": b"r"}
        assert _FakeFTP.instances[-1].closed


class TestPathMatches:
    def test_plain_string_is_substring(self):
        assert records._path_matches("/ct/2015/06/CT.molefrac_2015-06-01.nc", "2015-06")
        assert not records._path_matches(
            "/ct/2016/01/CT.molefrac_2016-01-01.nc", "2015"
        )

    def test_wildcards_glob_the_full_path(self):
        assert records._path_matches(
            "/ct/2015/06/CT.molefrac_2015-06-01.nc", "*2015-06*.nc"
        )
        assert not records._path_matches(
            "/ct/2015/06/CT.molefrac_2015-06-01.nc", "*.txt"
        )
