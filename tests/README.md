# lair tests

Tests are **containerized by module**. Every `lair` submodule has a matching
self-contained directory here:

```
tests/
├── conftest.py            # shared fixtures only — keep minimal
├── test_pkg.py            # package-level smoke tests
├── clock/                 # tests for lair.clock
│   ├── __init__.py
│   ├── conftest.py        # clock-only fixtures (optional)
│   ├── test_clock.py
│   └── data/              # clock-only sample data (optional)
├── meteorology/
│   └── test_meteorology.py
└── ...
```

**Why:** modules regularly graduate out of `lair` into their own packages. When
that happens you should be able to `git mv lair/<module>.py` and
`git mv tests/<module>/` into the new package and be done — without untangling a
shared `test_*.py` or a shared `conftest.py`.

## Rules

1. **One directory per module.** `lair.foo` → `tests/foo/`.
2. **Module-local fixtures/data live in the module's directory**, not in the
   top-level `tests/conftest.py`. The top-level conftest is reserved for things
   that are genuinely cross-cutting (and even those are a smell).
3. **No cross-module imports between test directories.** A test in
   `tests/clock/` must not import from `tests/meteorology/`. If two modules need
   the same fixture, that's a hint the fixture (or the code) belongs in a
   shared place — raise it rather than coupling the test dirs.
4. **Mark tests that aren't pure/offline** so the default run stays fast and
   hermetic (see markers below).

## Extracting a module into its own package

```bash
git mv lair/foo.py        <new-pkg>/src/foo/...
git mv tests/foo/         <new-pkg>/tests/
```

Then drop the now-redundant nesting in the new package (`tests/foo/test_foo.py`
→ `tests/test_foo.py`) and fix the import paths (`lair.foo` → `foo`). Because
nothing else referenced `tests/foo/`, the rest of the lair suite still passes.

## Markers

Defined in `pyproject.toml` (`[tool.pytest.ini_options]`):

| Marker     | Meaning                                                              |
|------------|---------------------------------------------------------------------|
| `network`  | needs live network (NOAA GML/FTP, AWS S3, MesoWest, soundings)       |
| `slow`     | expensive runtime / bandwidth / disk; excluded from the default run |
| `chpc`     | needs CHPC filesystem paths or group data; not runnable off-cluster |

```bash
just test                # everything except slow
uv run pytest -m "not network and not slow and not chpc"   # hermetic subset (CI default)
uv run pytest tests/clock                                  # just one module
```

## Optional dependencies

Modules behind optional extras (`geo`, `formats`, `regridding`, `requests`,
`science`) use `pytest.importorskip(...)` so their tests **skip** cleanly when
the extra isn't installed. The lean `uv` dev group does not install those
extras; for full coverage run the suite in the conda `lair-dev` env
(see `env-dev.yml`).

## Offline import

`import lair` will FTP-download the NOAA GML CCG filter on first import if
`lair/_ccg_filter.py` is missing. `tests/conftest.py` sets
`LAIR_SKIP_CCG_DOWNLOAD=1` so the suite never does network I/O at import time;
`lair.background` is then only importable when the file is already present.
