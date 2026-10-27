# AGENTS.md

Orientation for contributors and coding agents working on lair. Contribution
mechanics (setup, checks, the pixi environment, writing docs examples,
releasing), the tour of the modules, and where each fact goes in the docs are
in [CONTRIBUTING.md](CONTRIBUTING.md); the test layout is in
[tests/README.md](tests/README.md).

> **Keep this file current.** If you change the module layout, the build/test
> commands, or learn a new invariant or gotcha, update the matching section in
> the same change. A section that no longer matches the code is worse than no
> section: fix it or delete it.
>
> Personal or machine-specific notes (local paths, cluster setup) belong in an
> untracked file, not here. Anything matching `*.local.md` is gitignored for
> this (e.g. `CLAUDE.local.md`); add your tool's local files to `.gitignore` if
> they are not covered. Some agents stop reading `AGENTS.md` once a local
> instruction file exists, so import or reference it from yours.

## What this is

`lair` (Land-Air Interactions Research) is a toolkit of atmospheric-science
utilities: "a standard library for this research", not a framework. It is
marked Alpha: modules and classes still move, so don't treat a symbol as
stable for outside users without asking the maintainer. Distribution and
import name `lair`, installed from GitHub (the `lair` on PyPI is an unrelated
project). Docs: <https://jmineau.github.io/lair/>.

## Layout

Flat (`lair/`, not `src/lair/`), one module per topic, no subpackages; the
tour of the modules is in CONTRIBUTING.md, "Code layout". `_ccg_filter.py` is
NOAA's CCG filter, downloaded on first import; it is gitignored, never commit
it. Instrument and archive specifics stay out: pollutant column naming is in
`uataq.pollutants`, and readers of one site's archive or one platform (such as
TRAX transects) belong in the downstream packages. `lair` re-exports only
`units` (pint-xarray's registry), `config`, `ftp_download`, `unzip`,
`setup_ccg_filter` and `__version__`; everything else is imported from its
module (`from lair.meteorology import poisson`).

## Import side effects

`lair/__init__.py` does real work at import:

1. Imports `cf_xarray.units` **before** `pint_xarray`. The order is
   load-bearing.
2. Adds the pint context `mass_flux` (`[substance]/[area]/[time]` <->
   `[mass]/[area]/[time]` through a molecular weight `mw`, a quantity in g/mol).
   The registry is process-global, so importing lair changes pint for other
   code too.
3. Sets pint's default sort to `sort_by_dimensionality` (`kg/km**2/d`). Not
   cosmetic: `Inventory.get_units` splits a flux unit on `/` and reads
   quantity, area and time in that order. If pint lacks the function, `import
   lair` raises `ImportError` on purpose. Which pint versions have it, and why
   the floor is `pint>=0.24.4`, are in the comments in `lair/__init__.py` and
   `pyproject.toml`; `tests/test_pkg.py` checks both imports.
4. Calls `setup_ccg_filter()`. If `lair/_ccg_filter.py` is absent, it
   downloads NOAA's `ccg_filter.zip` and installs the file atomically (the
   docstring has the steps). At import a failure (or a read-only lair
   folder) only logs a WARNING; `lair.background` then raises an `ImportError`
   saying to run `lair.setup_ccg_filter()`, which raises when called.

Consequences:

- **`LAIR_SKIP_CCG_DOWNLOAD`** skips the download; any non-empty value skips,
  `0` included. `tests/conftest.py` and the Tests, Code Quality and
  Documentation workflows set it (CONTRIBUTING.md, "Setup", names the one CI
  job that clears it).
- **The cache folder is made on first use, never at import.**
  `config.get_cache_dir()` reads `LAIR_CACHE_DIR` at each call (no
  module-level constant). `lair.noaa._cache_file` makes the folder when it
  first writes; a new cache user calls it. `tests/config` and `tests/noaa`
  check a read-only home.
- **No global xarray options.** Code that needs attributes kept scopes it
  (`@_keep_attrs` in `inventories`, or `with xr.set_options(keep_attrs=True):`).
  The locked xarray keeps attributes by default, so `TestAttributes` passes
  either way; check a change to the scoping on xarray < 2025.11.

## Commands

uv manages the environment (`uv sync`, lean: no extras); every task is a
`just` recipe (`uv run just --list`; CONTRIBUTING.md, "Checks"), and CI runs the
same recipes. `just quality-check` runs `lint`, `type-check` (pyrefly), `docstr`
(at least 88%) and `test` (parallel, skipping `network` and `slow`). Tests of
modules behind a missing extra `importorskip` and skip. The pixi
environment has every extra (CONTRIBUTING.md, "Setup"): `pixi run pytest ...`
or `pixi run pyrefly check` (the recipes use uv). After editing dependencies,
run `uv lock` **and** `pixi lock` and commit both. A failure in one
environment only is usually a version difference. CI fails on a stale
`pixi.lock`, which is `merge=binary`: after a conflict, rerun `pixi lock`.

## Conventions

- Python 3.11+. Ruff rules `E, F, UP, B, SIM, I, D, D213, NPY, RUF100`
  (NumPy-convention docstrings, summary on the line after the opening quotes);
  `lair/` and `tests/` are `ruff format`-ed (the pre-commit hook formats on
  commit). After a reflow, put `# pyrefly: ignore[<kind>]` on its own line
  *above* the flagged line, and `# noqa` on the first line of a multi-line
  statement.
- **Typing:** pyrefly reports 0 errors (CI and a pre-commit hook), checked as
  Python 3.11 in the lean environment; `lair/py.typed` ships. Where a
  library's stubs are wrong (pandas overloads, pint callbacks), suppress inline
  with a reason, not in config and not with a blanket `Any` (`[tool.pyrefly]`
  in `pyproject.toml` says what the config does).
- **Docstrings:** each public attribute and property gets its own API page, so
  give each one a docstring (one without shows an empty row), with its See
  Also and Examples.
- **Logging:** `logger = logging.getLogger(__name__)` per module. Library code
  never adds handlers (not even a `NullHandler`, which hides warnings from
  Python's last-resort handler), sets levels, or prints (`utils.updating_print`
  aside). Progress is `INFO`, per-item detail `DEBUG`, recoverable problems
  `WARNING`; %-style arguments, not f-strings.
- **No built-in data paths.** A reader takes an explicit directory and falls
  back to a `LAIR_*` variable through `config.get_data_dir(env_var, path)`
  (a `ValueError` names the variable), resolved at call time, never at import.
  The docs' "Data locations" lists the variables.
- **Units:** SI is the baseline; prefer pint-aware code where it exists.
  `meteorology` is plain SI in and out (`_si_inputs` converts pint inputs;
  results are never Quantities); callers that want units convert at their own
  boundary, as `pcaps.valleyheatdeficit` does.
- **pandas 3 only** (`pandas>=3.0`): no branches, `try/except` or tests for
  pandas 2. `'1h'` (`'1H'` raises), `'D'`, `sep=r'\s+'`, no chained assignment.
- **Time:** a naive time is UTC (`examples/clock` has the rest). A time
  formatted into a request, an S3 key or a file name is converted to UTC first,
  since `strftime` writes a tz-aware time in its own zone
  (`soundings._naive_utc`, `hrrr.create_s3_group_url`).
- **Attribution:** code adapted from Brian Blaylock's Carpenter Workshop
  (credited in the README and docs), NOAA GML's CCG filter, and StackOverflow
  (linked in docstrings) keeps its credit when refactored.
- **Tooling** comes from [jmineau/python-template](https://github.com/jmineau/python-template)
  via copier (`.copier-answers.yml`, never edited by hand); tooling changes
  meant for every package belong there.

## Module gotchas

- **Inventories:** every loader labels a time step by the *start* of its
  period, from CF `time_bnds` when the file has them (`_label_period_starts`).
  NaN "no emission" cells become 0 before any conservative regrid, or every
  coarse cell touching one turns NaN. Without `src_units`, each variable is
  quantified with its own `units` attribute. `tests/inventories` checks every
  loader (`INVENTORY_CASES`) on tiny synthetic archives shaped like the real
  files; add new loaders there.
- **ESMF:** regridding (`geo.regrid`, `geo.resample` and the grids'
  `regrid`/`resample`) needs xesmf, and xesmf needs conda-forge's
  `esmf=*=nompi*` (pip installs xesmf without ESMF; the MPICH build can kill
  the process, pytest included, when MPI fails to start).
- **Opening many files:** call the module's `_open_mfdataset` (`inventories`,
  `noaa`), not `xr.open_mfdataset`: it names today's `compat`, `data_vars`,
  `coords` and `join`, which xarray is changing. In `noaa` it runs through
  `records.Cacher`, keyed on the pickled call arguments, so changing what is
  passed orphans every existing cache. Never open NetCDF files on threads
  (dask's default for `parallel=True`): the process crashes. `noaa`'s
  `parallel=True` uses worker processes, which need a `__main__` guard in
  scripts (the docstring says why).
- **CarbonTracker times** and **`thoning_filter`'s `sampleinterval`:**
  CONTRIBUTING.md, "Notes on the code".
- **An optional package** is imported with `import_optional_dependency("pkg")`
  and listed in `_optional._EXTRA_OF` under its extra, which the missing-package
  `ImportError` names with the install command; `tests/optional` checks the
  table (the comment in `_optional.py` says how).

## Docs

- They build in the lean environment: `docs/conf.py` mocks every optional
  extra (`autodoc_mock_imports`); add new optional imports there. Mocked types
  can't be combined at import (`Polygon | None` raises), so a module whose
  annotations name optional types needs `from __future__ import annotations`.
- `docs/api.rst` lists modules by hand; add new ones. Each class and member
  gets its own page (`docs/_templates/autosummary/`; `docs/_ext/api_pages.py`
  drops a class docstring's `Methods` section and the `Attributes` entries
  that have a row).
- Each page answers one question, the pages are listed in the order a new reader
  asks them, and each fact is written on one page: the others link to it
  (`:ref:`, `:doc:`) and do not restate it. A docs PR adds an example (one job,
  titled as the job, opening with the rules that come out of it) or text in a
  docstring, which shows on the API pages; it adds no "guide" page. Where a new
  example and a new fact go, with the table: CONTRIBUTING.md, "Where a fact goes
  in the docs".

## Tests

One directory per module (`lair.clock` -> `tests/clock/`), with no imports
between them. Network code is tested against fakes, never live. Warnings are
errors: assert an expected one with `pytest.warns`, and ignore one from another
library by message in `[tool.pytest]` `filterwarnings`, with a comment naming
the library and version. Tests run in parallel, so they must not share state.
Details, the markers and the fakes: `tests/README.md`. Docstring examples
(`>>>` in `lair/`) run as tests (`--doctest-modules`); `# doctest: +SKIP` marks
one that needs data CI lacks. The lean environment skips a module whose extra
is missing; the pixi job runs it.

## Commits, changelog, releases

- Conventional Commits with the module as scope (`fix(soundings): ...`), `!`
  for a breaking change. A pull request leaves `CHANGELOG.md` alone: at release
  `just changelog` drafts the entries from the commit messages, so write each
  as the line a user would read.
- The version comes from git tags (setuptools-scm). The maintainer cuts the
  releases (CONTRIBUTING.md).
