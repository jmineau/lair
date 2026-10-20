# AGENTS.md

Orientation for contributors and coding agents working on lair. Contribution
mechanics (setup, checks, the pixi environment, writing docs examples,
releasing) are in [CONTRIBUTING.md](CONTRIBUTING.md); the test layout is in
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

Flat (`lair/`, not `src/lair/`), one module per topic, no subpackages:

```
__init__.py      pint setup and the CCG filter download (see "Import side effects")
_ccg_filter.py   NOAA's CCG filter, downloaded on first import; gitignored, never commit
_optional.py     import_optional_dependency()
config.py        get_data_dir(), get_cache_dir(); no file-system work
constants.py     physical constants (pint has no constants support; issue linked inside)
hrrr.py          HRRR winds at a point, from the HRRR-Zarr archive on AWS
inventories.py   Inventory and its loaders: EDGAR, EPA, GFEI, Vulcan, WetCHARTs
meteorology.py   ideal gas law, hypsometric, poisson, vapor pressure, ... (plain SI)
noaa.py          CarbonTracker and NOAA GML readers
records.py       file utilities (ftp_download, unzip, read_kml, Cacher)
soundings.py     upper-air soundings from the Wyoming archive (siphon)
synoptic.py      Synoptic Data API (MesoWest's successor); token= or $SYNOPTIC_TOKEN
transects.py     mobile transect matrices: metrics and the builder
```

and `air`, `background`, `clock`, `dev`, `geo`, `parallel`, `pcaps`,
`plotter`, `pollutants` and `utils`, each named for what it holds. Instrument
and archive specifics stay out: pollutant column naming is in
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
   `[mass]/[area]/[time]` through a molecular weight `mw`, a quantity in g/mol;
   without one the conversion raises a `TypeError` that names `mw`). The
   registry is process-global, so importing lair changes pint for other code too.
3. Sets pint's default sort to `sort_by_dimensionality` (`kg/km**2/d`). Not
   cosmetic: `Inventory.get_units` splits a flux unit on `/` and reads
   quantity, area and time in that order. pint >= 0.26 exports the function;
   0.24 and 0.25 (the newest on Python 3.11) have it only privately, so lair
   tries both, and pyrefly flags whichever the installed pint lacks (each line
   carries its suppression). If neither exists, `import lair` raises
   `ImportError` on purpose. `pint>=0.24.4`: 0.24.0 to 0.24.3 do not import
   with flexparser 0.4. `tests/test_pkg.py` checks both.
4. Calls `setup_ccg_filter()`. If `lair/_ccg_filter.py` is absent, it
   downloads `ccg_filter.zip` from `ftp.gml.noaa.gov` into a temporary
   `lair/.ccg_filter-*` folder, takes only `ccg_filter.py`, and moves it into
   place with an atomic `os.replace`, so concurrent first imports are safe and
   a failure leaves no partial file. At import a failure (or a read-only lair
   folder) only logs a WARNING; `lair.background` then raises an `ImportError`
   saying to run `lair.setup_ccg_filter()`, which raises when called.

Consequences:

- **`LAIR_SKIP_CCG_DOWNLOAD`** skips the download; any non-empty value skips,
  `0` included. `tests/conftest.py` and the Tests, Code Quality and
  Documentation workflows set it. The Tests workflow's `pixi` job clears it
  for one step to download the filter (cached by month), so `tests/background`
  runs there; it is the only CI job that reaches NOAA.
- **`lair.background` imports `_ccg_filter` at the top**, so even
  `rolling_baseline` (pandas only) needs the downloaded file and scipy.
- **The cache folder is made on first use, never at import.**
  `config.get_cache_dir()` reads `LAIR_CACHE_DIR` at each call (no
  module-level constant). `lair.noaa._cache_file` makes the folder when it
  first writes, and raises an `OSError` naming the folder and the variable
  when it cannot; a new cache user calls it. `tests/config` and `tests/noaa`
  check a read-only home.
- **No global xarray options.** Code that needs attributes kept scopes it
  (`@_keep_attrs` in `inventories`, or `with xr.set_options(keep_attrs=True):`).
  The locked xarray keeps attributes by default, so `TestAttributes` passes
  either way; check a change to the scoping on xarray < 2025.11.

## Commands

uv manages the environment (`uv sync`, lean: no extras); every task is a
`just` recipe (`uv run just --list`), and CI runs the same recipes.
`just quality-check` runs `lint`, `type-check` (pyrefly), `docstr` (at least
88%) and `test` (parallel, skipping `network` and `slow`; extra args go to
pytest). `just format` fixes lint and formatting, `just build-docs` builds
the docs and runs every example (warnings are errors), `just docs-serve`
previews them, and `just pre-commit` runs every hook.

Tests of modules behind a missing extra `importorskip` and skip. The pixi
environment has every extra, including xesmf on conda-forge's no-MPI ESMF
(64-bit Linux): `pixi install`, then `pixi run pytest ...` or
`pixi run pyrefly check` (the recipes use uv). After editing dependencies, run
`uv lock` **and** `pixi lock` and commit both. They are solved separately, so
a failure in one environment only is usually a version difference.
Dependabot updates `uv.lock` only; CI fails on a stale `pixi.lock`, which is
`merge=binary`: after a conflict, rerun `pixi lock`.

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
  with a reason, not in config and not with a blanket `Any`. The config
  disables only `bad-override-mutable-attribute` and types the optional extras
  as `Any`, so both environments agree. `meteorology` uses a documented
  `Numeric = Any`; `geo` and `inventories` use the bound TypeVar `_XarrayT`.
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
- **Time:** a naive time is UTC. `convert_timezones` raises when a naive time
  has no `fromtz`, and, like `tz_localize`, on an ambiguous or nonexistent DST
  time unless `ambiguous=`/`nonexistent=` is given. A time formatted into a
  request, an S3 key or a file name is converted to UTC first, since
  `strftime` writes a tz-aware time in its own zone (`soundings._naive_utc`,
  `hrrr.create_s3_group_url`).
- **Attribution:** code adapted from Brian Blaylock's Carpenter Workshop
  (credited in the README and docs), NOAA GML's CCG filter, and StackOverflow
  (linked in docstrings) keeps its credit when refactored.

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
- **Areas:** `gridcell_area` on a lat-lon grid is the exact WGS84 ellipsoid
  area in closed form (the authalic-latitude `q`), with no ESMF, so
  `absolute_emissions` and `integrate()` need none either. A sphere (`R=`) is
  0.1-0.9% off depending on latitude; `tests/geo` checks against pyproj's
  geodesic areas and the ellipsoid's total.
- **Opening many files:** call the module's `_open_mfdataset` (`inventories`,
  `noaa`), not `xr.open_mfdataset`: it names today's `compat`, `data_vars`,
  `coords` and `join`, which xarray is changing. In `noaa` it runs through
  `records.Cacher`, keyed on the pickled call arguments, so changing what is
  passed orphans every existing cache. Never open NetCDF files on threads
  (dask's default for `parallel=True`): netCDF-C and HDF5 are not thread-safe
  and xarray reads metadata outside its lock, so the process crashes. `noaa`'s
  `parallel=True` uses worker processes, which need a `__main__` guard in scripts.
- **`thoning_filter`'s `sampleinterval`** is in days: the step of the even grid
  NOAA's filter interpolates onto before its FFT. lair passes the median step
  only when NOAA's own estimate would divide by zero. A grid coarser than the
  data samples it rather than averaging, which biases hourly data with a
  diurnal cycle; test a change against the daily curve.
- **The `requests` extra floors `s3fs`, not boto3** (reason in
  `pyproject.toml`).

## Docs

- They build in the lean environment: `docs/conf.py` mocks every optional
  extra (`autodoc_mock_imports`); add new optional imports there. Mocked types
  can't be combined at import (`Polygon | None` raises), so a module whose
  annotations name optional types needs `from __future__ import annotations`.
- `docs/api.rst` lists modules by hand; add new ones. Each class and member
  gets its own page (`docs/_templates/autosummary/`; `docs/_ext/api_pages.py`
  drops a class docstring's `Methods` section and the `Attributes` entries
  that have a row).
- Examples (`.. plot::` in docstrings, `.. ipython:: python` in pages) run
  when the docs build and can use only the core dependencies; a page that
  needs more shows recorded output. CONTRIBUTING.md, "Docs examples", has how.

## Docs pages

Each page answers one question, the pages are listed in the order a new reader
asks them, and each fact is written on one page: the others link to it
(`:ref:`, `:doc:`) and do not restate it. A docs PR adds an example (one job,
titled as the job, opening with the rules that come out of it) or text in a
docstring, which shows on the API pages; it adds no "guide" page. A new example
goes in `docs/examples/index.rst` only (a toctree entry and a line describing
it); the quickstart links to the Examples page, not to each example. A new fact
goes on the page that holds its kind:

| Fact | Its page | The others |
|---|---|---|
| Install routes, extras, the first-import download, CHPC | front page, "Installation" | link by `installation` or `optional-dependencies`; the README keeps a short copy, as GitHub's front page |
| `LAIR_*` variables, what needs which data, the pollutant spelling, the cache | `data`, "Data locations" | link by `data-locations` |
| A dataset's folder layout and where to get it | its example, "The data"; no example: its class docstring | "Data locations" links to it |
| A time without a zone is UTC; time zones; seasons | `examples/clock` | a clause and a link |
| Units; the `mass_flux` context | `examples/units` | link |
| Plain SI in and out of `meteorology` | quickstart, "A first calculation" | link |
| Inventory totals; conservative against bilinear regridding. Drawing a map: the time step, state lines, ticks, grid, inset | `examples/inventory`; for the map, `examples/maps` | link |
| `get_soundings` heights and downloads | `examples/soundings` | link |
| CarbonTracker sampling and background | `examples/carbontracker` | link |
| Background methods, side by side, and which one to use | `examples/background` | link |
| Function and class details | the API pages, from docstrings | `:func:` and `:class:` links, not `autofunction` |

## Tests

One directory per module (`lair.clock` -> `tests/clock/`), with no imports
between them, so a module can leave lair with two `git mv`s. Network code is
tested against fakes, never live. Warnings are errors: assert an expected one
with `pytest.warns`, and ignore one from another library by message in
`[tool.pytest]` `filterwarnings`, with a comment naming the library and
version. Tests run in parallel, so they must not share state. Details, the
markers and the fakes: `tests/README.md`.

## Commits, changelog, releases

- Conventional Commits with the module as scope (`fix(soundings): ...`), `!`
  for a breaking change. User-visible changes go under `## [Unreleased]` in
  `CHANGELOG.md`; `just changelog` drafts entries.
- The version comes from git tags (setuptools-scm). Releases are CalVer tags
  `vYYYY.MM.PATCH`, MM = 05, 08 or 12, made by `just release`; the Publish
  workflow creates the GitHub Release and Zenodo archives it (CONTRIBUTING.md).
  **Do not cut a release, tag, or push to `main` unless the maintainer asks.**

## Template

Tooling comes from [jmineau/python-template](https://github.com/jmineau/python-template)
via copier (`.copier-answers.yml`, never edited by hand); `copier update` pulls
in template changes. Tooling changes meant for every package belong there.
