# AGENTS.md

Orientation for contributors and coding agents working on lair. Contribution
mechanics (setup, checks, the pixi environment, docs pages with recorded
output, releasing) are in [CONTRIBUTING.md](CONTRIBUTING.md); the test layout
is in [tests/README.md](tests/README.md).

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
project). Flat layout (`lair/`, not `src/lair/`), no subpackages. Docs:
<https://jmineau.github.io/lair/>.

## Layout

```
lair/
  __init__.py      pint setup and the CCG filter download (see "Import side effects")
  _ccg_filter.py   NOAA's CCG filter, downloaded on first import; gitignored, never commit
  _optional.py     import_optional_dependency()
  config.py        get_data_dir(), get_cache_dir(); no file-system work
  constants.py     physical constants (pint has no constants support yet)
  air.py           bin_polar, wind_components, wind_direction (NaN for a zero vector)
  background.py    background concentrations (CCG/thoning filter, rolling baseline)
  clock.py         TimeRange, diurnal/seasonal means, decimal dates, timezones
  dev.py           development helpers
  geo.py           CRS, grids, clip, gridcell_area, regrid/resample
  hrrr.py          HRRR winds at a point, from the HRRR-Zarr archive on AWS
  inventories.py   Inventory and its loaders: EDGAR, EPA, GFEI, Vulcan, WetCHARTs
  meteorology.py   ideal gas law, hypsometric, poisson, vapor pressure, ... (plain SI)
  noaa.py          CarbonTracker and NOAA GML readers
  parallel.py      parallelize() around multiprocessing
  pcaps.py         valley heat deficit and PCAP detection
  plotter.py       matplotlib helpers
  pollutants.py    pollutant metadata registry; get_pollutant() is case-insensitive
  records.py       file utilities (ftp_download, unzip, read_kml, Cacher)
  soundings.py     upper-air soundings from the Wyoming archive (siphon)
  synoptic.py      Synoptic Data API (MesoWest's successor); token= or $SYNOPTIC_TOKEN
  transects.py     mobile transect matrices: metrics and the builder
  utils.py         small helpers
```

Instrument and archive specifics stay out: pollutant column naming is in
`uataq.pollutants`, and readers of one site's archive or one platform (such as
TRAX transects) belong in the downstream packages. `lair` re-exports only
`units` (pint-xarray's registry), `config`, `ftp_download`, `unzip`,
`setup_ccg_filter` and `__version__`; everything else is imported from its
module (`from lair.meteorology import poisson`).

## Import side effects

`lair/__init__.py` does real work at import:

1. Imports `cf_xarray.units` **before** `pint_xarray`. The order is
   load-bearing; don't reorder.
2. Adds the pint context `mass_flux` (`[substance]/[area]/[time]` <->
   `[mass]/[area]/[time]` through a molecular weight `mw`). The registry is
   process-global, so importing lair changes pint for other code too.
3. Sets pint's default sort to `sort_by_dimensionality` (`kg/km**2/d`). Not
   cosmetic: `Inventory.get_units` splits a flux unit on `/` and reads
   quantity, area and time in that order. pint >= 0.26 exports the function;
   0.24 and 0.25 (the newest on Python 3.11) have it only privately, so lair
   tries both and pyrefly flags whichever the installed pint lacks (each line
   carries its suppression). If neither exists, `import lair` raises
   `ImportError` on purpose. `pint>=0.24.4` because 0.24.0 to 0.24.3 do not
   import with flexparser 0.4. `tests/test_pkg.py` checks both.
4. Calls `setup_ccg_filter()`. If `lair/_ccg_filter.py` is absent, it
   downloads `ccg_filter.zip` from `ftp.gml.noaa.gov` into a temporary
   `lair/.ccg_filter-*` folder, takes only `ccg_filter.py`, and moves it into
   place with an atomic `os.replace`: concurrent first imports are safe and a
   failure leaves no partial file. At import a failure (or a read-only lair
   folder) only logs a WARNING; `lair.background` then raises an `ImportError`
   that says to run `lair.setup_ccg_filter()`, which raises when called.

Consequences:

- **`LAIR_SKIP_CCG_DOWNLOAD`** skips the download. Any non-empty value skips,
  `0` included (`os.getenv`). `tests/conftest.py` and the Tests, Code Quality
  and Documentation workflows set it. The Tests workflow's `pixi` job clears it
  for one step to download the filter (cached by month), so `tests/background`
  runs there; it is the only CI job that reaches NOAA.
- **`lair.background` imports `_ccg_filter` at the top**, so even
  `rolling_baseline` (pandas only) needs the downloaded file and scipy.
- **The cache folder is made on first use, never at import.**
  `config.get_cache_dir()` returns `$LAIR_CACHE_DIR` or `.cache/lair` in the
  home folder, reading the variable at each call (no module-level constant).
  `lair.noaa._cache_file` makes the folder when it first writes, and raises an
  `OSError` naming the folder and `LAIR_CACHE_DIR` when it cannot; a new cache
  user calls it. `tests/config` and `tests/noaa` check a read-only home.
- **No global xarray options.** Code that needs attributes kept scopes it
  (`@_keep_attrs` in `inventories`, or `with xr.set_options(keep_attrs=True):`).
  Both locked environments have an xarray that keeps attributes by default, so
  `TestAttributes` passes either way; check a change to the scoping on
  xarray < 2025.11.

## Commands

uv manages the environment; every task is a `just` recipe (`uv run just
--list`), and CI runs the same recipes.

| Task | Command |
|---|---|
| Set up / refresh `.venv` (lean: no extras) | `uv sync` |
| Lint, type check, docstrings, tests | `just quality-check` |
| Tests (no network/slow, parallel) | `just test` (extra args go to pytest) |
| Lint / fix and format | `just lint` / `just format` |
| Type check | `just type-check` (pyrefly) |
| Docstring coverage (at least 88%) | `just docstr` |
| Docs (runs every example; warnings are errors) | `just build-docs` / `just docs-serve` |
| All pre-commit hooks | `just pre-commit` |

The uv environment has no optional extras (the README lists them), so tests of
modules behind a missing extra `importorskip` and skip. The pixi environment
has every extra, including xesmf on conda-forge's no-MPI ESMF (64-bit Linux
only): `pixi install`, then `pixi run pytest ...` or `pixi run pyrefly check`
(the `just` recipes use uv). After editing dependencies, run `uv lock` **and**
`pixi lock` and commit both. They are solved separately and can hold different
versions, so a failure in one environment only is usually a version
difference. Dependabot updates `uv.lock` only; CI fails on a stale
`pixi.lock`, which is `merge=binary`: after a conflict, rerun `pixi lock`.

## Conventions

- Python 3.11+. Ruff rules `E, F, UP, B, SIM, I, D, D213, NPY, RUF100`
  (NumPy-convention docstrings, summary on the line after the opening quotes);
  `lair/` and `tests/` are `ruff format`-ed, and the pre-commit hook formats on
  commit. Suppressions sit on the line the tool reports: after a reflow, put
  `# pyrefly: ignore[<kind>]` on its own line *above* the flagged line and
  `# noqa` on the first line of a multi-line statement.
- **Typing:** pyrefly must report 0 errors (CI and a pre-commit hook), checked
  as Python 3.11 in the lean environment. `lair/py.typed` ships, so
  annotations are public API. Where a library's stubs are wrong (pandas
  overloads, pint callbacks), suppress inline with a reason, not in config and
  not with a blanket `Any`. The config disables only
  `bad-override-mutable-attribute` and types the optional extras as `Any` so
  both environments agree. `meteorology` uses a documented `Numeric = Any`;
  `geo` and `inventories` use the bound TypeVar `_XarrayT`.
- **Docstrings:** each public attribute and property gets its own API page, so
  give each one a docstring (one without shows an empty row) and put its See
  Also and Examples there.
- **Logging:** `logger = logging.getLogger(__name__)` per module. Library code
  never adds handlers (not even a `NullHandler`, which hides warnings from
  Python's last-resort handler), sets levels, or prints (`utils.updating_print`,
  whose job is printing, aside). Progress is `INFO`, per-item detail `DEBUG`,
  recoverable problems the caller should know about `WARNING`; %-style
  arguments, not f-strings.
- **No built-in data paths.** A reader takes an explicit directory and falls
  back to a `LAIR_*` variable through `config.get_data_dir(env_var, path)`
  (a `ValueError` names the variable), resolved at call time, never at import.
  The docs' "Data locations" lists the variables.
- **Units:** SI is the baseline; prefer pint-aware code where it exists.
  `meteorology` is plain SI in and out: `_si_inputs` converts pint inputs to SI
  magnitudes, and results are never Quantities. Callers that want units (e.g.
  `pcaps.valleyheatdeficit`) convert at their own boundary.
- **pandas 3 only** (`pandas>=3.0`): no branches, `try/except` or tests for
  pandas 2. `'1h'` (`'1H'` raises), `'D'`, `sep=r'\s+'`, no chained assignment
  (copy-on-write is always on).
- **Time:** a naive time is UTC. `convert_timezones` raises when a naive time
  has no `fromtz` (it never guesses the machine's zone) and, like
  `tz_localize`, on an ambiguous or nonexistent DST time unless
  `ambiguous=`/`nonexistent=` is given. A time formatted into a request, an S3
  key or a file name is converted to UTC first, because `strftime` and
  f-strings write a tz-aware time in its own zone (`soundings._naive_utc`,
  `hrrr.create_s3_group_url`).
- **Attribution:** code adapted from Brian Blaylock's Carpenter Workshop
  (credited in the README and docs), NOAA GML's CCG filter, and StackOverflow
  (linked in docstrings) keeps its credit when refactored.

## Module gotchas

- **Inventories.** Every loader labels a time step by the *start* of its
  period, from CF `time_bnds` when the file has them (`_label_period_starts`).
  NaN "no emission" cells become 0 before any conservative regrid, or every
  coarse cell touching one turns NaN. Without `src_units`, each variable is
  quantified with its own `units` attribute. `tests/inventories` checks every
  loader (`INVENTORY_CASES`) on tiny synthetic archives shaped like the real
  files; add new loaders there.
- **ESMF:** areas and totals (`gridcell_area`, `absolute_emissions`,
  `integrate()`) need xesmf, which pip installs without ESMF. ESMF comes from
  conda-forge's `esmf=*=nompi*` build; the MPICH build starts MPI on import,
  and a failed start kills the process (pytest included).
- **Opening many files:** call the module's `_open_mfdataset` (`inventories`,
  `noaa`), not `xr.open_mfdataset`: it names today's `compat`, `data_vars`,
  `coords` and `join`, which xarray is changing. In `noaa` it runs through
  `records.Cacher`, keyed on the pickled call arguments, so changing what is
  passed orphans every existing cache.
- **`thoning_filter`'s `sampleinterval`** is in days: the step of the even grid
  NOAA's filter interpolates onto before its FFT. lair passes the median step
  only when NOAA's own estimate would divide by zero. A grid coarser than the
  data samples it rather than averaging it, which biases hourly data with a
  diurnal cycle; test a change against the daily curve.
- **The `requests` extra floors `s3fs`, not boto3** (reason in
  `pyproject.toml`); without it, resolvers fall back to an s3fs from 2020.

## Docs

- They build in the lean environment: `docs/conf.py` mocks every optional
  extra (`autodoc_mock_imports`); add new optional imports there. Mocked types
  can't be combined at import (`Polygon | None` raises), so a module whose
  annotations name optional types needs `from __future__ import annotations`.
- `docs/api.rst` lists modules by hand; add new ones. Each class and member
  gets its own page (`docs/_templates/autosummary/`, `docs/_ext/api_pages.py`,
  which drops a class docstring's `Methods` section and the `Attributes`
  entries that have a row).
- Examples run when the docs build: `.. plot::` in a docstring's Examples
  section, `.. ipython:: python` in an `.rst` page (`@savefig name.png` above a
  plotting call). A block that raises or warns fails the build unless marked
  `:okexcept:`/`:okwarning:`; all pages share one session. Data is synthetic, a
  small file in the repository, or a public download, never a local path; seed
  random numbers. An example can use only the core dependencies; a page that
  needs an extra or a large download shows recorded output instead
  (CONTRIBUTING.md). Notebooks are off (`docs_notebooks: false`).

## Tests

- One directory per module (`lair.clock` -> `tests/clock/`) with its own
  fixtures and data and no imports between test directories, so a module can
  leave lair with two `git mv`s. Keep `tests/conftest.py` minimal.
- Markers `network` and `slow`; `just test` and CI skip both. Network code is
  tested against fakes, never live (`tests/README.md`).
- Warnings are errors: assert an expected one with `pytest.warns`; ignore one
  raised inside another library by message in `[tool.pytest]`
  `filterwarnings`, with a comment naming the library and version.
- Tests run in parallel, so must not share state. Debug serially:
  `uv run pytest tests/<module> -x`.

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
