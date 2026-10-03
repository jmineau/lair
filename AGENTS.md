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

# AGENTS.md — Developer and Agent Guide for lair

`lair` (Land-Air Interactions Research) is a personal toolkit of atmospheric-
science utilities the user has accumulated/adapted for their research. Less
opinionated and more loosely organized than the user's newer packages — think
"my standard library", not "a framework".

PyPI/import name: `lair`. Calendar-versioned (e.g. `2026.05.10`). Layout is
**flat** (`lair/`, not `src/lair/`). Docs published on GitHub Pages at
<https://jmineau.github.io/lair/> (the old CHPC site was retired 2026-09-21;
`~/public_html/lair/` only holds redirect pages; `~/public_html/software.php`
links to GitHub Pages).

**Versioning (switched 2026-09-21):** setuptools-scm derives the version from
git tags (`dynamic = ["version"]`; `lair.__version__` reads the installed
metadata). Releases are CalVer tags `vYYYY.MM.PATCH` with MM = 05/08/12, made
by `just release` (clean main, in sync with origin; new release month -> .0,
otherwise PATCH+1; pushes the tag, then `gh release create` so Zenodo archives
it and mints a DOI — Zenodo ignores bare tags). Citation metadata lives in
`CITATION.cff` + `.zenodo.json` (same shape as fips/PYSTILT/slv). Between tags installs report e.g.
`2026.12.4.dev3+g<hash>`. `just version` prints the current one. The old CI
bump bot (`update_version.yml` + `lair/_version.py`) is gone, so no more
pull-before-push. CI checkouts use `fetch-depth: 0` so the tags are visible;
`.git_archival.txt` covers GitHub archive downloads. First scm tag:
`v2026.12.3` (the last bot-bumped commit).

## What lair is and isn't

- **Is**: a grab-bag of analysis helpers — meteorology, geo, HRRR access,
  inventories, NOAA GML data, upper-air soundings, PCAP/VHD calculations,
  background CCG filtering, plotting helpers, parallel/file utilities.
- **Isn't**: a strictly versioned API. The user marks it Alpha; classes
  and module boundaries shift. Don't treat any symbol as load-bearing for
  external consumers without checking with the user.

## Layout (flat, single-package)

```
lair/
  __init__.py        sets up pint units (incl. custom mass_flux context),
                     runs setup_ccg_filter() at import — see "Import side
                     effects" below
  _ccg_filter.py     NOAA GML CCG filter (downloaded on first import;
                     **do not commit**)
  _optional.py       import_optional_dependency() helper
  config.py          data-dir resolution (get_data_dir), CACHE_DIR
  constants.py       physical constants (workaround until pint has them)
  air.py             general atmospheric helpers (bin_polar, wind_components,
                     wind_direction -- NaN for a zero vector, not 270)
  background.py      background concentration via CCG filter
  clock.py           time/date utilities (TimeRange-ish, decimal date conv.)
  dev.py             dev helpers (e.g. public_attrs)
  geo.py             geo-spatial utilities (cartopy / shapely / pyproj based)
  hrrr.py            HRRR winds at a point (zarr / MesoWest backend)
  inventories.py     emissions inventory loaders
  meteorology.py     met calcs (ideal gas, hypsometric, poisson, ...)
  noaa.py            NOAA greenhouse-gas data loaders
  parallel.py        parallelize() wrapper around multiprocessing
  pcaps.py           valley heat deficit (VHD) + PCAP detection
  plotter.py         matplotlib helpers (log formatter, legend handlers, ...)
  records.py         file/dir utilities (ftp_download, unzip)
  soundings.py       upper-air sounding fetch/parse
  synoptic.py        Synoptic Data API (MesoWest successor): metadata, timeseries,
                     hourly_mean (vector-mean wind). Token from token= or
                     $SYNOPTIC_TOKEN at call time; needs the `requests` extra.
  transects.py       mobile transect matrices (obs[transit, point]): metrics
                     (enhancement, robust_z, detection_frequency, magnitude,
                     profile, merge_route_points/pool_routes) and the builder
                     (lag_positions, snap_to_route, split_transits — one-way
                     transits cut at reversals of travel via find_peaks prominence
                     and at time gaps — transect_matrix). scipy is optional:
                     the builder tests importorskip it. Platform specifics
                     (TRAX) live in slv.measurements.mobile.transects.
  utils.py           tiny helpers (updating_print, DotDict)
docs/                Sphinx source (conf.py, index.rst, api.rst, _templates/);
                     .github/workflows/docs.yml deploys to GitHub Pages
env-dev.yml          conda dev env
tests/               pytest suite, CONTAINERIZED per module (see "Testing")
justfile             dev tasks (uv-based): sync/lint/type-check/test/...
.pre-commit-config.yaml   pre-commit + ruff hooks (pyrefly deliberately omitted)
.github/workflows/   tests.yml (matrix; Codecov upload from ubuntu/3.12, needs the
                     CODECOV_TOKEN repo secret), quality.yml (ruff + pyrefly gates),
                     docs.yml (GitHub Pages)
```

There are no subpackages; everything is a top-level module.

> **No WIP modules left:** `noaa.py` (CarbonTracker CH4/CO2 + GML) was the last
> one excluded from ruff/pyrefly; it is fully linted and type-checked since #25.
> The SODAR reader (`lair.mesowest`) moved to the `uataq` package on 2026-09-21
> (CHPC-archive readers don't belong in lair).

## Public API surface

`lair.__init__` only re-exports a couple of things explicitly
(`ftp_download`, `unzip`, and `config`) plus the pint `units` registry. In
practice everything else is reached via submodule import:

```python
from lair import units                    # pint unit registry (with mass_flux ctx)
from lair.config import get_data_dir
from lair.background import ccgFilter
from lair.meteorology import ideal_gas_law, hypsometric, poisson
from lair.pcaps import ...                # VHD / PCAP helpers
from lair.hrrr import ...                 # HRRR point sampling
from lair.inventories import ...
from lair.noaa import ...
```

## Import side effects (read this before debugging strange import behavior)

`lair/__init__.py` runs unconditionally at import:

1. Imports `cf_xarray.units` (**must come before `pint_xarray`** — order is
   load-bearing).
2. Registers a custom pint context `mass_flux` that converts
   `[substance] / [area] / [time]` ↔ `[mass] / [area] / [time]` using
   molecular weight.
3. Sets pint's default sort to `sort_by_dimensionality`.
4. Calls `setup_ccg_filter()` which, **on first import**, FTP-downloads
   `ccg_filter.zip` from `ftp.gml.noaa.gov` into a temporary directory
   (`lair/.ccg_filter-*`, removed afterwards), takes only `ccg_filter.py` from
   it (other zip members are ignored, so NOAA can change them), and moves it
   into place as `_ccg_filter.py` with an atomic `os.replace`. A failed
   download raises (so `import lair` fails) but leaves no partial file;
   concurrent first imports (SLURM arrays) are safe. After that it's a no-op.
   Tests call it with `lair_dir=` and a stubbed `lair.ftp_download`
   (`tests/test_pkg.py`).

Implications:
- First import does network I/O. Don't `import lair` inside a tight test
  loop and don't run it on a machine without outbound FTP if you've never
  imported it before.
- **Escape hatch:** set `LAIR_SKIP_CCG_DOWNLOAD=1` to skip the FTP download
  (for tests / CI / read-only / offline environments). When skipped *and* the
  file is absent, `lair.background` is not importable. The test suite sets this
  in `tests/conftest.py`; CI sets it at the workflow level.
- `lair/_ccg_filter.py` is generated, not authored. Do not commit it. The
  user's `.gitignore`/`.git/info/exclude` already handles it.
- The pint registry is process-global; importing `lair` mutates other code's
  pint behavior in the same process.
- **No global xarray options** (#38): `lair.inventories` used to run
  `xr.set_options(keep_attrs=True)` at import. Code that needs attributes kept
  scopes it: `@_keep_attrs` (absolute_emissions, integrate, plot) or a `with
  xr.set_options(keep_attrs=True):` block (loaders' `_open`/`_process` in
  `Inventory.__init__`, `Vulcan.get_uncertainties`). Only xarray < 2025.11
  drops attrs by default, so run the attr tests (`TestAttributes`) in
  `lair-dev`.

## Logging

lair uses the standard `logging` module (since #22; `config.verbose` and
`vprint` are gone). Each module has `logger = logging.getLogger(__name__)`;
no handler is attached anywhere. Never add handlers (not even a `NullHandler`,
which would also hide warnings from Python's last-resort handler), set levels,
or `print()` in library code: output is the application's choice
(`logging.basicConfig(level=logging.INFO)`).

Levels: progress ("Downloading ...", "Executing ... in parallel") is `INFO`;
chatty per-item detail (cache hits, skipped files) is `DEBUG`; recoverable
problems the caller should know about (a failed download that is skipped, an
unreadable file) are `WARNING`, which shows by default. Use %-style arguments
(`logger.info("x %s", y)`), not f-strings. `background.thoning_filter` turns
the CCG filter's own `debug` prints on when `lair.background` is at `DEBUG`.

## Dev workflow

Tooling mirrors the user's newer packages (e.g. `arl-met`): `uv` + `justfile`
+ `ruff` + `pyrefly` + `pytest` + `pre-commit`. Config lives in
`pyproject.toml` (`[dependency-groups]`, `[tool.uv]`, `[tool.pytest.ini_options]`,
`[tool.coverage.*]`, `[tool.pyrefly]`, `[tool.ruff]`).

```bash
# uv-based dev env (LEAN: tooling only, no optional extras). Fast + reliable.
uv sync --group dev          # or: just sync
just quality-check           # ruff (gate) + pyrefly (advisory) + hermetic tests
just test                    # pytest, excluding slow
just lint                    # ruff check lair tests
just build-docs              # Sphinx HTML -> docs/_build/html (deployed by CI)
just type-check              # pyrefly (advisory)
```

`uv run just --list` shows all recipes. The uv dev group does **not** install
the optional extras (geo/formats/regridding need conda-built ESMF / heavy
wheels). For full optional-dependency coverage, use the conda `lair-dev` env
(`env-dev.yml`) and run `pytest` there — modules behind missing extras
`importorskip` and skip cleanly otherwise.

```bash
# conda-based install (required for esmpy / regridding extra)
mamba activate lair-dev
mamba install -c conda-forge esmpy   # only if you need the regridding extra
pip install -e .                     # or: uv pip install -e .
```

**Linting/formatting (ruff):** `ruff check` (E, F) is an enforced gate, and
since 2026-09-21 the whole of `lair/` and `tests/` is `ruff format`-ed (commit
bc973df). Keep it that way: the
pre-commit `ruff-format` hook formats on commit, or run `just format`.
Suppression comments must sit on the line the tool reports; after a reflow,
put `# pyrefly: ignore[...]` on its own line *above* the flagged line (the
formatter never moves those), and `# noqa` on the first line of a multi-line
statement. Generated/vendored modules are in `extend-exclude`.

**Typing (pyrefly): enforced gate since 2026-09-21** (CI Code Quality job,
`just quality-check`, pre-commit hook). It passes with 0 errors on both the
lean uv stack (pandas 3) and `lair-dev` (pandas 2.2). Rules: fix real
annotation problems; where library stubs are wrong/narrow (pandas `cut`/
`resample.agg` overloads, pint transformation callbacks, pint's
version-dependent private import), suppress **inline on that line with a
reason** (`# pyrefly: ignore[<kind>]`), not in config. Config only disables
`bad-override-mutable-attribute` (subclass class constants like
`version: str = 'v8'` over `str | None` base attrs). Optional extras are in
`replace-imports-with-any` so both envs agree. `meteorology` uses a
documented `Numeric = Any` alias (floats / numpy / xarray, plain SI);
`geo`/`inventories` use a bound TypeVar `_XarrayT` so DataArray in ->
DataArray out.

## Testing

Tests are **containerized by module**: every submodule has its own directory
under `tests/` (`lair.clock` → `tests/clock/`), holding that module's tests,
`conftest.py` fixtures, and `data/`. Motivation: modules regularly graduate out
of lair into their own packages — `git mv lair/<mod>.py` + `git mv tests/<mod>/`
should be all it takes. See `tests/README.md`. Rules: no cross-imports between
test dirs; module-local fixtures stay in the module's dir; keep the top-level
`tests/conftest.py` minimal.

- **Markers** (`pyproject.toml`): `network`, `slow`, `chpc`. The hermetic CI
  subset is `-m "not network and not slow and not chpc"`.
- **Coverage so far:** every module has real tests (lair-dev: 268 passed,
  ~70% coverage; lean uv: 187 passed / 8 skipped). Thinnest: `soundings`
  (offline parsing/interpolation only; live fetches untested), `records`
  (network helpers only via a stubbed `subprocess.run`). `pcaps.valleyheatdeficit`
  is checked against an independent trapezoid VHD of a synthetic sounding.
- **Two pandas majors:** the lean uv env resolves pandas 3.x, `lair-dev` has
  pandas 2.2. Run both before calling a pandas change done.

## Conventions

- **Style**: ruff-formatted (E/F lint rules). Keep diffs focused; don't mix
  refactors into fixes.
- **Typing**: partial but checked. pyrefly must stay at 0 errors — annotate
  new code; don't add blanket `Any` to silence it.
- **Pint**: prefer pint-aware code where it already exists, but SI units
  are the implicit baseline (the meteorology module documents this).
- **Paths (no CHPC locations in lair — done 2026-09-21)**: lair has no
  built-in data paths. Readers take an explicit directory argument and fall
  back to a `LAIR_*` env var via `lair.config.get_data_dir(env_var, path)`
  (raises `ValueError` naming the variable). Current variables:
  `LAIR_INVENTORY_DIR` (inventories root; every concrete inventory class takes
  `inventory_dir=`), `LAIR_SOUNDING_DIR` (root with one subdir per station),
  `LAIR_CARBONTRACKER_DIR`, `LAIR_GML_DIR`, plus `LAIR_CACHE_DIR`. Resolve
  lazily (at call/construction time), never at import. Readers tied to a CHPC
  archive belong in `uataq`/`slv`, not lair. Don't add new hard-coded paths.
- **Attribution**: portions adapted from Brian Blaylock's Carpenter
  Workshop, NOAA GML's CCG filter, and StackOverflow. Preserve attribution
  if refactoring those sections.

## Optional dependency groups

| Extra | Adds | Why |
|---|---|---|
| `requests` | boto3, s3fs, siphon | Remote data (AWS, soundings) |
| `formats` | zarr, numcodecs, fastkml, lxml | Alt file formats |
| `geo` | cartopy, networkx, pyproj, rasterio, rioxarray, shapely | Geographic processing |
| `science` | molmass, scipy | Chemistry / CCG filter |
| `regridding` | xesmf | needs conda-built ESMF (see README) |
| `complete` | all of the above | one-shot |

## Gotchas

- `cf_xarray.units` *must* be imported before `pint_xarray` in
  `__init__.py`. Don't reorder.
- `lair.constants` exists because pint hasn't shipped real constants
  support; if pint adds it (see linked issue 1078 in the module
  docstring), revisit.
- `lair.meteorology` is **plain SI in, plain SI out** (issue #41): it uses
  the `lair.constants` values as plain SI floats (`.m_as(...)`), so float /
  numpy / xarray inputs give plain numbers (`hypsometric` -> m,
  `ideal_gas_law("p")` -> Pa). pint Quantity inputs (incl. pint-quantified
  DataArrays) are converted to SI magnitudes by the `_si_inputs` decorator;
  the result is never a Quantity. Callers that want units (e.g.
  `pcaps.valleyheatdeficit`) convert at their own boundary.
  `mixing_ratio` is the exact ε·e/(p − e).
- **pandas 3 idioms:** hourly alias is lowercase (`'1h'`; `'1H'` raises) and
  daily is uppercase (`'D'`; `'d'` is deprecated). `read_csv(delim_whitespace=)`
  is gone (use `sep=r'\s+'`), `DataFrame.interpolate` refuses object-dtype
  columns, and `pd.options.mode.copy_on_write` is always on (config only sets
  it on pandas < 3).
- **Timezones in `clock`:** naive times are UTC by convention.
  `dt2decimalDate` converts tz-aware input to UTC first (same answer under
  pytz/pandas 2 and zoneinfo/pandas 3). `convert_timezones` raises if a naive
  time has no `fromtz` (it never guesses the machine's zone), and localizes
  naive times with pandas `tz_localize` semantics in both drivers, so a DST
  ambiguous/nonexistent time raises unless `ambiguous=`/`nonexistent=` is given.
- **Vulcan (fixed 2026-09-21):** the files are `(time, y, x)` on an LCC grid
  with 2D lat/lon, so `Inventory.__init__` picks x/y spatial dims when there's
  no `lon` dim. `Vulcan.clip`/`reproject` follow the base-class API (return the
  result, `inplace=` optional; reproject refuses an unclipped grid unless
  `force=True`). **Units:** Vulcan stores *tonnes of carbon*; `_process`
  converts to tonnes of CO2 (x M(CO2)/M(C)) and fills NaN with 0.
- **Inventory loaders (2026-10):** NaN "no emission" cells must be 0 before any
  conservative regrid (Vulcan, WetCHARTs), or each coarse cell touching one goes
  NaN. Without `src_units`, each variable is quantified with its own `units`
  attr. EPA v2 monthly scale factors cover only 10 of ~28 sectors:
  `scale_by_month` keeps the rest at their annual rate. EDGAR v8 drops
  `Fuel_exploitation` (PRO_FFF) only when its COAL/GAS/OIL parts are loaded
  (monthly has no split). Test loaders on tiny synthetic archives shaped like
  the real files (check layouts with `ncdump -h`).
- **Inventory time labels (2026-10, #37):** every loader labels a step by the
  *start* of its period (Jan 1 / the 1st / 00:00 / top of the hour). Use the
  file's CF `time_bnds` when it has them (`_label_period_starts`: Vulcan,
  WetCHARTs); EDGAR v8 monthly has none (15th of each month) and is floored to
  the month. `tests/inventories` checks every loader (`INVENTORY_CASES`) and
  that totals don't depend on the labels; add new loaders there.
- **Docs:** `just build-docs` (or `sphinx-build -M html docs docs/_build`).
  They build in the lean uv env because `docs/conf.py` mocks every optional
  extra via `autodoc_mock_imports` — add new optional imports there. Mocked
  types can't be combined at import time (`Polygon | None` raises), so modules
  whose annotations name optional types need `from __future__ import
  annotations` (geo, inventories). `docs/api.rst` lists modules by hand; add
  new modules to it. The "duplicate object description" warnings come from
  numpydoc `Methods` sections in class docstrings (napoleon renders them).
- `setup_ccg_filter()` at import time touches the filesystem. Don't import
  `lair` inside an immutable / read-only environment without
  pre-installing `_ccg_filter.py` (or set `LAIR_SKIP_CCG_DOWNLOAD=1`).
