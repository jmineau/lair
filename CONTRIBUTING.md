# Contributing to lair

Thank you for considering a contribution. Bug reports and feature requests go
in [the issue tracker](https://github.com/jmineau/lair/issues);
code changes come as pull requests.

Coding agents: read [AGENTS.md](AGENTS.md) as well.

## Setup

You need [uv](https://docs.astral.sh/uv/) and git. [just](https://just.systems/)
comes with the dev tools.

```bash
git clone https://github.com/jmineau/lair.git
cd lair
uv sync                    # .venv with the package (editable) and the dev tools
uv run pre-commit install  # run the hooks on every commit
```

The uv environment is lean: it does not install the optional extras (`geo`,
`formats`, `regridding`, ...), so the tests of the modules behind them skip.
For full coverage, use the [pixi](https://pixi.sh) environment, which has
every extra (the `regridding` extra needs conda-forge's ESMF, which uv cannot
install). It is locked in `pixi.lock`, for 64-bit Linux:

```bash
pixi install     # builds .pixi/envs/default
pixi run pytest -n auto --maxprocesses=8 -m "not network and not slow"
pixi run pyrefly check   # the type check, in the same environment
```

CI runs the tests in both environments. The first `import lair` in an
environment downloads NOAA's CCG filter (the
[installation page](https://jmineau.github.io/lair/#installation) says why),
and `LAIR_SKIP_CCG_DOWNLOAD` (any non-empty value, `0` included) skips that.
`tests/conftest.py` and the Tests, Code Quality and Documentation workflows set
it. The Tests workflow's `pixi` job clears it for one step to download the
filter (cached by month), so `tests/background` runs there; it is the only CI
job that reaches NOAA.

## Checks

Every check has a `just` recipe, and CI runs the same recipes. Run them with
`uv run just <recipe>`, or plain `just` if it is on your PATH (`just --list`
shows them all).

| Recipe | What it does |
|---|---|
| `just quality-check` | `lint`, `type-check` (pyrefly), `docstr` (docstring coverage of at least 88%) and `test`: what CI checks |
| `just test` | pytest in parallel, skipping tests marked `network` or `slow`; it also runs the `>>>` examples in the docstrings under `lair/` (extra args go to pytest; plain `uv run pytest` runs serially, for debugging) |
| `just format` | fix lint and format the code with ruff |
| `just build-docs` | build the HTML docs into `docs/_build/`, running every example (warnings are errors) |
| `just docs-serve` | preview the docs at <http://127.0.0.1:8000>, rebuilt on every save (on a remote machine, forward the port, e.g. VS Code's Ports panel) |
| `just changelog` | draft CHANGELOG entries from the commits since the last release (run at release) |
| `just pre-commit` | run every hook on every file |

After changing dependencies in `pyproject.toml`, run `just lock` (the
`uv-lock` hook also does this on commit) and `pixi lock`, and commit `uv.lock`
and `pixi.lock` with the change. CI fails on a `pixi.lock` that is out of date.
The two are solved separately, so a test that fails in one environment only is
usually a version difference. `pixi.lock` is `merge=binary` (`.gitattributes`):
after a merge or rebase conflict in it, rerun `pixi lock`.

## Making a change

1. Branch from `main`.
2. Make the change, with tests and NumPy-style docstrings.
3. Commit with a [Conventional Commits](https://www.conventionalcommits.org/)
   message that names the module as its scope, e.g.
   `feat(inventories): sum_sectors(exclude=)` or `fix(soundings): ...`, with
   `!` for a breaking change. Leave [CHANGELOG.md](CHANGELOG.md) alone: the
   release notes are drafted from these messages when the maintainer releases.
4. Open a pull request. Link the issue it addresses ("Fixes #12").

## Code layout

The package is flat (`lair/`, not `src/lair/`): one module per topic, no
subpackages, and one test directory per module (`tests/README.md`).

```
__init__.py      pint setup and the CCG filter download (AGENTS.md, "Import side effects")
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
`plotter`, `pollutants` and `utils`, each named for what it holds.

## Notes on the code

Two traps that a change to these modules can fall into. The docstrings of the
functions named here hold the details.

- **CarbonTracker times:** each mole fraction step is a 3-hour mean labeled by
  its center, taken from the file's `time_components` in every release
  (`CarbonTracker._preprocess_molefractions`): CT-CH4-2025's own `time` and
  `time_bounds` are 1.5 h late. Check a new release's `time` against its
  `time_components` before trusting either.
- **`thoning_filter`'s `sampleinterval`** is in days: the step of the even grid
  NOAA's filter interpolates onto before its FFT. lair passes the median step
  only when NOAA's own estimate would divide by zero. A grid coarser than the
  data samples it rather than averaging, which biases hourly data with a
  diurnal cycle; test a change against the daily curve.

## Docs examples

Examples run when the docs build, so they show real output, and the build
fails when one breaks.

- **A figure in a docstring:** put a `.. plot::` directive in the Examples
  section. Its code runs (doctest `>>>` lines work), and the figure appears
  under it.
- **Code in an `.rst` page:** a `.. ipython:: python` block runs and shows each
  line with its output, as in an IPython session. For a figure, put
  `@savefig name.png` on the line above the plotting call. A block that raises
  or warns fails the build; `:okexcept:` or `:okwarning:` under the directive
  allows one that is meant to. All pages share one session, so each page
  imports and defines what it uses. Split a long block so that no multi-line
  statement is its 10th (or 100th) statement: the directive runs only the first
  line of that one.
- **When they run:** `just build-docs` runs everything, as CI does;
  `just docs-serve` reruns a page's blocks whenever that page is rebuilt.
  Notebooks and code cells in Markdown pages are off in this project.
- **Docstring examples are tests too:** `just test` runs the `>>>` lines under
  `lair/` (`--doctest-modules`). An example that needs data CI lacks gets
  `# doctest: +SKIP` on its `>>>` line. In the lean environment a module that
  imports an extra it lacks is skipped (`--doctest-ignore-import-errors`), so its
  examples run only in the pixi job: `pixi run pytest lair` runs them all.
- **Data:** an example uses synthetic data, a small file in the repository, or
  a public download, never a machine-specific path. Seed random numbers so the
  figures don't change from build to build.
- **Optional dependencies:** the docs build in the lean uv environment, where
  `docs/conf.py` mocks the extras, so an example that runs can use only the
  core dependencies.

### Pages with recorded output

A page whose code needs an optional extra, ESMF, or a large download cannot
run in the docs build. Such a page shows its code and output as
`.. code-block:: ipython`, recorded from one real run in the pixi environment,
and says so in a note at the top with the date of the run
(`docs/examples/inventory.rst`, `soundings.rst`, `hrrr_winds.rst`,
`carbontracker.rst`, `background.rst` and `maps.rst` do; `clock.rst` runs when
the docs build). A recorded page can show a figure: the recorded code saves it
(`fig.savefig("name.png")`), and the page includes the PNG, which is committed
beside the page (`maps.rst`).

When the code such a page calls changes, run the page's code again and paste
the new output. Run it in an empty folder: the blocks use relative paths, so
no machine's paths reach the page.

## Where a fact goes in the docs

Each fact is written on one page, and the others link to it rather than
restate it (AGENTS.md, "Docs"). A new example goes in `docs/examples/index.rst`
only (a toctree entry and a line describing it); the quickstart links to the
Examples page, not to each example. A new fact goes on the page that holds its
kind:

| Fact | Its page | The others |
|---|---|---|
| Install routes, extras, the first-import download, CHPC | front page, "Installation" | link by `installation` or `optional-dependencies`; the README keeps a short copy, as GitHub's front page |
| `LAIR_*` variables, what needs which data, the pollutant spelling, the cache | `data`, "Data locations" | link by `data-locations` |
| A dataset's folder layout and where to get it | its example, "The data"; no example: its class docstring | "Data locations" links to it |
| A time without a zone is UTC; time zones; seasons | `examples/clock` | a clause and a link |
| Units; the `mass_flux` context | `examples/units` | link |
| Plain SI in and out of `meteorology` | quickstart, "A first calculation" | link |
| Inventory totals; conservative against bilinear regridding. Drawing a map: the time step, state lines, ticks, grid, locator map | `examples/inventory`; for the map, `examples/maps` | link |
| `get_soundings` heights and downloads | `examples/soundings` | link |
| CarbonTracker sampling and background | `examples/carbontracker` | link |
| Background methods, side by side, and which one to use | `examples/background` | link |
| Function and class details | the API pages, from docstrings | `:func:` and `:class:` links, not `autofunction` |

## Releasing

The version comes from git tags, through setuptools-scm, so there is no
version string to bump. Releases are calendar-based, `YYYY.MM.PATCH`: `MM` is
05, 08 or 12 (the release period the month falls in), and the patch counts up
within a period and restarts at 0 in a new one. `just next-version` prints the
next one.

1. Write the changelog: `just changelog` drafts the entries from the commit
   messages since the last release. Edit them into `CHANGELOG.md` under
   `## [Unreleased]`, then rename that heading to `## [X.Y.Z] - YYYY-MM-DD` (the
   version `just next-version` prints) and start a new empty `## [Unreleased]`
   above it. Commit and push to `main`.
2. Run `just release`. It checks that the tree is clean, that `main` is in sync
   with GitHub, that `CHANGELOG.md` has the section, and that the version is
   newer than every existing tag, then pushes the tag `vX.Y.Z`.
3. The Publish workflow builds the tag and creates a GitHub Release from the
   CHANGELOG section. Zenodo archives the release and mints a DOI. The
   Documentation workflow publishes the docs as `X.Y.Z/` (and as `stable/`) in
   the version dropdown.

lair is not published to PyPI; it is installed from GitHub. Citation
metadata is in `CITATION.cff` and `.zenodo.json`.

## Dependency updates

Dependabot opens one pull request a month for each of GitHub Actions and
pre-commit hooks. Merge it when CI passes.

`uv.lock` (the dev tools and every dependency) is updated by the Upgrade lock
workflow alone, not by Dependabot, which would leave a package at its old
version when the newest release needs a newer Python than the oldest one this
project supports. On the 1st of each month the workflow runs `uv lock --upgrade`
(`uv.lock` then holds the newest release for each Python; it never raises the
minimum versions in `pyproject.toml`) and, when the lock changes, opens the pull
request `chore/lock-upgrade` (or updates it). GitHub does not start workflows on
a pull request that its own token opened, so the workflow starts Tests, Code
Quality and Documentation on the branch itself; their results show on the pull
request. Merge it when they pass. You can also run it by hand: Actions > Upgrade
lock > Run workflow.

The workflow does not touch `pixi.lock` (the environment with every extra), so the two lockfiles can hold different versions of a package.

## Template

The tooling (CI workflows, pre-commit, justfile, packaging configuration) comes
from [jmineau/python-template](https://github.com/jmineau/python-template).
`.copier-answers.yml` records the template version and is never edited by hand;
`copier update` pulls in later template changes. Improvements that would help
every package are best made in the template.
