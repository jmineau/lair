# Changelog

All notable changes to lair are documented here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and versions
are calendar-based (`YYYY.MM.PATCH`, with `MM` = 05, 08 or 12). Releases up to
2026.12.7 are described in their
[GitHub Releases](https://github.com/jmineau/lair/releases).

## [Unreleased]

### Added

- A quickstart page, and a worked example that totals an inventory over a
  region with `integrate()`, regrids it conservatively and checks that the
  total is kept. It explains why a flux goes through `Inventory.regrid` or
  `resample` (conservative) rather than `lair.geo.resample` (bilinear).
- Worked examples for soundings (download, read, potential temperature, and
  when `get_soundings` downloads), `clock` (local time, daily cycle, afternoon
  and seasonal means, decimal dates), HRRR `Winds` (UTC times, wind direction
  convention) and CarbonTracker (data folder, `download`, `sample`,
  `background`).
- The API pages of `diurnalPlot`, `seasonalPlot`, `polarPlot`, `polarFreq`,
  `log10formatter`, `truncate_colormap` and `terrain_cmap` show a figure, drawn
  from synthetic data. The examples run whenever the docs build, so a change
  that breaks one fails the build.
- A worked example that converts a methane flux from moles to mass with the
  `mass_flux` pint context and a molecular weight, and warns that pint's `kt` is
  a knot. The clock example gains a section on the errors `convert_timezones`
  raises: a time without `fromtz`, and a local time that daylight saving time
  repeats or skips (`ambiguous=`, `nonexistent=`).
- A worked example, "Estimate a background", that runs `rolling_baseline`,
  `get_well_mixed` with `thoning`, and `phase_shift_corrected_baseline` side by
  side on one series with a known background, and links to CarbonTracker's
  background for air from upwind.
- A worked example, "Draw a map of an inventory": EDGAR's methane over Utah with
  state lines, tick labels, the grid and a small map of where it is, and the same
  for a made-up footprint. It says that `Inventory.plot()` averages every time
  step by default (about a minute for EDGAR v8's 53 years), as the docstring now
  does too.

### Changed

- lair requires pandas 3 or newer (`pandas>=3.0`; breaking). Every environment
  lair is tested in already ran pandas 3, and nothing tested pandas 2. If yours
  has pandas 2, upgrade it (`pip install -U "pandas>=3"`, or the conda
  equivalent). The one piece of code that only pandas 2 needed is gone:
  `clock.convert_timezones` no longer translates pytz's `AmbiguousTimeError` and
  `NonExistentTimeError` into a `ValueError`, because pandas 3 raises a
  `ValueError` itself.
- lair requires pint 0.24.4 or newer (`pint>=0.24.4`; breaking), and `import lair`
  raises `ImportError` when pint has no `sort_by_dimensionality`, which sets the
  order a unit prints in (`kg/km**2/d`). It used to go on with pint's own order
  (`kg/d/km**2`), and `Inventory.get_units`, which reads a flux unit's quantity,
  area and time in the dimension order, then mislabeled the parts. pint 0.24.0
  is the first release with the function, but 0.24.0 to 0.24.3 do not import
  with flexparser 0.4, so the floor is 0.24.4. If you have an older pint,
  upgrade it (`pip install -U "pint>=0.24.4"`).
- The install instructions (README and front page) are rewritten. They install
  from GitHub with the extras (`pip install lair` is an unrelated project on
  PyPI), with a conda route that has ESMF (needed for grid-cell areas, inventory
  totals and regridding) and a pip route without it, and they say which
  `LAIR_*` variables each module needs, that the first `import lair` downloads
  NOAA's CCG filter (and how to skip it or place it by hand), and how to install
  on CHPC. The extras table lists `synoptic` under `requests` and `pollutants`
  under `science`.
- The environment with every extra is built by [pixi](https://pixi.sh) from
  `pyproject.toml` and `pixi.lock` (`pixi install`), with ESMF and `xesmf` from
  conda-forge. `env-dev.yml` is gone, so lair's dependencies are listed in one
  place, and CI runs the whole test suite in that environment.
- The Examples page describes each example, and the quickstart links to it
  instead of listing them. The front page links to the soundings example for
  when `get_soundings` downloads, and its Disclaimer has the README's text on
  AI coding agents.
- "Data locations" and "What needs which data" are on a page of their own, second
  in the top bar, instead of on the front page (a link to
  `index.html#data-locations` no longer lands on them). The page also links each
  variable to the example that shows its folder layout, says which pollutant
  spellings are case-sensitive, and says what lair caches and how large that
  gets. The layouts of EDGAR v7 and v8 monthly, EPA, GFEI, Vulcan, WetCHARTs and
  `GMLData` are in the Notes of their classes.
- `CarbonTrackerCH4.molefractions` and `CarbonTrackerCO2.molefractions` raise a
  `ValueError` when two files cover one day, naming the day and both files. They
  read every file of the folder, and two files for a day (the same day from two
  releases, such as `CTCH4_2024.molefrac_...` beside `CTCH4_2025.molefrac_...`)
  made them compare and keep every value of the two, so the cache grew to the
  size of the data and the open could run out of memory. The day is the date the
  file name ends with, so no file is opened to find it. A folder where an older
  release's files sit beside the current release's raises until those files are
  moved out; two releases that cover different days read as before. `sample`
  and `background` still take the last file of a day by name, and do not stop.

### Removed

- The `**kwargs` of `soundings.get_soundings`. It never used them, so a
  misspelled argument was ignored without a message (`month=[1]` returned every
  month); it now raises `TypeError`. The arguments are `station`, `start`, `end`,
  `sounding_dir`, `months` and `driver`.
- `lair.config.CACHE_DIR`, a string fixed when lair was imported (#105). Code
  that read it calls `lair.config.get_cache_dir()`, which returns a `Path`
  (`str(get_cache_dir())` for the old string). Code that set it, such as a test,
  sets the `LAIR_CACHE_DIR` environment variable instead
  (`os.environ["LAIR_CACHE_DIR"] = ...`, or pytest's `monkeypatch.setenv`).

### Fixed

- A conversion in the `mass_flux` pint context without `mw` (the molar mass),
  or with one that is not a molar mass (a plain number such as `16.04`), raises
  a `TypeError` that names `mw` and shows what to pass,
  `mw=16.04 * units("g/mol")`. It used to raise
  `TypeError: <lambda>() missing 1 required positional argument: 'mw'`, or, for
  a plain number, a `DimensionalityError` that did not mention `mw`. A
  conversion given a molar mass is unchanged.
- `CarbonTrackerCH4.molefractions` and `CarbonTrackerCO2.molefractions` no
  longer crash or fail with `NetCDF: HDF error` when they open the files in
  parallel, the default (#121). The files were opened on threads, and netCDF-C
  and HDF5 are not thread-safe; xarray holds its lock only while it opens a
  file, so reading the files' variables and attributes at once fails with
  netCDF4 1.6.1 and later, which release Python's GIL around each netCDF-C
  call (most often with 1.7). They are now opened in worker processes, one for each CPU the
  job may use, which also makes the first open of a year of daily files about
  three times faster than opening them one at a time (9 s against 25 s with 8
  CPUs). A script that reads `molefractions` needs the
  `if __name__ == "__main__":` guard (the workers import it again); without
  it, a `RuntimeError` says so. `parallel_parse=False` opens the files one at a
  time, as before.
- `hrrr.Winds` reads the right hour for times with a time zone (#99). The S3
  key was formatted in the time's own zone, so 05:00 in `America/Denver` (12:00
  UTC) read the 05 UTC analysis while the index still showed 05:00-07:00. Times
  with a time zone are now converted to UTC to choose the analysis, in
  `create_s3_group_url` and so in every key built on it; times without one are
  taken as UTC, as before, and `Winds.data` keeps the times as given.
- `soundings.download_sounding` asks Wyoming for the right hour, and names the
  file for it, when `date` has a time zone (#104). It formatted the time in its
  own zone, though Wyoming reads UTC, so 05:00 in `America/Denver` (12:00 UTC)
  asked for the 05 UTC sounding and saved it as `SLC_2024011505.csv`. Times with
  a time zone are now converted to UTC first, and times without one are taken
  as UTC, as before. `download_soundings` and `get_soundings` already converted,
  so only a direct call was affected.
- `background.thoning` and `thoning_filter` no longer raise `ZeroDivisionError`
  on data more often than daily (hourly, 30-minute) with no gap of a day, or on
  daily data within one leap year (#94). NOAA's filter estimates
  `sampleinterval` only from steps of about a day or more; when there are none,
  lair now passes the median step between samples (in days, 1/24 for hourly
  data). Every call that worked before gives the same result, and the
  docstrings document `sampleinterval`.
- `import lair.soundings` without `requests` raises the same "Optional `lair`
  dependency 'requests' not found" error as the other optional imports, not a
  bare `ModuleNotFoundError`.
- `import lair` works when the home directory is read-only (a container or a
  sandbox). It used to make the cache folder (`LAIR_CACHE_DIR`, default
  `.cache/lair` in the home directory) and fail with a `PermissionError`,
  though only `lair.noaa.CarbonTracker` uses the cache. The folder is now made the first
  time `lair.noaa` saves a cache file there. When it cannot be made, the error
  names the folder and `LAIR_CACHE_DIR` (`cache=False` caches nothing).
  The folder may not exist until something is cached.
- `LAIR_CACHE_DIR` takes effect when it is set after `import lair` (#105). lair
  read it once, when `lair.config` was imported, so a script or notebook that
  set it below its imports got the default folder (`.cache/lair` in the home
  directory) and no message. It is now read when the cache is used, as the other
  `LAIR_*` variables are, through the new `lair.config.get_cache_dir()`. An
  empty `LAIR_CACHE_DIR` is now read as unset (the default folder), as the data
  variables already are; it used to mean the current folder.
- Docstrings that said something false are corrected:
  `background.phase_shift_corrected_baseline` no longer lists a `timezone`
  parameter it does not have, `rolling_baseline` no longer prints `{q}` and
  `{window}` literally, `Sounding.interpolate` documents `interval` (not
  `step`), `thoning` documents `smooth_time`, and `hrrr`'s `chunk_id` is
  documented as the string it is (`"4.5"`), not an `int` or a `float`.
  `geo.add_extent_map` and `Sounding.units`, which were unfinished or cut off,
  are documented in full.
- The constants of `lair.meteorology` (`Rstar`, `Rd`, `kb`, `cp`, `g`,
  `epsilon`) and `lair.pcaps` (`Rd`, `cp`) are plain floats, as their comments
  said, not 0-d numpy arrays, so `help()` no longer prints numpy's
  documentation under each. The values are the same; a scalar call such as
  `ideal_gas_law` now returns a `float` where it returned a `numpy.float64`.
- The `requests` extra requires s3fs 2023.10.0 or newer. Without a floor, a new
  install could pair the newest boto3 with s3fs 0.4.2, from 2020, because the
  newest boto3 is often ahead of what s3fs's aiobotocore allows.
- `inventories` and `noaa` no longer raise xarray's `FutureWarning` about the
  changing defaults of `compat`, `data_vars`, `coords` and `join` when they
  open several files (Vulcan, WetCHARTs, EDGAR v8 annual, the CarbonTracker
  mole fractions). lair names today's values, so what the loaders return does
  not change when xarray's defaults do.
- `parallel.parallelize` counts the CPUs the process may use, not every CPU of
  the machine (#114). `num_processes="max"` started one process per CPU of the
  node (104 on the node where it was found) in a Slurm job, or under `taskset`,
  that was limited to a few, and an integer request was capped at that same
  count. Both now use the CPUs the process is allowed (`os.sched_getaffinity`),
  so a job with 16 CPUs gets at most 16 processes. On macOS and Windows, which
  do not report them, the machine's CPU count is used, as before.

## [2026.12.8] - 2026-10-09

### Added

- lair ships `py.typed`, so type checkers read its annotations.
- The documentation has a version dropdown. The site opens at the latest
  release, `dev/` follows `main`, and each release keeps its own pages.
- The API reference has a page for each class, with tables of its attributes and
  methods, and a page for each member, as in pandas. A subclass links to the
  members it inherits.
- `inventories.sum_sectors(exclude=[...])` leaves the named sectors out of a
  total, and raises `ValueError` for a name the inventory does not have. The
  EPA v1 and v2 docstrings say what each product's total covers: v1 annual
  includes `Forest_Fires`, v1 monthly and daily files hold only some sectors,
  and v2 express includes the supplemental `PostMeter`.

### Changed

- Python 3.11 or newer is required, and `typing-extensions` is no longer a
  dependency (breaking).
- lair no longer turns on pandas' copy-on-write option under pandas 2. Setting
  it changed pandas for the caller's whole session. `config.pandas_CoW` is gone.

### Fixed

- `plotter.HandlerDashedLines` draws legend dashes that match a thick dashed
  segment. They were scaled by the line width twice, so a segment 3 points wide
  got dashes 3 times too long.
- `records.list_files(ignore_case=False)` is case-sensitive on every operating
  system. It ignored case on Windows.
- `clock.convert_timezones(driver="pandas")` raises the `ValueError` it
  documents for input that is not a DataFrame or Series. It raised `TypeError`.
- `records.ftp_download` closes its FTP connection however the download ends.
  It left the control socket open when a transfer failed or the server did not
  answer QUIT (Python then warned of an unclosed socket, which fails strict test
  runs downstream; seen on the first `import lair`, which downloads the CCG filter).
