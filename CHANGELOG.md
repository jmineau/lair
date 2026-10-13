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

### Changed

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

### Fixed

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
  `lair.config.CACHE_DIR` is unchanged, but the folder may not exist until
  something is cached.
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
