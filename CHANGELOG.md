# Changelog

All notable changes to lair are documented here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and versions
are calendar-based (`YYYY.MM.PATCH`, with `MM` = 05, 08 or 12). Releases up to
2026.12.7 are described in their
[GitHub Releases](https://github.com/jmineau/lair/releases).

## [Unreleased]

### Added

- The API pages of `diurnalPlot`, `seasonalPlot`, `polarPlot`, `polarFreq`,
  `log10formatter`, `truncate_colormap` and `terrain_cmap` show a figure, drawn
  from synthetic data. The examples run whenever the docs build, so a change
  that breaks one fails the build.

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
