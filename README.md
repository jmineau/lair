<div align=center>

# LAIR (Land-Air Interactions Research)

</div>

`lair` is a collection of tools that I have developed/acquired for my research regarding land-air interactions. I spent a lot of time developing the `uataq` subpackage which provides a simple interface for reading in data from the Utah Atmospheric Trace Gas and Air Quality (UATAQ) project - the idea being that we as scientists spend too much time on data wrangling and not enough time on analysis. The `lair` package is designed to make it easier to work with atmospheric data, particularly data from the UATAQ project.

# Installation

`lair` can be installed via `pip`:
```bash
pip install git+https://github.com/jmineau/lair.git
```

or by cloning the repository and installing it as an editable package:
```bash
git clone https://github.com/jmineau/lair.git
cd lair
pip install -e .
```

## Optional Dependencies

`lair` is a rather dependecy-heavy package, however, many are common in the field of atmospheric science. To keep `lair` as lightweight as possible, some dependencies related to meteorology are currently optional. These include:
 - `boto3`
 - `metpy`
 - `s3fs`
 - `siphon`
 - `synopticpy`
 - `zarr`

The following modules are impacted:
 - `lair.air.hrrr`
 - `lair.air.soundings`

### Installing Optional Dependencies

To install the optional dependencies, use the following command:

```bash
# via conda (preferred on CHPC)
conda activate <lair-env>  # activate your lair environment
conda install -c conda-forge boto3 metpy s3fs siphon synopticpy zarr

# via pip
pip install git+https://github.com/jmineau/lair.git[met]
```

# Contents

### `lair.air`

The `air` module contains various modules for working with atmospheric data:
 - `air.air` - Miscellaneous functions for working with atmospheric data.
 - `air.enhancements` - Functions for calculating a background/baseline from a time series.
 - `air.hrrr` - Functions for working with data from the High-Resolution Rapid Refresh (HRRR) model.
 - `air.inventory` - Classes and functions for working with emissions inventories.
 - `air.mesowest` - Functions for working with data from the MesoWest API.
 - `air.meteorology` - Functions for various meteorological equations.
 - `air.noaa` - Functions to get NOAA greenhouse gas data including flasks and CarbonTracker.
 - `air.plots` - Functions for plotting atmospheric data.
 - `air.soundings` - Functions for working with atmospheric soundings.
 - `air.stilt` - Classes and functions for working with data from the Stochastic Time-Inverted Lagrangian Transport (STILT) model.

### `lair.config`

The `config` module contains configuration settings for the `lair` package.

### `lair.constants`

The `constants` module contains various constants used throughout the `lair` package.

### `lair.uataq`

The `uataq` subpackage contains classes and functions for working with data from the Utah Atmospheric Trace Gas and Air Quality (UATAQ) project.

 - [Documentation](docs/uataq.md)

### `lair.units`

The `units` module contains various unit conversion functions & factors used throughout the `lair` package.

### `lair.utils`

The `utils` module contains various utility functions used throughout the `lair` package.

### `lair.valley`

The `valley` module contains classes and functions for creating maps of measurements taken` in the Salt Lake Valley.

# Verbosity

Verbosity for the `lair` package is set via `lair.config.verbose` as a boolean.

For early versions of the package, `verbose` will be set to `True` by default. This will be changed in future versions.

# Acknowledgements

This package was partially inspired and uses some code generously provided by [Brian Blaylock's Carpenter Workshop python package](https://github.com/blaylockbk/Carpenter_Workshop).

# Disclaimer

 - Portions of this package were written with AI-based tools including Github CoPilot, ChatGPT, and Google Gemini.
 - Additionally, various code snippets were borrowed from StackOverflow and other online resources.

# Contributing

Contributions are welcome! Please take a look at current [issues](https://github.com/jmineau/lair/issues) and feel free to submit a pull request with new features or bug fixes.

# Citation

If you use any portion of this package in your research, please cite the software and/or acknowledge me.

A DOI will be provided in the future.