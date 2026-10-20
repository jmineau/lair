<div align=center>
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="/docs/_static/lair_fordark_r.png">
    <source media="(prefers-color-scheme: light)" srcset="/docs/_static/lair_forlight_r.png">
    <img alt="lair logo" src="/docs/_static/lair_fordark_r.png">
</picture>

# Land-Air Interactions Research

## [:scroll: Documentation](https://jmineau.github.io/lair/)

[![Tests](https://github.com/jmineau/lair/actions/workflows/tests.yml/badge.svg)](https://github.com/jmineau/lair/actions/workflows/tests.yml)
[![Documentation](https://github.com/jmineau/lair/actions/workflows/docs.yml/badge.svg)](https://github.com/jmineau/lair/actions/workflows/docs.yml)
[![Code Quality](https://github.com/jmineau/lair/actions/workflows/quality.yml/badge.svg)](https://github.com/jmineau/lair/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/jmineau/lair/branch/main/graph/badge.svg)](https://codecov.io/gh/jmineau/lair)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22900040.svg)](https://doi.org/10.5281/zenodo.22900040)

</div>

`lair` is a collection of tools that I have developed/acquired for my research regarding land-air interactions. The `lair` package is designed to make it easier to work with atmospheric data.

# Installation

`lair` is installed from GitHub, not from PyPI. On PyPI, `lair` is an unrelated
project (a helper for Flask web services), so `pip install lair` installs the
wrong package.

You need Python 3.11 or newer and [git](https://git-scm.com/). Install lair into a
new environment, so that its packages stay apart from the rest of your work.

Regridding (`geo.regrid`, `geo.resample`, and an inventory's `regrid` and
`resample`) needs ESMF, a compiled library that pip cannot install. For it, use
conda:

```bash
mamba create -n lair -c conda-forge python=3.12 esmpy "esmf=*=nompi*"
conda activate lair
pip install "lair[complete] @ git+https://github.com/jmineau/lair.git"
```

For everything else, pip is enough:

```bash
python3 -m venv lair-env
source lair-env/bin/activate
pip install "lair[requests,formats,geo,science] @ git+https://github.com/jmineau/lair.git"
```

The names in square brackets are the extras, groups of optional packages:

| Extra | Packages | Used by |
|---|---|---|
| `requests` | boto3, requests, s3fs, siphon | `hrrr`, `soundings`, `synoptic` |
| `formats` | zarr, numcodecs, fastkml, lxml | `hrrr`, `records.read_kml` |
| `geo` | cartopy, networkx, pyproj, rasterio, rioxarray, shapely | `geo`, `hrrr`, `inventories`, `transects` |
| `science` | molmass, scipy | `background`, `inventories`, `pollutants`, `transects` |
| `regridding` | xesmf (needs ESMF) | `geo` (regrid, resample), `inventories` (regrid, resample) |
| `complete` | all of the above | |

The first `import lair` in a new environment downloads one file, NOAA's CCG
filter (used by `lair.background`), from `ftp.gml.noaa.gov`. Without a network
`import lair` still works and only `lair.background` is unavailable. Set
`LAIR_SKIP_CCG_DOWNLOAD=1` to skip the download.

The [documentation](https://jmineau.github.io/lair/) has the rest: uv, a release
or no git, placing the CCG filter by hand, which `LAIR_*` variables each module
needs, and the steps for CHPC (University of Utah): `module load miniforge3`,
then the conda commands above. To change lair's code, see
[CONTRIBUTING.md](CONTRIBUTING.md), which also covers the
[pixi](https://pixi.sh) environment with every extra.

# Logging

`lair` logs through Python's standard `logging` module, under the `lair` logger
(one child logger per module, e.g. `lair.records`). Warnings are shown by
default; progress messages are logged at `INFO` and hidden unless you ask for
them:

```python
import logging
logging.basicConfig(level=logging.INFO)                 # everything, incl. lair
logging.getLogger("lair").setLevel(logging.INFO)        # or tune lair alone
```

# Acknowledgements

This package was partially inspired and uses some code generously provided by [Brian Blaylock's Carpenter Workshop python package](https://github.com/blaylockbk/Carpenter_Workshop).

# Use of AI coding agents

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science.

Portions of this package were also written with AI-based tools including
GitHub Copilot, ChatGPT, and Google Gemini, and some code snippets were
borrowed from StackOverflow and other online resources.

# Contributing

Contributions are welcome! Please take a look at current [issues](https://github.com/jmineau/lair/issues) and feel free to submit a pull request with new features or bug fixes.

# Citation

If you use any portion of this package in your research, please cite the software and/or acknowledge me.
Citation metadata is in [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository" button uses it), and each release is archived on [Zenodo](https://zenodo.org/) with a DOI.
