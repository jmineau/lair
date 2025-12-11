<div align=center>
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="/docs/source/_static/lair_fordark_r.png">
    <source media="(prefers-color-scheme: light)" srcset="/docs/source/_static/lair_forlight_r.png">
    <img alt="lair logo" src="/docs/source/_static/lair_fordark_r.png">
</picture>

# Land-Air Interactions Research

## [:scroll: Documentation](https://jamesmineau.chpc.utah.edu/lair)

</div>

`lair` is a collection of tools that I have developed/acquired for my research regarding land-air interactions. The `lair` package is designed to make it easier to work with atmospheric data.

# Installation

The `lair` package is installable from the git repository via `pip`, however, some dependencies can only be installed via `conda`.
Additionally, many components of `lair` require access to CHPC which encourages the use of `conda`. Therefore, we recommend using `conda` to install the package.

> If you are using CHPC, it is assumed that `miniforge3` is installed following the instructions at https://www.chpc.utah.edu/documentation/software/python-anaconda.php

To create a new conda environment for `lair`, use the following command:

```bash
mamba create -n lair -c conda-forge python=3.10 esmpy
```

If you already have a conda environment, simply install the dependencies:

```bash
mamba activate <lair-env>
mamba install -c conda-forge esmpy
```

> `lair` requires Python 3.10 or higher.

Now we can install the package via `pip`. Either directly from the git repository:

```bash
pip install git+https://github.com/jmineau/lair.git
```

or by cloning the repository and installing it as an editable package:

```bash
git clone https://github.com/jmineau/lair.git
cd lair
pip install -e .
```

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