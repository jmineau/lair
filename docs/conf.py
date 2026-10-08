"""
Sphinx configuration for the lair docs.

The full list of settings: https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import datetime as dt
import os
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "_ext"))  # api_pages

# Never reach for the NOAA GML FTP download (CCG filter) while building docs;
# lair._ccg_filter is mocked below instead.
os.environ.setdefault("LAIR_SKIP_CCG_DOWNLOAD", "1")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "LAIR"
copyright = f"{dt.date.today():%Y}, James Mineau"
author = "James Mineau"


def _detect_release() -> str:
    # setuptools-scm writes the version into the installed package metadata
    try:
        return package_version("lair")
    except PackageNotFoundError:
        return "0+unknown"


release = _detect_release()
version = release
# Builds from main (and local builds) are "dev"; release builds are their version.
version_match = "dev" if (".dev" in release or "+" in release) else release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "api_pages",  # _ext/api_pages.py: class pages with member tables
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

# -- Extension configuration -------------------------------------------------

# Docstrings are numpydoc style
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
napoleon_use_ivar = True  # what a class page's tables leave in "Attributes"

# Members are documented on their own pages (_templates/autosummary/), not by
# autodoc's :members:.
autodoc_default_options = {
    "member-order": "bysource",
}
autodoc_typehints = "description"

# Optional extras (and the generated CCG filter) are mocked so the docs build in
# the lean uv environment. Core dependencies are imported for real.
autodoc_mock_imports = [
    "boto3",
    "botocore",
    "cartopy",
    "fastkml",
    "lair._ccg_filter",
    "molmass",
    "networkx",
    "numcodecs",
    "pyproj",
    "rasterio",
    "requests",
    "rioxarray",
    "s3fs",
    "scipy",
    "shapely",
    "siphon",
    "xesmf",
    "zarr",
]

autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "pint": ("https://pint.readthedocs.io/en/stable", None),
}

todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_title = f"lair {version_match}"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/jmineau/lair",
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # The version dropdown. The Documentation workflow publishes dev/ (main),
    # one folder per release and stable/, and writes switcher.json listing them.
    "switcher": {
        "json_url": "https://jmineau.github.io/lair/switcher.json",
        "version_match": version_match,
    },
    "check_switcher": False,  # switcher.json exists only on the deployed site
    "show_version_warning_banner": True,  # point old versions at stable
    "logo": {
        "image_light": "_static/lair_forlight_r.png",
        "image_dark": "_static/lair_fordark_r.png",
        "alt_text": "LAIR - Home",
    },
}
