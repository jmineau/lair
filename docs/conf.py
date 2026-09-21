# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime as dt
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

sys.path.insert(0, os.path.abspath('..'))

# Never reach for the NOAA GML FTP download (CCG filter) while building docs;
# lair._ccg_filter is mocked below instead.
os.environ.setdefault('LAIR_SKIP_CCG_DOWNLOAD', '1')

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LAIR'
copyright = f'{dt.date.today():%Y}, James Mineau'
author = 'James Mineau'


def _detect_release() -> str:
    try:
        return package_version('lair')
    except PackageNotFoundError:
        pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
        match = re.search(r'^version\s*=\s*"([^"]+)"',
                          pyproject.read_text(encoding='utf-8'),
                          flags=re.MULTILINE)
        return match.group(1) if match else '0+unknown'


release = _detect_release()
version = release

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.ipynb_checkpoints']

# -- Extension configuration -------------------------------------------------

# Docstrings are numpydoc style
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

autodoc_default_options = {
    'member-order': 'bysource',
    'exclude-members': '__weakref__',
}
autodoc_typehints = 'description'

# Optional extras (and the generated CCG filter) are mocked so the docs build in
# the lean uv environment. Core dependencies are imported for real.
autodoc_mock_imports = [
    'boto3',
    'botocore',
    'cartopy',
    'fastkml',
    'lair._ccg_filter',
    'molmass',
    'networkx',
    'numcodecs',
    'pyproj',
    'rasterio',
    'requests',
    'rioxarray',
    's3fs',
    'scipy',
    'shapely',
    'siphon',
    'xesmf',
    'zarr',
]

autosummary_generate = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'pint': ('https://pint.readthedocs.io/en/stable', None),
}

todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_theme_options = {
    'github_url': 'https://github.com/jmineau/lair',
    'show_toc_level': 2,
    'navbar_align': 'left',
    'logo': {
        'image_light': '_static/lair_forlight_r.png',
        'image_dark': '_static/lair_fordark_r.png',
        'alt_text': 'LAIR - Home',
    },
}
