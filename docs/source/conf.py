# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.abspath('../..'))

import lair

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

now = dt.datetime.now(dt.timezone.utc)

project = 'LAIR'
copyright = f'{now: %Y}, James Mineau | Last Updated: {now: %B %d, %Y}'
author = 'James Mineau'

release = lair.__version__

# Get the version from the environment if it's set
# This allows me to set the version to 'dev' when autobuilding dev docs
version = os.getenv('LAIR_DOCS_VERSION', release)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'myst_parser',
    'numpydoc',  # needs to be loaded *after* autodoc
    'sphinx_design'
]

# Set autodoc defaults
autodoc_default_options = {
    'members': True,  # Document all functions/members
    'member-order': 'bysource',  # Order members by source order
    'special-members': '__init__',
    'autosummary': True,  # Include a members 'table of contents'
}
autodoc_mock_imports = [
    'boto3',
    'cartopy',
    'cf-xarray',
    'fastkml',
    'geopandas',
    'matplotlib',
    'metpy',
    'numpy',
    'pandas',
    'pint',
    'pint-xarray',
    'pyproj',
    'rioxarray',
    's3fs',
    'scipy',
    'shapely',
    'siphon',
    'synopticpy',
    'tables',
    'xarray',
    'xesmf',
    'zarr'
]
autodoc_typehints = "signature"

autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Set up mapping for other projects' docs
intersphinx_mapping = {
    'boto3': ('https://boto3.amazonaws.com/v1/documentation/api/latest', None),
    'cartopy': ('https://scitools.org.uk/cartopy/docs/latest', None),
    'cf-xarray': ('https://cf-xarray.readthedocs.io/en/latest', None),
    'dask': ('https://docs.dask.org/en/stable', None),
    'esmpy': ('https://earthsystemmodeling.org/esmpy_doc/release/latest/html', None),
    'fastkml': ('https://fastkml.readthedocs.io/en/latest', None),
    'geopandas': ('https://geopandas.org/en/stable', None),
    'metpy': ('https://unidata.github.io/MetPy/latest', None),
    'matplotlib': ('https://matplotlib.org/stable', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'pint': ('https://pint.readthedocs.io/en/stable', None),
    'pint-xarray': ('https://pint-xarray.readthedocs.io/en/stable', None),
    'pyproj': ('https://pyproj4.github.io/pyproj/stable', None),
    'python': ('https://docs.python.org/3', None),
    'rioxarray': ('https://corteva.github.io/rioxarray/stable', None),
    's3fs': ('https://s3fs.readthedocs.io/en/latest', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable', None),
    'siphon': ('https://unidata.github.io/siphon/latest', None),
    'synopticpy': ('https://synopticpy.readthedocs.io/en/stable', None),
    'tables': ('https://www.pytables.org', None),
    'xarray': ('https://docs.xarray.dev/en/stable', None),
    'xesmf': ('https://xesmf.readthedocs.io/en/stable', None),
    'zarr': ('https://zarr.readthedocs.io/en/stable', None)
}

numpydoc_show_class_members = False
numpydoc_attributes_as_param_list = False

todo_include_todos = True

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = [
    'build',
    'Thumbs.db',
    '.DS_Store',
    '.ipynb_checkpoints',
    '.vscode',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
     'github_url': 'https://github.com/jmineau/lair',
     'header_links_before_dropdown': 7,
     'icon_links': [
         {
             'name': 'James Mineau',
             'url': 'https://jamesmineau.chpc.utah.edu',
             'icon': 'fas fa-user'
         }
        ],
     'logo': {
         'image_light': '_static/lair_forlight_r.png',
         'image_dark': '_static/lair_fordark_r.png',
         'alt_text': 'LAIR - Home'
      },
     'navbar_start': ['navbar-logo', 'version-switcher'],
    #  'show_version_warning_banner': True,
     'switcher': {
         'json_url': 'https://raw.githubusercontent.com/jmineau/lair/main/docs/versions.json',
         'version_match': version
     }
}

html_static_path = ['_static']
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
]
