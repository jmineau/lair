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
release = lair.VERSION

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
    'fastkml',
    'geopandas',
    'matplotlib',
    'metpy',
    'numpy',
    'pandas',
    'pyproj',
    'rioxarray',
    's3fs',
    'scipy',
    'shapely',
    'siphon',
    'synopticpy',
    'tables',
    'xarray',
    'zarr'
]
autodoc_typehints = "signature"

autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Set up mapping for other projects' docs
intersphinx_mapping = {
    'cartopy': ('https://scitools.org.uk/cartopy/docs/latest/', None),
    'fastkml': ('https://fastkml.readthedocs.io/en/latest/', None),
    'geopandas': ('https://geopandas.org/en/stable/', None),
    'metpy': ('https://unidata.github.io/MetPy/latest/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'pyproj': ('https://pyproj4.github.io/pyproj/stable/', None),
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'rioxarray': ('https://corteva.github.io/rioxarray/stable/', None),
    's3fs': ('https://s3fs.readthedocs.io/en/latest/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'shapely': ('https://shapely.readthedocs.io/en/stable/', None),
    'siphon': ('https://unidata.github.io/siphon/latest/', None),
    'synopticpy': ('https://synopticpy.readthedocs.io/en/stable/', None),
    'tables': ('https://www.pytables.org/', None),
    'xarray': ('https://xarray.pydata.org/en/stable/', None),
    'zarr': ('https://zarr.readthedocs.io/en/stable/', None)
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
     "header_links_before_dropdown": 7,
     'icon_links': [
         {
             'name': 'James Mineau',
             'url': 'https://jamesmineau.chpc.utah.edu',
             'icon': 'fas fa-user'
         }
        ],
     'logo': {
         'text': f'LAIR {lair.VERSION} docs',
         'alt_text': 'LAIR - Home'
     }
}

html_static_path = ['_static']
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.1.1/css/all.min.css"
]
