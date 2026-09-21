LAIR: Land-Air Interactions Research
====================================

.. image:: https://github.com/jmineau/lair/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/jmineau/lair/actions/workflows/tests.yml
   :alt: Tests

.. image:: https://github.com/jmineau/lair/actions/workflows/docs.yml/badge.svg
   :target: https://github.com/jmineau/lair/actions/workflows/docs.yml
   :alt: Documentation

.. image:: https://github.com/jmineau/lair/actions/workflows/quality.yml/badge.svg
   :target: https://github.com/jmineau/lair/actions/workflows/quality.yml
   :alt: Code Quality

``lair`` is a collection of tools that I have developed/acquired for my research
regarding land-air interactions. It is designed to make it easier to work with
atmospheric data: meteorological calculations, emissions inventories, NOAA
greenhouse-gas products, soundings, HRRR winds, background estimation, mobile
transects, geospatial helpers, and plotting.

.. toctree::
   :maxdepth: 2
   :hidden:

   API Reference <api>

Installation
------------

Install from the git repository with ``pip``:

.. code-block:: bash

   pip install git+https://github.com/jmineau/lair.git

or clone the repository and install it as an editable package:

.. code-block:: bash

   git clone https://github.com/jmineau/lair.git
   cd lair
   pip install -e .

``lair`` requires Python 3.10 or higher.

.. _optional-dependencies:

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

The core install is light. Modules that need heavier packages import them only
when used; install the matching extra (e.g. ``pip install "lair[geo,science]"``):

============== ====================================== ===========================================
Extra          Packages                               Used by
============== ====================================== ===========================================
``requests``   boto3, requests, s3fs, siphon          ``hrrr``, ``soundings``
``formats``    zarr, numcodecs, fastkml, lxml         ``hrrr``, ``records.read_kml``
``geo``        cartopy, networkx, pyproj, rasterio,   ``geo``, ``hrrr``, ``inventories``,
               rioxarray, shapely                     ``transects``
``science``    molmass, scipy                         ``background``, ``inventories``,
                                                      ``transects``
``regridding`` xesmf                                  ``geo`` (regrid / resample)
``complete``   all of the above
============== ====================================== ===========================================

``xesmf`` needs the ESMF library, which is easiest to get from conda-forge:

.. code-block:: bash

   mamba install -c conda-forge esmpy

Data locations
--------------

``lair`` has no built-in data paths. Functions and classes that read from a
local data archive take an explicit directory argument and otherwise fall back
to an environment variable:

============================ ==================================================
Variable                     Used by
============================ ==================================================
``LAIR_INVENTORY_DIR``       ``lair.inventories`` (root with ``EDGAR/``,
                             ``EPA/``, ``GFEI/``, ``vulcan/``, ``WetCHARTs/``)
``LAIR_SOUNDING_DIR``        ``lair.soundings`` (one subdirectory per station)
``LAIR_CARBONTRACKER_DIR``   ``lair.noaa.CarbonTracker``
``LAIR_GML_DIR``             ``lair.noaa.GMLData``
``LAIR_CACHE_DIR``           cached results (default ``~/.cache/lair``)
============================ ==================================================

For example:

.. code-block:: bash

   export LAIR_INVENTORY_DIR=/path/to/inventories

Verbosity
---------

Verbosity is set via ``lair.config.verbose`` as a boolean:

.. code-block:: python

   import lair
   lair.config.verbose = False

For early versions of the package ``verbose`` defaults to ``True``; this will
change in a future version.

Acknowledgements
----------------

This package was partially inspired by, and uses some code generously provided
by, Brian Blaylock's `Carpenter Workshop
<https://github.com/blaylockbk/Carpenter_Workshop>`_. Background filtering wraps
NOAA GML's `CCG curve-fitting code
<https://gml.noaa.gov/ccgg/mbl/crvfit/crvfit.html>`_.

Disclaimer
----------

* Portions of this package were written with AI-based tools including GitHub
  Copilot, ChatGPT, and Google Gemini.
* Various code snippets were borrowed from StackOverflow and other online
  resources.

Contributing
------------

Contributions are welcome! Please take a look at the current `issues
<https://github.com/jmineau/lair/issues>`_ and feel free to submit a pull request
with new features or bug fixes. Please document your code using `numpydoc
<https://numpydoc.readthedocs.io/en/latest/format.html>`_ style docstrings.

Citation
--------

If you use any portion of this package in your research, please cite the
software and/or acknowledge me.
