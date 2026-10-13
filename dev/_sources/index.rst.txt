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

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.22900040.svg
   :target: https://doi.org/10.5281/zenodo.22900040
   :alt: DOI

``lair`` is a collection of tools that I have developed/acquired for my research
regarding land-air interactions. It is designed to make it easier to work with
atmospheric data: meteorological calculations, emissions inventories, NOAA
greenhouse-gas products, soundings, HRRR winds, background estimation, mobile
transects, geospatial helpers, and plotting.

New to lair? Install it (below), then start with the :doc:`quickstart`.

.. toctree::
   :maxdepth: 2
   :hidden:

   Quickstart <quickstart>
   Examples <examples/index>
   API Reference <api>

.. _installation:

Installation
------------

``lair`` is installed from GitHub, not from PyPI. On PyPI, ``lair`` is an
unrelated project (a helper for Flask web services), so ``pip install lair``
installs the wrong package.

You need Python 3.11 or newer and `git <https://git-scm.com/>`_ (pip uses git to
download the code). Install lair into a new environment, so that its packages
stay apart from the rest of your work.

The commands on this page were run on 64-bit Linux. lair's own tests also run on
macOS and Windows, without the extras.

Pick a route
^^^^^^^^^^^^

Grid-cell areas (``geo.gridcell_area``) and regridding (``geo.regrid``,
``geo.resample``) need ESMF, a compiled library that pip cannot install. The
inventories use both, so ``Inventory.integrate()`` (the total of an inventory
over a region) and ``Inventory.absolute_emissions`` need ESMF too.

* To use the inventories or to regrid, use **conda**.
* For everything else, **pip** is enough.

conda (everything)
^^^^^^^^^^^^^^^^^^

conda is a package manager. It installs Python itself, and compiled libraries
such as ESMF, into separate environments. ``mamba`` is a faster command with the
same arguments. Both come with `Miniforge <https://conda-forge.org/download/>`_.

.. code-block:: bash

   mamba create -n lair -c conda-forge python=3.12 esmpy "esmf=*=nompi*"
   conda activate lair
   pip install "lair[complete] @ git+https://github.com/jmineau/lair.git"

The first line makes an environment named ``lair`` with Python and ESMF. It asks
for the no-MPI build of ESMF: lair does not use MPI, and the MPI build can print
errors when it starts on a machine where MPI is not set up (this happened on a
compute node of a cluster). lair is not on conda-forge, so the last line uses
pip. It installs lair and every extra (see :ref:`optional-dependencies`).

pip (everything except ESMF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``venv`` comes with Python. It makes a folder that holds a private copy of the
packages you install.

.. code-block:: bash

   python3 -m venv lair-env
   source lair-env/bin/activate
   pip install "lair[requests,formats,geo,science] @ git+https://github.com/jmineau/lair.git"

If ``python3 --version`` shows less than 3.11, use ``python3.12`` (or another
version you have) in the first line. On Windows, activate with
``lair-env\Scripts\activate``.

`uv <https://docs.astral.sh/uv/>`_ does the same job as ``venv`` and pip, faster,
and can fetch the Python version for you:

.. code-block:: bash

   uv venv --python 3.12 lair-env
   source lair-env/bin/activate
   uv pip install "lair[requests,formats,geo,science] @ git+https://github.com/jmineau/lair.git"

``lair[complete]`` also installs with pip, but ``import xesmf`` then fails with
``No module named 'ESMF'``, and so do the functions listed above.

A release, or no git
^^^^^^^^^^^^^^^^^^^^

To install a release instead of the newest code, add ``@`` and its tag to the
address (the tags are on the `Releases
<https://github.com/jmineau/lair/releases>`_ page):

.. code-block:: bash

   pip install "lair[geo,science] @ git+https://github.com/jmineau/lair.git@v2026.12.8"

If git is not installed, pip can install from a zip of the repository instead:

.. code-block:: bash

   pip install "lair[geo,science] @ https://github.com/jmineau/lair/archive/refs/heads/main.zip"

From a clone of the repository, `pixi <https://pixi.sh>`_ builds an environment
with every extra, ESMF included, from ``pixi.lock``
(``pixi install``; 64-bit Linux only). To change lair's code, see
`CONTRIBUTING.md <https://github.com/jmineau/lair/blob/main/CONTRIBUTING.md>`_.

.. _optional-dependencies:

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

The core install is light. Modules that need heavier packages import them only
when used; name the matching extras in square brackets, as in the commands above:

============== ====================================== ===========================================
Extra          Packages                               Used by
============== ====================================== ===========================================
``requests``   boto3, requests, s3fs, siphon          ``hrrr``, ``soundings``, ``synoptic``
``formats``    zarr, numcodecs, fastkml, lxml         ``hrrr``, ``records.read_kml``
``geo``        cartopy, networkx, pyproj, rasterio,   ``geo``, ``hrrr``, ``inventories``,
               rioxarray, shapely                     ``transects``
``science``    molmass, scipy                         ``background``, ``inventories``,
                                                      ``pollutants``, ``transects``
``regridding`` xesmf                                  ``geo`` (areas, regrid, resample),
                                                      ``inventories`` (totals, regrid)
``complete``   all of the above
============== ====================================== ===========================================

``regridding`` installs the ``xesmf`` package, which works only with ESMF in the
same environment: see "conda" above.

Check the install
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -c "import lair; print(lair.__version__)"

This prints the version, for example ``2026.12.8``. The first import in a new
environment also downloads a file; the next section says which.

The first import downloads a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``lair.background`` uses the CCG filter, curve-fitting code that NOAA's Global
Monitoring Laboratory publishes. It is not part of lair's repository. The first
time you ``import lair`` in an environment, lair downloads NOAA's
``ccg_filter.zip`` from ``ftp.gml.noaa.gov`` and saves ``ccg_filter.py`` from it
as ``_ccg_filter.py`` inside the installed ``lair`` folder. Each environment
downloads it once. The filter also needs ``scipy`` (the ``science`` extra).

Only ``lair.background`` uses the file. Without it, the rest of lair works as
usual. There are three ways to be without it:

* **No network.** ``import lair`` still works. It logs a warning, ``Could not
  install NOAA's CCG filter (...). lair.background is unavailable until
  lair.setup_ccg_filter() succeeds.``, and ``import lair.background`` raises an
  ``ImportError`` that says the filter is not installed. The next ``import
  lair`` tries the download again.
* **Skip the download.** Set ``LAIR_SKIP_CCG_DOWNLOAD`` before importing, for
  example in a script that runs offline:

  .. code-block:: bash

     export LAIR_SKIP_CCG_DOWNLOAD=1

  Any text counts, including ``0``. An empty value does not.
* **Place the file yourself,** for example when your network allows HTTPS but
  not FTP. NOAA serves the same zip over HTTPS. The last two lines copy the file
  into the installed ``lair`` folder (this does not import lair):

  .. code-block:: bash

     curl -O https://gml.noaa.gov/aftp/user/thoning/ccgcrv/ccg_filter.zip
     python -m zipfile -e ccg_filter.zip ccg_filter
     LAIR_DIR=$(python -c "import importlib.util as u, os; print(os.path.dirname(u.find_spec('lair').origin))")
     cp ccg_filter/ccg_filter.py "$LAIR_DIR/_ccg_filter.py"

On CHPC (University of Utah)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The steps above work on CHPC's general-environment clusters (tested on
Notchpeak), on a login node or in a job: compute nodes can reach GitHub, PyPI and
NOAA. Use conda, which CHPC provides as the ``miniforge3`` module:

.. code-block:: bash

   module load miniforge3
   mamba create -n lair -c conda-forge python=3.12 esmpy "esmf=*=nompi*"
   conda activate lair
   pip install "lair[complete] @ git+https://github.com/jmineau/lair.git"

* ``python3`` on CHPC is Python 3.6, too old for lair. The environment brings
  its own Python.
* Run ``module load miniforge3`` in every new shell and in every job script
  before ``conda activate lair``. CHPC advises against loading it automatically
  at login, because that breaks remote desktop sessions (FastX and OnDemand). See
  CHPC's `Python and Anaconda
  <https://www.chpc.utah.edu/documentation/software/python-anaconda.php>`_ page.
* ``mamba create`` may print two ``Could not open lockfile`` errors. They come
  from the shared module, and the environment is made anyway.
* The environment is about 1.3 GB and goes in ``~/.conda/envs``, which counts
  against your home directory's quota. To put it elsewhere, use
  ``mamba create -p /path/to/lair-env ...`` and ``conda activate
  /path/to/lair-env``.
* For a lighter environment without ESMF (about 0.6 GB), run ``module load uv``
  and then the ``uv`` commands above. pixi is not a CHPC module.
* Your group may already keep copies of the data (see below). Ask where, and
  point the ``LAIR_*`` variables at them.

``import cartopy`` prints a PROJ error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In an environment built as above, ``import cartopy`` prints nothing. In a
conda-forge environment that also has ``fiona`` (cartopy loads it when it is
there), you may see

.. code-block:: text

   ERROR 1: PROJ: proj_create_from_database: Open of /path/to/env/share/proj failed

when you run that environment's ``python`` without activating the environment:
by its full path, from a job script, or in a notebook kernel. The message comes
from GDAL, which ``fiona`` loads, and it is harmless: coordinates transform to
the same numbers with and without it. ``conda activate`` removes it. Without
activating, set ``PROJ_DATA`` to the environment's ``share/proj`` folder.

.. _data-locations:

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

Importing lair reads none of these folders. A variable is read when you call code
that needs it, and a missing one raises a ``ValueError`` that names it.
``LAIR_CACHE_DIR`` is the exception twice over: it has a default, and lair reads
it when it is imported, so set it before ``import lair``. lair makes the folder
the first time it saves something there (only ``lair.noaa``'s CarbonTracker
classes do), not when it is imported, so ``import lair`` works with a read-only
home directory. If the folder cannot be made, the error names it and
``LAIR_CACHE_DIR``: set the variable to a folder you can write to, or pass
``cache=False``.

What needs which data
^^^^^^^^^^^^^^^^^^^^^

You can use lair with no data folder at all. These modules work on what you pass
in, so they need no ``LAIR_*`` variable: ``air``, ``clock``, ``constants``,
``meteorology``, ``pcaps``, ``plotter``, ``pollutants``, ``records``,
``transects``, ``parallel`` and ``utils``, and ``lair.units`` (the pint unit
registry). So does ``background``, which also needs the CCG filter and ``scipy``
(see above). Some of their functions need an extra.

These get data from somewhere else:

* ``inventories`` reads ``LAIR_INVENTORY_DIR``. lair does not download
  inventories: get them from their providers (EDGAR, EPA, GFEI, Vulcan,
  WetCHARTs).
* ``soundings`` uses ``LAIR_SOUNDING_DIR``. ``get_soundings`` and
  ``download_sounding`` fetch missing soundings from the University of Wyoming
  into one subfolder per station, so an empty folder is enough to start.
* ``noaa`` uses ``LAIR_CARBONTRACKER_DIR`` and ``LAIR_GML_DIR``. The
  ``download()`` methods of ``CarbonTracker`` and ``GMLData`` fetch the files
  from NOAA's FTP server into them.
* ``hrrr`` reads a public Amazon S3 bucket. It needs the internet and the
  ``requests``, ``formats`` and ``geo`` extras, and no folder.
* ``synoptic`` needs a Synoptic Data API token, passed as ``token=`` or set as
  ``SYNOPTIC_TOKEN``, and the ``requests`` extra.

Logging
-------

``lair`` logs through Python's standard :mod:`logging` module, under the
``lair`` logger (one child logger per module, e.g. ``lair.records``). Warnings
are shown by default; progress messages are logged at ``INFO`` and hidden
unless you ask for them:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.INFO)            # everything, incl. lair
   logging.getLogger("lair").setLevel(logging.INFO)   # or tune lair alone

``lair.background.thoning_filter`` prints the CCG filter's diagnostics when the
``lair.background`` logger is at ``DEBUG``.

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
