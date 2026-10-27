.. _data-locations:

Data locations
==============

``lair`` has no built-in data paths. Functions and classes that read from a
local data archive take an explicit directory argument and otherwise fall back
to an environment variable. The last column links to the folder layout each
variable expects:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Variable
     - Used by
     - Folder layout
   * - ``LAIR_INVENTORY_DIR``
     - ``lair.inventories`` (root with ``EDGAR/``, ``EPA/``, ``GFEI/``,
       ``vulcan/``, ``WetCHARTs/``)
     - :ref:`inventory-layouts`
   * - ``LAIR_SOUNDING_DIR``
     - ``lair.soundings`` (one subdirectory per station)
     - :ref:`The soundings example <soundings-data>`
   * - ``LAIR_CARBONTRACKER_DIR``
     - ``lair.noaa.CarbonTracker``
     - :ref:`The CarbonTracker example <carbontracker-data>`
   * - ``LAIR_GML_DIR``
     - ``lair.noaa.GMLData``
     - :class:`~lair.noaa.GMLData`, its Notes
   * - ``LAIR_CACHE_DIR``
     - cached results (default ``~/.cache/lair``)
     - :ref:`data-cache`

For example:

.. code-block:: bash

   export LAIR_INVENTORY_DIR=/path/to/inventories

Importing lair reads none of these folders. A variable is read when you call code
that needs it, so you can set it in a script or notebook after ``import lair``,
and a missing one raises a ``ValueError`` that names it. ``LAIR_CACHE_DIR`` is the
exception: it has a default, so a missing one is no error.

.. _inventory-layouts:

Inventory folders
-----------------

Each inventory has its own folder below ``LAIR_INVENTORY_DIR``. Its class page
says how the folder is laid out, which files lair reads, and where to download
them: :class:`~lair.inventories.EDGARv7`, :class:`~lair.inventories.EDGARv8`
(the monthly files), :class:`~lair.inventories.EPAv1`,
:class:`~lair.inventories.EPAv2`, :class:`~lair.inventories.GFEIv1`,
:class:`~lair.inventories.GFEIv2`, :class:`~lair.inventories.Vulcan` and
:class:`~lair.inventories.WetCHARTs`. :ref:`The inventory example
<inventory-data>` shows EDGAR v8's annual files.

What needs which data
---------------------

You can use lair with no data folder at all. These modules work on what you pass
in, so they need no ``LAIR_*`` variable: ``air``, ``clock``, ``constants``,
``meteorology``, ``pcaps``, ``plotter``, ``pollutants``, ``records``,
``transects``, ``parallel`` and ``utils``, and ``lair.units`` (the pint unit
registry). So does ``background``, which also needs the CCG filter and ``scipy``
(see :ref:`first-import-download`). Some of their functions need an extra.

These get data from somewhere else:

* ``inventories`` reads ``LAIR_INVENTORY_DIR``. lair does not download
  inventories: get them from their providers (EDGAR, EPA, GFEI, Vulcan,
  WetCHARTs).
* ``soundings`` uses ``LAIR_SOUNDING_DIR``. For when ``get_soundings``
  downloads, see :ref:`soundings-download`.
* ``noaa`` uses ``LAIR_CARBONTRACKER_DIR`` and ``LAIR_GML_DIR``. The
  ``download()`` methods of ``CarbonTracker`` and ``GMLData`` fetch the files
  from NOAA's FTP server into them.
* ``hrrr`` reads a public Amazon S3 bucket. It needs the internet and the
  ``requests``, ``formats`` and ``geo`` extras, and no folder.
* ``synoptic`` needs a Synoptic Data API token, passed as ``token=`` or set as
  ``SYNOPTIC_TOKEN``, and the ``requests`` extra.

Spelling a pollutant
--------------------

Spell a pollutant the way its folder, its file or its formula is spelled. Only a
label ignores case. Of the inventories, only EDGAR takes a pollutant: EPA, GFEI
and WetCHARTs hold methane, and Vulcan carbon dioxide.

* **In a folder or file name, the case must match.** ``EDGARv8("CH4")`` reads
  ``EDGAR/v8/CH4/``, the folder as EDGAR names it. ``EDGARv8("ch4")`` looks for
  ``EDGAR/v8/ch4/``, and on Linux, where folder names are case-sensitive, raises
  ``FileNotFoundError``. ``EDGARv7`` is the same. ``GMLData("ch4", "uta")`` builds
  the file name NOAA uses, ``ch4_uta_surface-flask_1_ccgg_event.txt``, so
  ``"CH4"`` or ``"UTA"`` names a file that does not exist. A CarbonTracker
  version is a folder name too, as NOAA writes it: ``CT-CH4-2025``, not
  ``ct-ch4-2025``.
* **A formula needs its capitals.**
  :func:`~lair.inventories.molecular_weight` takes ``"CH4"`` and raises a
  ``FormulaError`` for ``"ch4"``. NOx is the exception: any case gives the mass
  of NO2. An ``Inventory`` built from data you pass in
  upper-cases a pollutant that has no capital letter (``"ch4"`` becomes
  ``"CH4"``) and keeps one that has any as you typed it (``"Ch4"`` stays, and
  fails when lair needs its formula).
* **A label ignores case.** :func:`lair.pollutants.get_pollutant` takes
  ``"ch4"``, ``"CH4"``, ``"nox"`` and ``"PM25"``; it reads no folder.

.. _data-cache:

What lair caches
----------------

One thing: the ``molefractions`` attribute of :class:`~lair.noaa.CarbonTrackerCH4`
and :class:`~lair.noaa.CarbonTrackerCO2`, which opens every file of a version at
once. ``sample()`` and ``background()`` save nothing, and nothing else in lair
writes to the cache folder. The cache is a file in ``LAIR_CACHE_DIR``, with a
hidden index file beside it:

.. code-block:: text

   $LAIR_CACHE_DIR/
     carbontracker/
       ch4/CT-CH4-2025/molefractions.pkl
       co2/CT2019B/molefractions_nam1x1.pkl

lair makes the folder the first time it saves something there, not when it is
imported, so ``import lair`` works with a read-only home directory. If the folder
cannot be made, the error names it and ``LAIR_CACHE_DIR``: set the variable to a
folder you can write to, or pass ``cache=False``, which saves nothing.

The file holds how to open the files, not their values: 8 to 15 KB for each daily
file, so 3.2 MB for the 366 days of a year, whose files add up to 13 GB. That
needs one file for each day (:ref:`carbontracker-data` says what happens when
a day has two).

The file is for one list of files. When the folder gains or loses a file, the next
``molefractions`` adds an entry for the new list, as large as the first, to the
same file, and the old entry stays. To start again, delete the version's folder in
the cache; lair makes it again the next time.

Measured on 2026-10-14, with files of CT-CH4-2025 (36 MB each), CT-CH4-2023 and
CT2019B (18 MB).
