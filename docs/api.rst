API Reference
=============

``lair`` is a flat collection of modules; import the one you need, e.g.
``from lair.meteorology import poisson``. Some modules need an optional
dependency group (see :ref:`optional-dependencies`).

Atmosphere & Meteorology
------------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   lair.air
   lair.background
   lair.meteorology
   lair.pcaps
   lair.soundings

Data Sources
------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   lair.hrrr
   lair.inventories
   lair.noaa

Measurements
------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   lair.transects

Geospatial & Plotting
---------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   lair.geo
   lair.plotter

Utilities
---------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   lair.clock
   lair.config
   lair.constants
   lair.dev
   lair.parallel
   lair.records
   lair.utils
