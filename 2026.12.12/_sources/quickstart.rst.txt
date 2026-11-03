Quickstart
==========

This page takes you from installing lair to a first result in a few minutes.

Install
-------

lair is installed from GitHub, not from PyPI: :ref:`installation` has the
commands and says which route (conda or pip) your work needs.

Point lair at your data
-----------------------

Most modules need no data folder. They work on the numbers you pass in.

The modules that read files you keep on disk (``inventories``, ``soundings``,
and ``noaa``'s ``CarbonTracker`` and ``GMLData``) take the folder as an argument,
or read it from a ``LAIR_*`` environment variable. :ref:`data-locations` lists
the variables, shows how to set one and says what happens when one is missing.

A first calculation
-------------------

The potential temperature of a sounding, with
:func:`lair.meteorology.poisson`. The sounding here is made up: a temperature at
three pressure levels.

.. ipython:: python

   import xarray as xr
   from lair.meteorology import poisson

   levels = {"level": [850, 700, 500]}  # hPa, as labels
   p = xr.DataArray([85000.0, 70000.0, 50000.0], coords=levels, dims="level")  # Pa
   T = xr.DataArray([275.0, 265.0, 250.0], coords=levels, dims="level")  # K
   poisson(T, p)

The rule in :mod:`lair.meteorology` is plain SI in, plain SI out: pressure in
pascals, temperature in kelvin, and the result in kelvin. So 850 hPa goes in as
85000. If your values carry units, pass them as quantities of ``lair.units``
(the `pint <https://pint.readthedocs.io>`_ unit registry; :doc:`examples/units`
converts a flux from moles to mass with it). lair converts them to SI on the way
in, and the result is still a plain number in kelvin:

.. ipython:: python

   from lair import units

   poisson(units.Quantity(2, "degC"), 850 * units.hPa)

Plain numbers are the rule because the same function then works on a float, a
numpy array, a pandas column or an xarray ``DataArray``, and most of those
carry no units.

Where next
----------

* :doc:`examples/index`: worked examples, one per job.
* :doc:`api`: every module, with what each function and class does. In Python,
  ``help(lair.meteorology)`` shows the same text.
* :ref:`optional-dependencies`: which extra a module needs.
