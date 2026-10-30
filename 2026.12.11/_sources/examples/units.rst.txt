Convert a methane flux from moles to mass
=========================================

This example converts a methane flux of 10 nmol m-2 s-1 (nanomoles per square
meter per second) to kg km-2 d-1 (kilograms per square kilometer per day), the
unit lair's inventories convert to when you ask them to standardize their units.
Two rules come out of it:

* **Moles and mass need a molecular weight.** A flux in moles and a flux in mass
  are different dimensions, so pint converts between them only inside the
  ``mass_flux`` context, given ``mw``: the molecular weight, as a quantity in
  g/mol. Without the context the conversion raises ``DimensionalityError``;
  inside it, a missing ``mw``, or one that is a plain number, raises a
  ``TypeError`` that names ``mw`` and shows what to pass.
* **pint reads** ``kt`` **as a knot**, a unit of speed. For a mass of a thousand
  tonnes, write ``Gg`` or ``kilotonne``.

This page runs when the docs build.

The units
---------

``lair.units`` is a `pint <https://pint.readthedocs.io>`_ unit registry (the one
pint-xarray uses). A quantity is a number with a unit, and pint reads the unit
from text. Units written the way CF files write them parse too, so ``m-2 s-1``
works. In the output, ``~P`` asks pint for the short symbols:

.. ipython:: python

   from lair import units

   flux = 10 * units("nmol m-2 s-1")
   print(f"{flux:~P}")

Moles to mass
-------------

Ask for the flux in mass and pint refuses: it cannot get from moles to
kilograms without knowing which molecule the moles are of.

.. ipython:: python

   try:
       flux.to("kg km-2 d-1")
   except Exception as err:
       print(f"{type(err).__name__}: {err}")

The ``mass_flux`` context supplies the missing piece. Importing lair adds it to
the registry. Its ``mw`` is the molar mass of the gas, with its unit: 16.04 g/mol
for methane.

.. ipython:: python

   mw = 16.04 * units("g/mol")
   with units.context("mass_flux", mw=mw):
       flux_kg = flux.to("kg km-2 d-1")
   print(f"{flux_kg:~P.2f}")

The same context goes the other way, from mass to moles. This returns the
10 nmol m-2 s-1 it started from:

.. ipython:: python

   with units.context("mass_flux", mw=mw):
       back = flux_kg.to("nmol m-2 s-1")
   print(f"{back:~P.2f}")

To check the result by hand, convert one unit at a time. 10 nmol is 10e-9 mol,
which weighs 10e-9 mol x 16.04 g/mol in grams, and 1000 g is 1 kg. Per square
meter becomes per square kilometer by multiplying by 1e6, since a square
kilometer is 1e6 square meters. Per second becomes per day by multiplying by
86,400, the seconds in a day (``d`` is exactly 24 hours, whatever the local
clock does; see the :doc:`clock page <clock>` for times):

.. ipython:: python

   print(f"{10e-9 * 16.04 / 1000 * 1e6 * 86400:.2f}")

An array
--------

A grid of fluxes usually carries its unit as a ``units`` attribute, in CF form.
``.pint.quantify()`` reads it, ``.pint.to()`` converts, and ``.pint.dequantify()``
writes the new unit back into the attribute:

.. ipython:: python

   import xarray as xr

   grid = xr.DataArray(
       [[10.0, 20.0], [30.0, 40.0]],
       dims=("y", "x"),
       attrs={"units": "nmol m-2 s-1"},
   )
   with units.context("mass_flux", mw=mw):
       grid_kg = grid.pint.quantify().pint.to("kg km-2 d-1")
   grid_kg.pint.dequantify()

.. _unit-spellings:

The same unit has more than one spelling, and pint reads them alike: ``kg km-2
d-1`` and ``kilogram / kilometer ** 2 / day`` (what ``dequantify()`` writes) are
one unit, and so are ``kg m-2 s-1`` and ``kg/m**2/s`` (the ``units`` attribute of
an inventory's ``total_emissions``).

:func:`lair.inventories.convert_units` does the same for any array or dataset
with a ``units`` attribute, given the pollutant instead of ``mw``:
``convert_units(grid, "CH4", "kg km-2 d-1")`` returns a plain array with the new
values and ``units`` attribute. It needs the ``geo`` and ``science`` extras (see
:ref:`optional-dependencies`). An :class:`~lair.inventories.Inventory` has the
same conversion for its data, with its own pollutant:
:meth:`~lair.inventories.Inventory.convert_units`.

The molecular weight
--------------------

For a gas other than methane, use its formula's molar mass.
:func:`~lair.inventories.molecular_weight` and the ``molar_mass`` of a
:class:`~lair.pollutants.Pollutant` compute it with the ``molmass`` package, which
the ``science`` extra installs (see :ref:`optional-dependencies`). This page runs
without the extras, so it writes the number. For methane ``molmass`` gives 16.0425
g/mol, which moves the result from 13.8586 to 13.8607.

Kilotonnes
----------

pint's ``kt`` is the knot, one nautical mile per hour. It parses without
complaint, so the mistake shows only when you convert to a mass, with a message
about lengths and times:

.. ipython:: python

   print(f"{(1 * units('kt')).to('m/s'):~P.3f}")
   try:
       (22.6 * units("kt")).to("kg")
   except Exception as err:
       print(f"{type(err).__name__}: {err}")

A thousand tonnes is a gigagram, ``Gg``, and pint also accepts ``kilotonne``:

.. ipython:: python

   print(f"{(22.6e6 * units('kg')).to('Gg'):~P.1f}")
   print(f"{(22.6 * units('kilotonne')).to('Gg'):~P.1f}")

Next
----

Times and time zones are the other conversion that goes wrong without a rule.
They are on the :doc:`clock page <clock>`, which also lists the errors
``convert_timezones`` raises.
