Total an inventory over a region
================================

This example totals EDGAR's methane emissions over the Salt Lake Valley, moves
them to a finer grid, and checks that the total did not change. Three rules
come out of it:

* **The regional total is** ``integrate()``: the mass emitted over the region
  in each time step, in kilograms per year here. ``total_emissions`` is not a
  total over the region: it sums the sectors in each grid cell, and is still a
  flux per area and time. ``absolute_emissions`` is the mass in each cell,
  sector by sector (see `Three sums`_).
* **Convert the total, not the flux.** ``integrate()`` gives the inventory's own
  mass unit (here kg); divide or convert the result (see `Other units`_).
* **To change the grid of an inventory, use its own** ``regrid`` **or**
  ``resample``. They are conservative by default, so the total stays the same.

.. note::

   This page does not run when the docs build. It needs ESMF, which the conda
   route of :ref:`installation` installs, and EDGAR's files, which are too large
   for the docs build. The output below was recorded from one run of this code,
   with ESMF, on 2026-10-19.

.. _inventory-data:

The data
--------

`EDGAR v8.0 <https://edgar.jrc.ec.europa.eu/dataset_ghg80>`_ publishes global
maps of methane emissions at 0.1 degrees, one file per sector and year, from
1970 to 2022. Download the annual flux files (``flx_nc``, in kg m-2 s-1) of
every CH4 sector, and unzip each sector into a folder of its own:

.. code-block:: text

   $LAIR_INVENTORY_DIR/
     EDGAR/v8/CH4/
       AGS/v8.0_FT2022_GHG_CH4_1970_AGS_flx.nc
           v8.0_FT2022_GHG_CH4_1971_AGS_flx.nc
           ...
       ENF/...
       ...

lair reads every ``.nc`` file under ``CH4/`` except those in ``TOTALS/``. Set
``LAIR_INVENTORY_DIR`` to the folder that holds ``EDGAR/`` (see
:ref:`data-locations`), or pass it to the class as ``inventory_dir=``.

Load
----

:class:`~lair.inventories.EDGARv8` opens every year of every sector. Each sector
is a variable, and each time step is labeled by the start of its year (January
1). Opening is lazy: lair reads the values only when a result needs them. Opening
all 53 years takes 20 to 50 seconds: when a calculation needs only some, pass
them, as in ``EDGARv8("CH4", start=2015, end=2022)``, and lair opens only the
files of those years (both included). This page reads every year.

.. code-block:: ipython

   In [1]: from lair.inventories import EDGARv8
      ...:
      ...: inv = EDGARv8("CH4")
      ...: inv.data.sizes
   Out[1]: Frozen({'time': 53, 'lat': 1800, 'lon': 3600})

.. code-block:: ipython

   In [2]: list(inv.data.data_vars)[:4]
   Out[2]: ['Solid_waste_landfills', 'Agricultural_soils', 'Agricultural_waste_burning', 'Chemical_processes']

Clip
----

``clip`` takes a bounding box as (west, south, east, north), in degrees. This
box, 112.3 to 111.7 W and 40.4 to 41.0 N, holds 6 by 6 cells of 0.1 degrees.

.. code-block:: ipython

   In [3]: slv = inv.clip(bbox=(-112.3, 40.4, -111.7, 41.0))  # west, south, east, north
      ...: slv.data.sizes
   Out[3]: Frozen({'time': 53, 'lat': 6, 'lon': 6})

.. _inventory-totals:

Three sums
----------

An inventory adds up in three ways, and only one of them is the total over the
region:

.. list-table::
   :header-rows: 1
   :widths: 22 30 22 26

   * - Name
     - What it is
     - Units here
     - Use it for
   * - ``total_emissions``
     - the sectors added in each cell: a flux, on ``time``, ``lat``, ``lon``
     - kg m-2 s-1
     - a map; the flux at a place
   * - ``absolute_emissions``
     - each cell's flux times its area and the length of the time step: a mass,
       one variable per sector, on ``time``, ``lat``, ``lon``
     - kg per cell per year
     - the mass in a cell; one sector's total
   * - ``integrate()``
     - ``absolute_emissions`` added over the cells and the sectors: a mass, on
       ``time`` only
     - kg per year
     - **the total over the region**

The kilograms are the mass unit of the inventory's flux: the area and the time
cancel. "Per year" is because ``EDGARv8`` loads annual data, one value per year;
with monthly data each value is one month's mass. The length of a year follows
the calendar (366 days in a leap year).

A flux in each cell: ``total_emissions``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``total_emissions`` sums the sectors in each cell. Its ``units`` attribute shows
that it is still a flux, in kilograms per square meter per second:

.. code-block:: ipython

   In [4]: slv.total_emissions.attrs["units"]
   Out[4]: 'kg/m**2/s'

Do not add it up over the cells: the cells differ in area, so a sum of fluxes is
not a mass.

The total: ``integrate()``
^^^^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~lair.inventories.Inventory.integrate` multiplies each cell's flux by the
cell's area and by the length of the time step, then sums the cells and the
sectors. The ``units`` attribute of the result is just ``kg``: the mass emitted
over the whole region in one time step, here one year.

``integrate()`` returns a lazy (dask) array: ``compute()`` reads the files and
does the work. That took a minute or two here.

.. code-block:: ipython

   In [5]: total = slv.integrate().compute()
      ...: total.attrs
   Out[5]: {'long_name': 'Total Emissions', 'units': 'kg'}

.. code-block:: ipython

   In [6]: total.sel(time=slice("2020", "2022")).to_series() / 1e6  # kg to kt
   Out[6]:
   time
   2020-01-01    22.089165
   2021-01-01    22.363681
   2022-01-01    22.588048
   Freq: YS-JAN, Name: emissions, dtype: float64

So EDGAR puts 22.6 kt (thousand tonnes) of methane in this box in 2022.

The mass in each cell: ``absolute_emissions``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`~lair.inventories.Inventory.absolute_emissions` is the step before the
sum: the mass each cell emitted in each year, one variable per sector. To check
it by hand, take one cell in 2022 (not a leap year): its flux, summed over the
sectors, times its area (``gridcell_area``, in
km2) times the seconds in the year:

.. code-block:: ipython

   In [7]: from lair.inventories import sum_sectors
      ...:
      ...: cell = dict(lat=40.75, lon=-111.95, method="nearest")  # one 0.1 degree cell
      ...:
      ...: flux = float(slv.total_emissions.sel(time="2022-01-01").sel(**cell))  # kg m-2 s-1
      ...: area = float(slv.gridcell_area.sel(**cell)) * 1e6  # km2 to m2
      ...: seconds = 365 * 24 * 3600
      ...: mass = slv.absolute_emissions.sel(time="2022-01-01").sel(**cell).compute()
      ...:
      ...: print(f"by hand:            {flux * area * seconds:,.0f} kg")
      ...: print(f"absolute_emissions: {float(sum_sectors(mass)):,.0f} kg")
   by hand:            8,566,596 kg
   absolute_emissions: 8,566,596 kg

EDGAR leaves a sector's cell empty (NaN) where that sector emits nothing.
:func:`~lair.inventories.sum_sectors`, which ``total_emissions`` and
``integrate()`` use, skips those, as xarray's ``sum`` does; adding the sectors
with Python's ``+`` or ``sum`` gives NaN.

Summed over the cells, each sector gives its own total over the region. Added
together, they are ``integrate()``'s total:

.. code-block:: ipython

   In [8]: by_sector = slv.absolute_emissions.sel(time="2022-01-01").sum(["lat", "lon"]).compute()
      ...: kt = by_sector.to_pandas() / 1e6  # kg to kt
      ...: print(kt.sort_values(ascending=False).head().round(3).to_string())
      ...: print(f"all sectors: {kt.sum():.3f} kt")
   Solid_waste_landfills       12.407
   Waste_water_handling         5.195
   Energy_for_buildings         1.601
   Fuel_exploitation_GAS        1.269
   Solid_waste_incineration     0.649
   all sectors: 22.588 kt

To leave a sector out of a total, pass it to
:func:`~lair.inventories.sum_sectors` as ``exclude=``.

A total adds the sectors as numbers, so they must be in the same unit. EDGAR's
are. An inventory built from sectors in different units (a dataset whose
variables carry their own ``units``) raises a ``ValueError`` from
``total_emissions`` and ``integrate()`` that names each sector's units; convert
them first with ``convert_units``.

Other units
^^^^^^^^^^^

``integrate()`` keeps the inventory's mass unit, so divide the result: kg /
1e6 is kt (thousand tonnes), as above, which is also Gg. Converting the flux
first gives a slightly different number. A flux per year (``Gg/km**2/yr``)
goes through pint's year of 365.25 days, and ``integrate()`` then takes each
value as one year's worth, whatever the year's length (the details are under
:attr:`~lair.inventories.Inventory.absolute_emissions`):

.. code-block:: ipython

   In [9]: kg = total.sel(time=["2020-01-01", "2022-01-01"])
      ...: gg = slv.convert_units("Gg/km**2/yr").integrate().sel(time=["2020-01-01", "2022-01-01"]).compute()
      ...:
      ...: (100 * (gg / (kg / 1e6) - 1)).to_series().round(3)  # percent
   Out[9]:
   time
   2020-01-01   -0.205
   2022-01-01    0.068
   Name: emissions, dtype: float64

2020 is a leap year: converting first loses 0.2% of it, and adds 0.07% to 2022.

Regrid, and check the total
---------------------------

``resample(0.01)`` moves the region to a grid of 0.01 degrees (60 by 60 cells).
An inventory's ``resample`` and ``regrid`` are conservative by default: each new
cell gets the area-weighted flux of the old cells it overlaps, so flux times
area, and with it the total, stays the same.

.. code-block:: ipython

   In [10]: fine = slv.resample(0.01)  # conservative by default
       ...: fine.data.sizes
   Out[10]: Frozen({'time': 53, 'lat': 60, 'lon': 60})

Now compare the 2022 total before and after:

.. code-block:: ipython

   In [11]: fine_total = fine.integrate().compute()
       ...:
       ...: before = float(total.sel(time="2022-01-01"))
       ...: after = float(fine_total.sel(time="2022-01-01"))
       ...: print(f"before: {before / 1e6:.3f} kt")
       ...: print(f"after:  {after / 1e6:.3f} kt")
       ...: print(f"change: {100 * (after - before) / before:+.4f} %")
   before: 22.588 kt
   after:  22.588 kt
   change: -0.0005 %

The total changed by 0.0005% (5 parts per million): the mass is kept.

A resolution that does not divide the box still keeps the total. The new grid
covers the whole box: ``resample(0.25)`` gives 3 by 3 cells, 0.75 degrees wide,
and the last row and column reach past the box's edge. They hold only the
emissions that were inside it, so the total is still kept.

Why not ``lair.geo.resample``?
------------------------------

:func:`lair.geo.resample` and :func:`lair.geo.regrid` are for any field: a
temperature, a wind speed, a mole fraction. Their default is bilinear
interpolation, which suits a field you read at a point. It reads each value as a
point at its cell's center and interpolates linearly between those points, so
it does not keep the sum of flux times area.

A flux is different: its total must not change when the grid does. On the box
above, ``lair.geo.resample`` with its default lost 6.6% of the 2022 total at
0.01 degrees. The 1,100 new cells along the edges of the box, outside the
outermost cell centers, came out zero.

So send a flux through the inventory's own ``regrid`` or ``resample``, or pass
the method yourself: ``method="conservative"`` to ``lair.geo.regrid``, and
``regrid_method="conservative"`` to ``lair.geo.resample``. The two defaults
differ on purpose: ``lair.geo`` handles any field, and not every field is
conserved, while an inventory always holds fluxes.

Next
----

:mod:`lair.inventories` lists the other inventories (EPA, GFEI, Vulcan,
WetCHARTs) and what each one's total includes.
