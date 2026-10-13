Sample CarbonTracker at points
==============================

This example downloads one day of NOAA's CarbonTracker-CH4, samples it at four
points west of Salt Lake City, and averages them into a background. Two rules
come out of it:

* **LAIR_CARBONTRACKER_DIR is the folder that holds** ``ch4/`` **and**
  ``co2/``, not one of them. lair adds the gas and the version.
* :meth:`~lair.noaa.CarbonTracker.background` **is the mean mole fraction at a
  set of points**, in ppm, such as the ends of STILT back trajectories.

.. note::

   This page does not run when the docs build. It downloads a 35 MB file from
   NOAA's FTP server. The output below was recorded from one run of this code
   on 2026-10-12.

The data
--------

`CarbonTracker-CH4 <https://gml.noaa.gov/ccgg/carbontracker-ch4/>`_ and
`CarbonTracker <https://gml.noaa.gov/ccgg/carbontracker/>`_ (CO2) are NOAA's
global models of methane and carbon dioxide. Their mole fraction files hold one
day each: the gas, and the height of each model level, on a 3 by 2 degree
global grid (CO2 also on a 1 by 1 degree grid over North America). lair reads
them in the folders of NOAA's FTP server:

.. code-block:: text

   $LAIR_CARBONTRACKER_DIR/
     ch4/
       CT-CH4-2025/
         molefractions/2020/01/CTCH4_2025.molefrac_glb3x2_2020-01-15.nc
     co2/
       CT2019B/
         molefractions/co2_total/CT2019B.molefrac_nam1x1_2015-06-01.nc

Download
--------

:class:`~lair.noaa.CarbonTrackerCH4` takes the version (``CT-CH4-2025`` by
default) and the folder, as ``carbon_tracker_directory`` or
``$LAIR_CARBONTRACKER_DIR`` (see :ref:`data-locations`):

.. code-block:: ipython

   In [1]: from lair.noaa import CarbonTrackerCH4
      ...:
      ...: ct = CarbonTrackerCH4(carbon_tracker_directory="carbontracker")
      ...: ct
   Out[1]: CarbonTrackerCH4(version=CT-CH4-2025, directory=carbontracker/ch4/CT-CH4-2025)

:meth:`~lair.noaa.CarbonTracker.download` copies files from NOAA's FTP server
into that folder. With no arguments it downloads every flux and mole fraction
file of the version (the CT-CH4-2025 mole fractions of 2015 to 2023 alone are
127 GB), so give a ``pattern``:

.. code-block:: ipython

   In [2]: ct.download(sub_dirs=["molefractions"], pattern="*2020-01-15*")
      ...: [p.as_posix() for p in sorted(ct.directory.rglob("*.nc"))]
   Out[2]: ['carbontracker/ch4/CT-CH4-2025/molefractions/2020/01/CTCH4_2025.molefrac_glb3x2_2020-01-15.nc']

That took about five minutes, most of it spent listing the folders on the
server.

Sample at points
----------------

The points are a ``DataFrame`` with the columns ``datetime`` (UTC),
``lati`` and ``long`` (degrees), and ``zagl`` (meters above ground): the names
of a PYSTILT particle table. Here, four made-up points west of Salt Lake City:

.. code-block:: ipython

   In [3]: import pandas as pd
      ...:
      ...: points = pd.DataFrame(
      ...:     {
      ...:         "datetime": pd.to_datetime(["2020-01-15 18:00"] * 4),  # UTC
      ...:         "lati": [41.5, 41.5, 39.5, 39.5],
      ...:         "long": [-114.5, -114.5, -114.5, -114.5],
      ...:         "zagl": [100.0, 2000.0, 100.0, 2000.0],
      ...:     }
      ...: )
      ...: sampled = ct.sample(points)
      ...: sampled[["lati", "zagl", "ct_ch4_ppb", "ct_latitude", "ct_longitude", "ct_level"]]
   Out[3]:
      lati    zagl   ct_ch4_ppb  ct_latitude  ct_longitude  ct_level
   0  41.5   100.0  1953.038326         41.0        -115.5       2.0
   1  41.5  2000.0  1925.469503         41.0        -115.5       8.0
   2  39.5   100.0  1983.360137         39.0        -115.5       2.0
   3  39.5  2000.0  1916.952418         39.0        -115.5       8.0

:meth:`~lair.noaa.CarbonTracker.sample` takes the grid cell and time nearest to
each point, and the model level whose layer holds the point's height (its
``zagl`` plus the height of the ground in the model). ``ct_ch4_ppb`` is the
value there; the ``ct_`` columns say which cell, time and level it came from.
A point outside the grid gets NaN, and a point on a day with no file in the
folder is dropped.

Background
----------

:meth:`~lair.noaa.CarbonTracker.background` samples the points and averages
them:

.. code-block:: ipython

   In [4]: ct.background(points)
   Out[4]:
      background_ppm  sigma_ppm  n_endpoints
   0        1.944705   0.030022            4

The result is in ppm, for both gases. ``sigma_ppm`` is the standard deviation
over the points, and ``n_endpoints`` the number of points that had a value.
With ``by=``, the name of a column of the points (``by="receptor"`` in a STILT
run), you get one row per group.

Next
----

:class:`~lair.noaa.CarbonTrackerCO2` works the same way for CO2; it takes the
version (``"CT2019B"``, for example) and the ``grid``. The ``molefractions``
attribute opens every file of the version at once and saves a copy in
``LAIR_CACHE_DIR``, so use ``sample`` or ``background`` when you need only a
few days.
