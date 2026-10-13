Get HRRR winds at a point
=========================

This example downloads six hours of HRRR's 10 m winds over the University of
Utah campus. Two rules come out of it:

* **Winds downloads as it is created.** :class:`~lair.hrrr.Winds` fetches the
  wind analysis at the HRRR grid point nearest to the point you give, one hour
  at a time, from a public archive on AWS. Nothing is saved to disk, and no
  ``LAIR_*`` variable is needed.
* **A time without a time zone is UTC.** A time with one is converted to UTC
  to choose the hour, and ``data`` keeps it as you gave it, so you can ask in
  local time and get local time back (see `Local time`_).

.. note::

   This page does not run when the docs build. It needs the ``requests``,
   ``formats`` and ``geo`` extras (see :ref:`installation`) and downloads from
   AWS. The output below was recorded from one run of this code on 2026-10-13.

The data
--------

`HRRR <https://rapidrefresh.noaa.gov/hrrr/>`_ (High-Resolution Rapid Refresh)
is NOAA's hourly weather model over the United States, on a 3 km grid. The
University of Utah's MesoWest group keeps its analyses in Zarr format in the
public `hrrrzarr <https://mesowest.utah.edu/html/hrrr/>`_ bucket on AWS.
``Winds`` reads the 10 m ``u`` and ``v`` of the hourly analyses, not the
forecasts.

Download
--------

.. code-block:: ipython

   In [1]: import pandas as pd
      ...:
      ...: from lair.hrrr import Winds
      ...:
      ...: times = pd.date_range("2024-01-15 12:00", periods=6, freq="1h").to_pydatetime().tolist()  # UTC
      ...: w = Winds(lat=40.766, lon=-111.848, times=times)
      ...: w.data
   Out[1]:
                               u         v        ws          wd
   2024-01-15 12:00:00  3.967581 -4.437818  5.952808  318.202055
   2024-01-15 13:00:00  2.486016 -0.986341  2.674536  291.640982
   2024-01-15 14:00:00  1.992718 -4.930086  5.317581  337.991665
   2024-01-15 15:00:00  0.995803 -4.007268  4.129142  346.044711
   2024-01-15 16:00:00  0.417582 -2.144768  2.185042  348.982453
   2024-01-15 17:00:00  1.670684 -0.887114  1.891602  297.967782

The columns of ``data``:

* ``u`` and ``v``: the wind toward the east and toward the north, in m/s. HRRR
  stores them along its own grid, which points away from true north except at
  97.5 W, the middle of the domain; ``Winds`` turns them to east and north.
* ``ws``: the wind speed, in m/s.
* ``wd``: the direction the wind blows from, in degrees clockwise from north (0
  is a wind from the north, 90 from the east). It is NaN when ``u`` and ``v``
  are both zero.

Each hour takes two requests, one for ``u`` and one for ``v``: these six hours
took under 10 seconds.

The grid point
--------------

``nearest_point`` is the HRRR grid point the values come from. Here it is
about 1.7 km from the point asked for; HRRR's cells are 3 km across.

.. code-block:: ipython

   In [2]: float(w.nearest_point.latitude), float(w.nearest_point.longitude)
   Out[2]: (40.75468890804213, -111.83443829829807)

Local time
----------

:mod:`lair.clock` converts the UTC index to local time once the data are in
hand:

.. code-block:: ipython

   In [3]: from lair.clock import UTC2MTN
      ...:
      ...: UTC2MTN(w.data, driver="pandas")
   Out[3]:
                                     u         v        ws          wd
   2024-01-15 05:00:00-07:00  3.967581 -4.437818  5.952808  318.202055
   2024-01-15 06:00:00-07:00  2.486016 -0.986341  2.674536  291.640982
   2024-01-15 07:00:00-07:00  1.992718 -4.930086  5.317581  337.991665
   2024-01-15 08:00:00-07:00  0.995803 -4.007268  4.129142  346.044711
   2024-01-15 09:00:00-07:00  0.417582 -2.144768  2.185042  348.982453
   2024-01-15 10:00:00-07:00  1.670684 -0.887114  1.891602  297.967782

Or ask in local time to begin with. ``Winds`` converts each time to UTC to
choose the hour (5:00 MST is 12:00 UTC), so these are the same six analyses,
and the index stays in Mountain time:

.. code-block:: ipython

   In [4]: local = pd.date_range("2024-01-15 05:00", periods=6, freq="1h", tz="America/Denver").to_pydatetime().tolist()
      ...: Winds(lat=40.766, lon=-111.848, times=local).data
   Out[4]:
                                     u         v        ws          wd
   2024-01-15 05:00:00-07:00  3.967581 -4.437818  5.952808  318.202055
   2024-01-15 06:00:00-07:00  2.486016 -0.986341  2.674536  291.640982
   2024-01-15 07:00:00-07:00  1.992718 -4.930086  5.317581  337.991665
   2024-01-15 08:00:00-07:00  0.995803 -4.007268  4.129142  346.044711
   2024-01-15 09:00:00-07:00  0.417582 -2.144768  2.185042  348.982453
   2024-01-15 10:00:00-07:00  1.670684 -0.887114  1.891602  297.967782

Next
----

:mod:`lair.hrrr` lists the lower-level functions ``Winds`` is built on.
