Read a sounding and compute its potential temperature
=====================================================

This example downloads two soundings from Salt Lake City, reads them, and
computes potential temperature on regular heights. Two rules come out of it:

* **lair reads soundings from a folder you keep**, one file per launch.
  :func:`~lair.soundings.download_soundings` fills it, and
  :func:`~lair.soundings.get_soundings` reads it.
* **get_soundings puts every sounding on the same heights**: every 10 m from
  the ground at the station (1289 m above sea level at Salt Lake City) up to
  4999 m.

.. note::

   This page does not run when the docs build. It needs the ``requests`` extra
   (see :ref:`installation`) and downloads from the University of Wyoming. The
   output below was recorded from one run of this code on 2026-10-29, with lair
   2026.12.11.dev9+g55e932b18.

.. _soundings-data:

The data
--------

The `University of Wyoming <https://weather.uwyo.edu/upperair/sounding.shtml>`_
keeps the radiosonde soundings of stations around the world, launched at 00
and 12 UTC. lair downloads them with `siphon <https://unidata.github.io/siphon/>`_
and saves each one as a CSV file named by station and UTC time, in one folder
per station:

.. code-block:: text

   $LAIR_SOUNDING_DIR/
     SLC/
       SLC_2024011500.csv
       SLC_2024011512.csv
       ...

The functions take that folder as ``sounding_dir`` (``dst`` for the downloads),
or use ``$LAIR_SOUNDING_DIR/<station>`` (see :ref:`data-locations`).

Download
--------

:func:`~lair.soundings.download_soundings` downloads every sounding from
``start`` to ``end`` (both included; times follow :ref:`the UTC rule <utc-rule>`). It
skips a file that is already in the folder, so you can run it again to add a
period.

.. code-block:: ipython

   In [1]: import os
      ...:
      ...: from lair.soundings import download_soundings
      ...:
      ...: download_soundings("SLC", "2024-01-15 00:00", "2024-01-15 12:00", dst="soundings/SLC")
      ...: sorted(os.listdir("soundings/SLC"))
   Out[1]: ['SLC_2024011500.csv', 'SLC_2024011512.csv']

Read one sounding
-----------------

:class:`~lair.soundings.Sounding` reads one file. ``data`` holds one row per
level the balloon reported, from the ground up:

.. code-block:: ipython

   In [2]: from lair.soundings import Sounding
      ...:
      ...: snd = Sounding("soundings/SLC/SLC_2024011512.csv")
      ...: snd.station, snd.time
   Out[2]: ('SLC', datetime.datetime(2024, 1, 15, 12, 0))

.. code-block:: ipython

   In [3]: snd.data[["pressure", "height", "temperature", "dewpoint"]].head()
   Out[3]:
      pressure  height  temperature  dewpoint
   0     871.0  1289.0          0.6      -0.7
   1     855.0  1436.0          1.0      -3.0
   2     850.0  1482.0          0.8      -4.2
   3     845.6  1524.0          0.6      -5.0
   4     834.0  1634.0         -0.1      -7.1

Pressure is in hPa, height in meters above sea level, temperature and dewpoint
in degrees Celsius, and the winds (``speed``, ``u_wind`` and ``v_wind``) in
knots. The ``units`` attribute of :class:`~lair.soundings.Sounding` lists the
unit of each column, and every file ``download_soundings`` saves is in those
units: Wyoming sends the winds in m/s today and sent knots before, and lair
converts them to knots as it saves, so a folder never mixes the two:

.. code-block:: ipython

   In [4]: snd.data[["height", "speed"]].head(3)
   Out[4]:
      height      speed
   0  1289.0   1.943844
   1  1436.0  11.079914
   2  1482.0  13.995680

The station's ``elevation``, which Wyoming no longer sends, is saved as the
height of the first level, the ground:

.. code-block:: ipython

   In [5]: print(snd.units["speed"], snd.elevation)
   knot 1289.0

The columns are otherwise whatever Wyoming sends, so a file downloaded today
also has ``relative_humidity`` and ``mixing_ratio``, and no ``pw``
(precipitable water).

.. _soundings-wrapped-heights:

**Heights that wrap at 10 km.** A few saved files have heights that start
again from 0 at 10 km: after 9998 m the next level is 96 m, while the pressure
keeps falling (four Salt Lake City soundings of March and April 2024).
``Sounding`` reads a fall of more than 9 km from one level to the next as a
wrap: it adds 10 km to that level and every level after it (another 10 km
after a second wrap) and logs a warning that names the file. A smaller fall is
kept as it is, since it is real: a balloon pushed down reports a level 10 to
50 m lower than the one before (about half of the Salt Lake City files have
one, all above 2200 m). ``data`` keeps every level, in the order of the file.

.. _soundings-heights:

Read many soundings
-------------------

:func:`~lair.soundings.get_soundings` reads every file in the folder from
``start`` to ``end`` (both included) and returns an xarray ``Dataset`` with a
``time`` and a ``height`` dimension:

.. code-block:: ipython

   In [6]: from lair.soundings import get_soundings
      ...:
      ...: ds = get_soundings("SLC", start="2024-01-15", end="2024-01-16", sounding_dir="soundings/SLC")
      ...: ds[["pressure", "temperature"]]
   Out[6]:
   <xarray.Dataset> Size: 15kB
   Dimensions:      (time: 2, height: 372)
   Coordinates:
     * time         (time) datetime64[us] 16B 2024-01-15 2024-01-15T12:00:00
     * height       (height) float64 3kB 1.289e+03 1.299e+03 ... 4.999e+03
   Data variables:
       pressure     (time, height) float64 6kB 868.0 866.9 865.8 ... 537.7 537.0
       temperature  (time, height) float64 6kB 1.2 0.9826 0.7652 ... -21.04 -21.1
   Attributes:
       station:                 SLC
       latitude:                40.77
       longitude:               -111.95
       elevation:               1289.0
       interpolation_interval:  10
       notes:                   Interpolated to 1289-5000m at 10m intervals.

Each sounding is interpolated linearly in height, between the levels it
reported, to 1289, 1299, ..., 4999 m. A height above the top of a sounding is
left NaN, not extrapolated. Wind direction and speed are rebuilt from the
interpolated ``u_wind`` and ``v_wind``, since a direction cannot be averaged
(350 and 10 degrees would give 180).

The heights start at the ground: :meth:`Sounding.interpolate()
<lair.soundings.Sounding.interpolate>` starts at the station's ``elevation``
by default, or, in a file that has none, at its first level, the ground, and
``get_soundings`` gives every sounding the heights of the lowest ground among
them, so that they line up. For other heights, interpolate each sounding
yourself, for example ``snd.interpolate(start=0, interval=50)`` (levels below
the ground come out NaN), or pass ``driver="pandas"`` to ``get_soundings`` to get the raw levels of
every sounding in one ``DataFrame``.

Potential temperature
---------------------

:func:`lair.meteorology.poisson` takes plain SI values (kelvin and pascals; see
:doc:`../quickstart`), so convert the sounding's degrees Celsius and hPa on the
way in:

.. code-block:: ipython

   In [7]: from lair.meteorology import poisson
      ...:
      ...: theta = poisson(ds.temperature + 273.15, ds.pressure * 100)  # degC to K, hPa to Pa
      ...: theta.sel(height=[1289, 1499, 1999, 2999]).to_pandas().round(1)
   Out[7]:
   height               1289.0  1499.0  1999.0  2999.0
   time
   2024-01-15 00:00:00   285.7   286.2   289.2   293.1
   2024-01-15 12:00:00   284.8   287.1   288.3   290.6

Potential temperature rises with height in both soundings, so the air over the
valley was stable. At 12 UTC (5 a.m. in Salt Lake City) it rose 2.3 K in the
lowest 210 m: a surface inversion.

.. _soundings-download:

When get_soundings downloads
----------------------------

``get_soundings`` reads the folder and downloads nothing, unless you pass
``download=True``. Without it, a period the folder does not cover raises a
``ValueError`` that names the folder and says how to download:

.. code-block:: ipython

   In [8]: get_soundings("SLC", start="2024-02-01", end="2024-02-02", sounding_dir="soundings/SLC")
   ValueError: No soundings found in soundings/SLC for the requested period. Pass download=True (with start and end) to download them.

With ``download=True`` it first downloads every sounding from ``start`` to
``end`` that the folder does not have (only those in ``months``, if you give
``months``), then reads them. It is ``download_soundings`` followed by the read,
so ``start`` and ``end`` are required and the folder must be one you can write
to. A sounding that Wyoming does not have is asked for again on every call, so
give the period you need.

.. code-block:: ipython

   In [9]: ds = get_soundings("SLC", start="2024-02-01", end="2024-02-02", sounding_dir="soundings/SLC", download=True)
      ...: sorted(os.listdir("soundings/SLC"))
   Out[9]: ['SLC_2024011500.csv', 'SLC_2024011512.csv', 'SLC_2024020100.csv', 'SLC_2024020112.csv', 'SLC_2024020200.csv']

Next
----

:mod:`lair.soundings` lists the rest. ``Sounding.plot()`` is not written yet:
it raises ``NotImplementedError``. :mod:`lair.pcaps` computes the valley heat
deficit from the output of ``get_soundings``.
