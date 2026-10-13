Average a time series by local hour and by season
=================================================

This example takes a year of hourly data stamped in UTC, moves it to local
time, and averages it by hour of the day, over the afternoon, and by season.
Two rules come out of it:

* **A time without a time zone is UTC** everywhere in lair. Convert to local
  time before you group by hour, or the hours are UTC hours.
* :func:`~lair.clock.diurnal` **and** :func:`~lair.clock.seasonal` average by
  hour of the day and by season and year.

This page runs when the docs build, on made-up data.

The data
--------

A year of hourly methane, in ppm, stamped in UTC. It is made up: a daily cycle
that peaks at 12 UTC (5 a.m. in Utah, under the night-time inversion), a
seasonal cycle that peaks in January, and noise.

.. ipython:: python

   import numpy as np
   import pandas as pd

   rng = np.random.default_rng(0)
   times = pd.date_range("2023-12-01", "2024-11-30 23:00", freq="1h")  # UTC
   daily = 0.05 * np.cos(2 * np.pi * (times.hour - 12) / 24)
   yearly = 0.02 * np.cos(2 * np.pi * (times.dayofyear - 15) / 366)
   data = pd.DataFrame({"ch4": 2.0 + daily + yearly + rng.normal(0, 0.01, times.size)}, index=times)
   data.head(3)

Local time
----------

:func:`~lair.clock.convert_timezones` converts between time zones, and its
shortcuts name the common ones: ``UTC2MST`` (UTC to Mountain Standard Time,
always 7 hours behind) and ``UTC2MTN`` (UTC to ``America/Denver``, which follows
daylight saving time). ``driver="pandas"`` converts the index of a
``DataFrame``:

.. ipython:: python

   from lair.clock import UTC2MST

   local = UTC2MST(data, driver="pandas")
   local.head(3)

Standard time is the same offset from UTC all year, so every day is 24 hours
long and noon is noon in January and in July. That is why the afternoon below
is in standard time.

By hour of the day
------------------

:func:`~lair.clock.diurnal` groups the rows by their time of day (rounded down
to ``freq``, an hour by default) and averages each group:

.. ipython:: python

   from lair.clock import diurnal

   cycle = diurnal(local)
   cycle.round(3).head(8)

   @savefig clock_diurnal.png width=5in
   cycle.plot(xlabel="hour (MST)", ylabel="CH$_4$ (ppm)", legend=False);

The cycle peaks at 05:00 MST, where the made-up data put it. Without the
conversion it would peak at 12:00: the same data, labeled in UTC.

The afternoon
-------------

``AFTERNOON`` lists the hours from noon to 4 p.m., standard time. Selecting
those hours and averaging each day gives one afternoon value per day:

.. ipython:: python

   from lair.clock import AFTERNOON

   AFTERNOON
   afternoon = local[local.index.hour.isin(AFTERNOON)].resample("D").mean()
   afternoon.round(3).head(3)

By season
---------

:func:`~lair.clock.seasonal` averages over DJF, MAM, JJA and SON in each year.
A DJF season takes the year of its January and February, so December 2023
counts toward DJF 2024.

.. ipython:: python

   from lair.clock import seasonal

   seasonal(local).round(3)

The data start at midnight UTC on December 1, which is 5 p.m. on November 30 in
MST. Those seven hours make an "SON 2023" of their own: a season the data cover
only in part is kept, not dropped, so drop it yourself if you do not want it.

Decimal dates
-------------

Some tools, NOAA's curve-fitting filter among them, count time in decimal
years. :func:`~lair.clock.dt2decimalDate` and
:func:`~lair.clock.decimalDate2dt` convert both ways (a time with a time zone
is converted to UTC first):

.. ipython:: python

   import datetime as dt

   from lair.clock import decimalDate2dt, dt2decimalDate

   dt2decimalDate(dt.datetime(2024, 7, 1))
   decimalDate2dt(2024.5)

Next
----

:mod:`lair.clock` lists the rest, including ``TimeRange`` (a start and stop
parsed from strings such as ``"2024-07"``) and ``convert_timezones``'s options
for the hours that daylight saving time repeats or skips.
