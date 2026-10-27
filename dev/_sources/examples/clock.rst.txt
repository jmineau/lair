Average a time series by local hour and by season
=================================================

This example takes a year of hourly data stamped in UTC, moves it to local
time, and averages it by hour of the day, over the afternoon, and by season.
Three rules come out of it:

.. _utc-rule:

* **A time without a time zone is UTC** in lair's functions that take a time,
  such as :func:`~lair.soundings.get_soundings` and
  :func:`~lair.clock.dt2decimalDate`. The exception is
  :func:`~lair.clock.convert_timezones`, which raises on one unless you give
  ``fromtz`` (the third rule). Convert to local time before you group by hour,
  or the hours are UTC hours.
* :func:`~lair.clock.diurnal` **and** :func:`~lair.clock.seasonal` average by
  hour of the day and by season and year.
* :func:`~lair.clock.convert_timezones` **does not guess.** The shortcuts
  (``UTC2MST`` and the rest) name the zone they convert from. The function
  itself needs ``fromtz`` for a time without a zone, and ``ambiguous=`` or
  ``nonexistent=`` for a local time that daylight saving time repeats or skips
  (see `Times that will not convert`_).

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
   yearly = 0.02 * np.cos(2 * np.pi * (times.day_of_year - 15) / 366)
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

Without ``driver="pandas"`` the shortcuts take a list of times, so to convert
one time, put it in a list:

.. ipython:: python

   import datetime as dt

   from lair.clock import UTC2MTN

   UTC2MTN([dt.datetime(2024, 7, 15, 18)])

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

Times that will not convert
---------------------------

:func:`~lair.clock.convert_timezones` stops with a ``ValueError`` when it would
have to guess. The message says what to add.

**A time without a zone, and no** ``fromtz``. The shortcuts above pass
``fromtz="UTC"``, which is the first rule at work. Called on its own, the
function does not apply that rule: it asks.

.. ipython:: python

   from lair.clock import convert_timezones

   try:
       convert_timezones(data, "America/Denver", driver="pandas")
   except ValueError as err:
       print(err)

   convert_timezones(data, "America/Denver", fromtz="UTC", driver="pandas").head(2)

A time that already has a zone, such as ``local`` above, needs no ``fromtz``: it
converts from its own.

.. ipython:: python

   convert_timezones(local, "UTC", driver="pandas").head(2)

**A local time that happens twice, or not at all.** Only a zone with daylight
saving time can do this, and only when you convert *from* it, as ``MTN2UTC`` does
(local time in Utah to UTC). In Utah, the clocks go back at 2 a.m. on the first
Sunday in November, so 1:00 to 2:00 happens twice. They go forward on the second
Sunday in March, so 2:00 to 3:00 never happens. A file whose times are
Utah clock time has these hours:

.. ipython:: python

   from lair.clock import MST2UTC, MTN2UTC

   fall = pd.DataFrame(
       {"ch4": range(5)},
       index=pd.date_range("2023-11-05 00:00", periods=5, freq="1h"),
   )  # Utah clock time
   try:
       MTN2UTC(fall, driver="pandas")
   except ValueError as err:
       print(err)

``ambiguous=`` says which of the two 1:00s the file means: ``True`` the first
(daylight time, 07:00 UTC) or ``False`` the second (standard time, 08:00 UTC).
``"NaT"`` gives up on it and leaves a missing time, and ``"infer"`` works out
which is which in a file that lists both, in order. Here is the 1:00 row each
way:

.. ipython:: python

   MTN2UTC(fall, driver="pandas", ambiguous=True).index[1]
   MTN2UTC(fall, driver="pandas", ambiguous=False).index[1]

An hour that never happened is the same kind of error, and ``nonexistent=`` is
the matching option. ``"shift_forward"`` moves a 2:00 time to 3:00, so it lands
on the same instant as the real 3:00 row. ``"shift_backward"`` moves it to the
last instant before the gap, and ``"NaT"`` leaves a missing time:

.. ipython:: python

   spring = pd.DataFrame(
       {"ch4": range(5)},
       index=pd.date_range("2024-03-10 00:00", periods=5, freq="1h"),
   )  # Utah clock time
   try:
       MTN2UTC(spring, driver="pandas")
   except ValueError as err:
       print(err)

   MTN2UTC(spring, driver="pandas", nonexistent="shift_forward")

Standard time and UTC have no such hours, so ``UTC2MTN``, ``UTC2MST`` and
``MST2UTC`` never raise them. If the instrument that wrote a file kept standard
time all year (its manual or settings say), convert it with ``MST2UTC`` and
nothing is ambiguous:

.. ipython:: python

   MST2UTC(fall, driver="pandas").index[1]

Next
----

:mod:`lair.clock` lists the rest, including ``TimeRange`` (a start and stop
parsed from strings such as ``"2024-07"``).
