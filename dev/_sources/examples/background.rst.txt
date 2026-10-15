Estimate a background
=====================

This example estimates the methane background under two years of made-up hourly
data from a city site, with each of lair's background methods, and checks each
against the background the data were made from. Three rules come out of it:

* **Each method finds the clean part of the data in its own way.**
  :func:`~lair.background.rolling_baseline` and
  :func:`~lair.background.phase_shift_corrected_baseline` take a low quantile
  over a window. :func:`~lair.background.thoning` fits a smooth curve through
  every value it is given, spikes included, so give it data that is already
  clean, such as afternoon means or a clean site's record.
* **A rolling window must be longer than the longest stretch without clean
  air.** Over a pollution episode of several days, the low quantile of a day is
  the episode, not the background.
* :func:`~lair.background.get_well_mixed` **reads the hours in local standard
  time.** Convert UTC data first (see the :doc:`clock page <clock>`).

.. note::

   This page does not run when the docs build. ``lair.background`` needs NOAA's
   CCG filter, which lair downloads the first time it is imported (see
   :ref:`first-import-download`), and ``scipy``, from the ``science`` extra
   (see :ref:`optional-dependencies`). The output below was recorded from one
   run of this code on 2026-10-14.

The methods
-----------

.. list-table::
   :header-rows: 1
   :widths: 24 26 26 24

   * - Function
     - Needs
     - Returns
     - Use it for
   * - :func:`~lair.background.rolling_baseline`
     - a ``Series`` with a time index; ``window`` (24 hours by default) and the
       quantile ``q`` (0.01)
     - the rolling low quantile, smoothed with a rolling mean over the same
       window, at the times of the data
     - a continuous record from one site, when every window holds some clean air
   * - :func:`~lair.background.phase_shift_corrected_baseline`
     - a ``Series`` of one-second data; the window ``n`` in seconds (3600) and
       ``q`` (0.01); each calendar day of the index is done on its own
     - a low quantile from windows ahead of and behind each time, smoothed
     - one-second data from a moving platform, whose background changes within
       the hour
   * - :func:`~lair.background.get_well_mixed`
     - a ``Series`` or ``DataFrame`` indexed in local standard time, and the
       hours to keep (12 to 16 by default, ``lair.clock.AFTERNOON``)
     - one mean per day over those hours
     - the afternoon, when the boundary layer is deep and a site sees regional
       air
   * - :func:`~lair.background.thoning`
     - a ``Series`` with a time index (NaN values are dropped), and the times
       to evaluate at, ``smooth_time``
     - NOAA's smooth curve (a polynomial trend and yearly harmonics, plus the
       residuals smoothed) at ``smooth_time``, NaN outside the time range of the
       data
     - a record that is already background: afternoon means, flasks, a clean
       site
   * - :meth:`CarbonTracker.background <lair.noaa.CarbonTracker.background>`
     - CarbonTracker's files and a set of points, such as the ends of STILT
       back trajectories
     - the mean mole fraction at the points, in ppm
     - air from upwind, with no measurement from the site (see `From upwind`_)

The data
--------

Two years of hourly methane, in ppm, stamped in UTC. The true background rises
0.01 ppm a year and peaks in winter. On top of it, the city adds an enhancement:
large at night, under the night-time inversion, and small in the afternoon. For
six days in January 2023 a winter inversion holds the city's methane in all day.

.. code-block:: ipython

   In [1]: import numpy as np
      ...: import pandas as pd
      ...:
      ...: rng = np.random.default_rng(0)
      ...: times = pd.date_range("2022-01-01", "2023-12-31 23:00", freq="1h")  # UTC
      ...: years = (times - times[0]).days / 365
      ...: true = pd.Series(
      ...:     1.95 + 0.01 * years + 0.008 * np.cos(2 * np.pi * (times.dayofyear - 30) / 365),
      ...:     index=times,
      ...: )
      ...: hour = (times.hour - 7) % 24  # Mountain Standard Time
      ...: night = (hour >= 19) | (hour < 9)
      ...: episode = (times >= "2023-01-20") & (times < "2023-01-26")
      ...: enhancement = rng.exponential(np.where(night, 0.10, 0.01)) + np.where(episode, 0.15, 0)
      ...: ch4 = (true + enhancement + rng.normal(0, 0.002, times.size)).rename("ch4")
      ...: ch4.describe().round(3)
   Out[1]:
   count    17520.000
   mean         2.023
   std          0.089
   min          1.943
   25%          1.968
   50%          1.985
   75%          2.047
   max          2.826
   Name: ch4, dtype: float64

A rolling low quantile
----------------------

:func:`~lair.background.rolling_baseline` needs nothing but the data. It takes
the 1st percentile over a window centered on each time, then smooths it with a
rolling mean over the same window:

.. code-block:: ipython

   In [2]: from lair.background import rolling_baseline
      ...:
      ...: day = rolling_baseline(ch4)  # window="24h", q=0.01
      ...: two_weeks = rolling_baseline(ch4, window="14d")
      ...: check = pd.DataFrame({"true": true, "24h": day, "14d": two_weeks})
      ...: check.loc["2023-01-18":"2023-01-27"].at_time("19:00").round(3)
   Out[2]:
                         true    24h    14d
   2023-01-18 19:00:00  1.968  1.969  1.967
   2023-01-19 19:00:00  1.968  1.978  1.967
   2023-01-20 19:00:00  1.968  2.104  1.967
   2023-01-21 19:00:00  1.968  2.118  1.968
   2023-01-22 19:00:00  1.968  2.120  1.968
   2023-01-23 19:00:00  1.969  2.120  1.968
   2023-01-24 19:00:00  1.969  2.119  1.968
   2023-01-25 19:00:00  1.969  1.999  1.968
   2023-01-26 19:00:00  1.969  1.967  1.968
   2023-01-27 19:00:00  1.969  1.969  1.968

Outside the episode both windows hold clean afternoons, and both follow the
true background. During the episode no hour of the day is clean, so the 24-hour
window rises with the city's methane. The 14-day window still holds clean days
on either side.

Afternoon means and a smooth curve
----------------------------------

:func:`~lair.background.thoning` fits NOAA's curve (see
:ref:`first-import-download` for where it comes from) through every value it is
given. Given every hour, it follows the mean of the data, enhancements
included. Given afternoon means, it follows the afternoon, when the city's
methane is mixed through a deep boundary layer.
:func:`~lair.background.get_well_mixed` takes those means, from a series in
local standard time:

.. code-block:: ipython

   In [3]: from lair.background import get_well_mixed, thoning
      ...: from lair.clock import UTC2MST
      ...:
      ...: local = UTC2MST(ch4.to_frame(), driver="pandas")["ch4"]
      ...: afternoon = get_well_mixed(local)
      ...: afternoon.head(3).round(3)
   Out[3]:
   2022-01-01 00:00:00-07:00    1.965
   2022-01-02 00:00:00-07:00    1.965
   2022-01-03 00:00:00-07:00    1.974
   Freq: D, Name: ch4, dtype: float64

Each day is labeled at its local midnight. Given the UTC series instead, it
would average 12:00 to 16:59 UTC, which is 5 to 9 in the morning in Utah, under
the night-time inversion. ``smooth_time`` evaluates the curve
at the hourly times, to put it beside the others:

.. code-block:: ipython

   In [4]: every_hour = thoning(ch4)
      ...: afternoons = thoning(afternoon, smooth_time=ch4.index)
      ...: int(afternoons.isna().sum())
   Out[4]: 23

The 23 missing values are the hours before the first afternoon mean and after
the last: the curve is NaN outside the time range of its data.

Side by side
------------

How far above the true background each estimate is, on average over every
hour, and over the hours of the episode:

.. code-block:: ipython

   In [5]: methods = pd.DataFrame(
      ...:     {
      ...:         "rolling_baseline, 24h": day,
      ...:         "rolling_baseline, 14d": two_weeks,
      ...:         "thoning, every hour": every_hour,
      ...:         "thoning, afternoon means": afternoons,
      ...:     }
      ...: )
      ...: above = methods.sub(true, axis=0)
      ...: pd.DataFrame({"all hours": above.mean(), "episode": above[episode].mean()}).round(3)
   Out[5]:
                             all hours  episode
   rolling_baseline, 24h         0.001    0.128
   rolling_baseline, 14d        -0.001   -0.001
   thoning, every hour           0.063    0.084
   thoning, afternoon means      0.011    0.035

The curve through every hour sits above the background by about the mean
enhancement, night-time plumes included. The curve through the afternoon means
sits above it by about the afternoon's enhancement, which is small at this
made-up site and larger at a real site in the middle of a city. Both rise
during the episode, when the afternoons are not clean either. The rolling low
quantile is close whenever its window holds clean air.

One-second data
---------------

:func:`~lair.background.phase_shift_corrected_baseline` is for one-second data
from a moving platform, a car or a train, whose background can change within
the hour. Here, two hours of it: a background that rises 0.05 ppm over 20
minutes, and 30 short plumes.

.. code-block:: ipython

   In [6]: from lair.background import phase_shift_corrected_baseline
      ...:
      ...: seconds = pd.date_range("2024-06-01 18:00", periods=7200, freq="1s")  # UTC, 1 Hz
      ...: i = np.arange(seconds.size)
      ...: true_drive = 1.98 + 0.05 * np.clip((i - 3000) / 1200, 0, 1)
      ...: plumes = np.zeros(seconds.size)
      ...: for center in rng.integers(0, seconds.size, 30):
      ...:     width = rng.integers(5, 30)  # seconds
      ...:     plumes += rng.exponential(0.3) * np.exp(-0.5 * ((i - center) / width) ** 2)
      ...: drive = pd.Series(true_drive + plumes + rng.normal(0, 0.003, i.size), index=seconds)
      ...:
      ...: fast = pd.DataFrame(
      ...:     {
      ...:         "phase_shift_corrected_baseline": phase_shift_corrected_baseline(drive),
      ...:         "rolling_baseline, 1h": rolling_baseline(drive, window="1h"),
      ...:     }
      ...: )
      ...: fast.sub(true_drive, axis=0).resample("10min").mean().round(3)
   Out[6]:
                        phase_shift_corrected_baseline  rolling_baseline, 1h
   2024-06-01 18:00:00                           0.010                -0.006
   2024-06-01 18:10:00                          -0.001                -0.006
   2024-06-01 18:20:00                          -0.004                -0.006
   2024-06-01 18:30:00                          -0.003                -0.006
   2024-06-01 18:40:00                           0.003                -0.005
   2024-06-01 18:50:00                           0.004                -0.016
   2024-06-01 19:00:00                          -0.006                -0.035
   2024-06-01 19:10:00                          -0.009                -0.040
   2024-06-01 19:20:00                          -0.007                -0.032
   2024-06-01 19:30:00                          -0.007                -0.025
   2024-06-01 19:40:00                          -0.006                -0.019
   2024-06-01 19:50:00                          -0.006                -0.012

Both use a one-hour window. While the background is flat, both stay within about
0.01 ppm of it. Where it rises, from 18:50 to 19:10, the rolling quantile,
centered on each time, falls up to 0.04 ppm below it and is still 0.01 ppm below
at the end. The phase shift corrected baseline, which looks ahead and behind
separately, stays within about 0.01 ppm.

From upwind
-----------

The methods above estimate the background from the site's own data. The other
way is to take it from upwind: sample a global model, such as NOAA's
CarbonTracker, where the air arriving at the site came from. That needs no
clean hour at the site, and it is the way lair's maintainer took for the Salt
Lake Valley methane inversion of his first paper:
CarbonTracker-CH4 version ``CT-CH4-2025``, sampled with
:meth:`~lair.noaa.CarbonTracker.background` at the ends of 24-hour STILT back
trajectories released from the site in the afternoon, and averaged to one value
per day. How to sample CarbonTracker at a set of points is on the
:ref:`CarbonTracker page <carbontracker-background>`.

Next
----

:func:`~lair.background.thoning_filter` returns NOAA's filter itself, which
also gives the trend and the growth rate. For mobile data on a fixed route,
:func:`lair.transects.enhancement` subtracts each transit's own low percentile
instead.
