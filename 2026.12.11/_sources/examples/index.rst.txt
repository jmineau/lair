Examples
========

Worked examples, one per job, each from loading the data to a checked result.
Each page says whether it runs when the docs build.

* :doc:`clock`: move UTC data to local time and average it by hour and by
  season, and what to do when a time will not convert.
* :doc:`units`: convert a flux between moles and mass with a molecular weight,
  and why pint's ``kt`` is not a kilotonne.
* :doc:`inventory`: why ``integrate()``, not ``total_emissions``, is the
  total, and how to move an inventory to another grid without losing mass.
* :doc:`maps`: add state lines, tick labels, the grid and a locator map (a
  small map of where it is) to a map, and draw any gridded field the same way.
* :doc:`soundings`: download soundings from Wyoming into a folder, read them
  on the same heights, and when ``get_soundings`` downloads.
* :doc:`hrrr_winds`: six hours of 10 m winds at the nearest HRRR grid point,
  asked for in UTC or in local time.
* :doc:`carbontracker`: download one day of CarbonTracker, sample it at points
  such as the ends of STILT back trajectories, and average them.
* :doc:`background`: lair's background methods side by side on one series,
  what each needs, and when to use which.

.. toctree::
   :maxdepth: 1
   :hidden:

   clock
   units
   inventory
   maps
   soundings
   hrrr_winds
   carbontracker
   background
