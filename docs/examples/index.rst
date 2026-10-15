Examples
========

Worked examples, one per job, each from loading the data to a checked result.
Each page says whether it runs when the docs build.

* :doc:`clock`: move UTC data to local time and average it by hour and by
  season, and what to do when a time will not convert.
* :doc:`units`: convert a flux between moles and mass with a molecular weight,
  and why pint's ``kt`` is not a kilotonne.
* :doc:`inventory`: total an emissions inventory over a region, and move it to
  another grid without losing mass.
* :doc:`maps`: draw a map of an inventory, with state lines, tick labels, the
  grid and a small map of where it is, and the same for any gridded field.
* :doc:`soundings`: download soundings, read them, and compute potential
  temperature.
* :doc:`hrrr_winds`: HRRR winds at a point.
* :doc:`carbontracker`: CarbonTracker's background at a set of points.
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
