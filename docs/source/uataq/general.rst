.. currentmodule:: lair.uataq

General Functions
=================

.. autosummary::

  read_data
  get_obs

.. rubric:: Functions

.. autofunction:: read_data

.. autofunction:: get_obs

Input Parameters
----------------

SID
^^^

``SID`` is the site identifier for a UATAQ site.
It is a capitalized string that corresponds to a key in the configuration file.

Instruments
^^^^^^^^^^^

``instruments`` is a single instrument name or a list of instrument names.
The instrument name is a string that corresponds to a key in the ``lab.instruments`` list.

Pollutants
^^^^^^^^^^

``pollutants`` is a single pollutant name or a list of pollutant names.
The pollutant name is a capitalized molecule abbreviation.

Research Group
^^^^^^^^^^^^^^

``group`` is the research group that collected the data.
It is a string that corresponds to a key in the ``uataq.filesystem.groups`` dictionary.


Processing Level
^^^^^^^^^^^^^^^^^

``lvl`` is the processing level of the data.
Available levels are:

- ``raw`` : Raw data.
- ``qaqc`` : `QAQC flags <https://github.com/uataq/data-pipeline/blob/main/QAQC_Flags.md>`_ applied to data.
- ``calibrated`` : Calibrated data. (Only available for instruments which receive calibration in post-processing.)
- ``final`` : Finalized data. (Flagged data and measurements of calibration tanks are dropped.)

Time Range
^^^^^^^^^^

``time_range`` filters the returned data to the specified time range. 

There are three primary formats for ``time_range``:

1. ``None``: Returns all available data.
2. Single string in ISO8601 format down to the hour:

   - The string is interpreted as a range from the start of the string to the start of the next time unit. 
   - Examples:

     - '2020' represents the year 2020.
     - '2020-01' represents January 2020 to February 2020.
     - '2020-01-01' represents January 1st, 2020 to January 2nd, 2020.
     - '2020-01-01T12' represents January 1st, 2020 from 12:00 to 13:00.

3. List, tuple, or slice of two datetime-like objects:
    - Datetime-like objects include datetime objects, Timestamp objects, and strings in ISO8601 format.
    - The first object is the start of the range and the second object is the end of the range.
      The range is inclusive of the start and exclusive of the end.
    - The use of ``None`` in place of a datetime-like object will set the range to be unbounded in that direction.

Number of Processes
^^^^^^^^^^^^^^^^^^^

``num_processes`` is the number of processes to use when reading data from each instrument. The default is 1.
 - If ``num_processes`` is set to 1, the data is read serially.
 - Setting ``num_processes`` to a number greater than 1 will read the data in parallel using the minimum of ``num_processes`` and the number of files for an instrument.
 - Setting ``num_processeSs`` to 'max' will use the minimum of the number of files for an instrument and the number of available CPU cores.
    Warning: Frequent use of ``num_processes='max'`` may upset your fellow node users.

File Pattern
^^^^^^^^^^^^

``file_pattern`` is a string that is used to filter the files. The primary use for this parameter is to filter raw lin gps data by nmea sentence type. The default is ``None``.
