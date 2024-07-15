.. currentmodule:: lair.uataq

Utah Atmospheric Trace-gas and Air Quality (UATAQ)
==================================================

.. toctree::
   :maxdepth: 1

   quickstart
   general
   laboratory
   filesystem
   sites
   instruments

Naming Convention
-----------------

I chose `UATAQ` as the name for the package because it is the most encompassing
name for the groups currently involved in the project.

Designing a user-friendly package is a challenge because the data is collected
by multiple research groups, each with their own naming conventions and data formats.
The package must be able to handle all of these different formats and provide a 
onsistent interface for the user.

I have defined a set of [standardized column names](/lair/uataq/columns.md) that each
groupspace module must define a :obj:`column_mapping` dictionary that maps the group's column
names to the standardized names when using the `GroupSpace.standardize_data` method.

Contents
--------

.. autosummary::

   laboratory
   get_site
   read_data
   get_obs
