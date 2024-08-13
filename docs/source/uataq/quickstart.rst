.. currentmodule:: lair.uataq

Quick Start
===========

.. ipython:: python
    :suppress:

    import lair

    # Set verbose to False to suppress logging
    lair.config.verbose = False

Laboratory
----------

It all starts in the lab...

.. ipython:: python

   from lair import uataq

   lab = uataq.laboratory

The :data:`laboratory` object is a singleton instance of the :class:`~lair.uataq._laboratory.Laboratory`
class which is initialized with the :doc:`UATAQ configuration file <config>`.
The configuration file is a JSON file which specifies UATAQ site characteristics
including name, location, status, research groups collecting data, and installed instruments.

The :class:`~lair.uataq._laboratory.Laboratory` object contains the following attributes:

 - sites : A list of site identifiers.
 - instruments : A list of instrument names.

Research Sites
--------------

The :class:`~sites.Site` object is the primary interface for accessing data from a UATAQ site.
Each site has a unique site identifier (SID) that corresponds to a key in the configuration file.
The :data:`lab <laboratory>` is responsible for constructing :class:`~sites.Site` objects from the configuration file,
including building the :class:`~instruments.InstrumentEnsemble` for each site.
The :class:`~instruments.InstrumentEnsemble` is a container object that hold different
:class:`~instruments.Instrument` objects which provide the linkage between a :class:`~sites.Site` and the data files.

.. ipython:: python

    sites = lab.sites          # list of sites
    wbb = lab.get_site('wbb')  # site object

..  Force code block to end

    For convenience, :meth:`uataq.laboratory.get_site` is aliased as :meth:`uataq.get_site`

The :class:`~sites.Site` object contains the following information as attributes:

 - SID : The site identifier.
 - config : A dictionary containing configuration information for the site from the config file.
 - instruments : An instance of the InstrumentEnsemble class representing the instruments at the site.
 - groups : The research groups that collect data at the site.
 - loggers : The loggers used by research groups that record data at a site.
 - pollutants : The pollutants measured at the site.

There are two primary methods for reading data from a site:

1. :ref:`Reading Instrument Data` - Data for each instrument at a site is read individually and stored in a dictionary with the instrument name as the key.
2. :ref:`Getting Observations` - Finalized observations from all instruments at a site are aggregated into a single dataframe.

    :meth:`Site.read_data` and :meth:`Site.get_obs` have been wrapped in
    :meth:`uataq.read_data` and :meth:`uataq.get_obs` respectively for convenience
    with an added `SID` parameter.

Reading Instrument Data
-----------------------

Using a :class:`~sites.Site` object we can read the data from each instrument
at the site for a specified processing lvl and time range:

.. ipython:: python

    data = wbb.read_data(instruments='all', lvl='qaqc', time_range='2024')

The data is returned as a dictionary of pandas dataframes, one for each instrument.
The dataframes are indexed by time and have columns for each variable:

.. ipython:: python

    lgr_ugga = data['lgr_ugga']
    lgr_ugga.head()

Getting Observations
--------------------

Or we can only get the finalized observations for a site which aggregates
the instruments into a single dataframe:

.. ipython:: python

    obs = wbb.get_obs(pollutants=['CO2', 'CH4', 'O3', 'NO2', 'NO', 'CO'],
                    time_range=['2024-02-08', None])
    obs.head()

Finalized observations only include data which has passed QAQC (``QAQC_Flag >= 0``)
and that are measurements of the ambient atmosphere (``ID == -10``).
The observations dataframe is indexed by time and aggregates pollutants into a single dataframe.
Two formats are available: ``wide`` or ``long``.
The ``wide`` format has columns for each pollutant and
the ``long`` format has a ``pollutant`` column with the pollutant name
and a ``value`` column with the measurement value.

.. ipython:: python

    obs_long = wbb.get_obs(pollutants=['CO2', 'CH4', 'O3', 'NO2', 'NO', 'CO'],
                        time_range=['2024-02-08', None],
                        format='long')
    obs_long.head(10)

Mobile Sites & Observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Included as part of UATAQ is the TRAX/eBus project, which collects data from mobile sites.
The :class:`~sites.MobileSite` object is a subclass of the :class:`~sites.Site` object.
The :obj:`laboratory` determines whether to build a :class:`~sites.Site` or
:class:`~sites.MobileSite` object based on the ``is_mobile`` attribute in the configuration file.

Mobile sites provide the same functionality as fixed sites, but merge location
data with observations when using the :meth:`~sites.MobileSite.get_obs` method
and return a geodataframe.

.. ipython:: python

    trx01 = lab.get_site('TRX01')
    mobile_data = trx01.get_obs(group='horel', time_range=['2019', '2021'])
    mobile_data.head()

Or in the long format:

.. ipython:: python

    mobile_data_long = trx01.get_obs(group='horel', time_range=['2019', '2021'], format='long')
    mobile_data_long.head()
