# Utah Atmospheric Trace-gas and Air Quality (UATAQ)

## UATAQ Lab

It all starts in the lab...

```python
from lair import uataq

lab = uataq.laboratory
```

The laboratory object is a singleton instance of the `Laboratory` class which is initialized with the [UATAQ configuration file](/lair/uataq/config.json). The configuration file is a JSON file which specifies UATAQ site characteristics including name, location, status, research groups collecting data, and installed instruments.

> Changes to UATAQ site infrastructure, including instrument installation/removal, must be reflected in the configuration file for the lab to be able to properly access the data.

The `Laboratory` object contains the following attributes:
 - sites : A list of site identifiers.
 - instruments : A list of instrument names.

## File System

The UATAQ subpackage operates on the idea that instrument data is collected differently by each research group, despite potentially being from the same site or instrument. Furthermore, the format in which data is written, and therefore read, is dependent on the system that logged the data rather than the instrument itself. To address this, UATAQ introduces a `filesystem` module which consists of `GroupSpace` and `DataFile` objects. The `GroupSpace` objects provide group-specific methods for working with data from that group, while the `DataFile` objects handle the actual parsing of the data files.

### Research Groups

Each research group has its own module in `uataq.filesystem.groupspaces` where group-specific code is stored. Each group module must contain a `GroupSpace` class that inherits from `uataq.filesystem.GroupSpace` and define `DataFile` classes that inherit from `uataq.filesystem.DataFile` for each file format that the group uses.

All `GroupSpace` objects are stored in the `uataq.filesystem.groups` dictionary with the group name as the key. Groups can also be accessed through `uataq.filesystem.get_group`.

> The default research group can be changed at runtime via `uataq.filesystem.DEFAULT_GROUP` or <i>permanently</i> (until the next update) changed in the [filesystem subpackage](/lair/uataq/filesystem/__init__.py).

## Research Sites

The `Site` object is the primary interface for accessing data from a UATAQ site. Each site has a unique site identifier (SID) that corresponds to a key in the configuration file. The `lab` is responsible for constructing `Site` objects from the configuration file, including building the `InstrumentEnsemble` for each site. The `InstrumentEnsemble` is a container object that hold different `Instrument` objects which provide the linkage between a `Site` and the data files.

```python
sites = lab.sites          # list of sites
wbb = lab.get_site('wbb')  # site object
```
> For convenience, `uataq.laboratory.get_site` is aliased as `uataq.get_site`

The `Site` object contains the following information as attributes:
 - SID : The site identifier.
 - config : A dictionary containing configuration information for the site from the config file.
 - instruments : An instance of the InstrumentEnsemble class representing the instruments at the site.
 - groups : The research groups that collect data at the site.
 - loggers : The loggers used by research groups that record data at a site.
 - pollutants : The pollutants measured at the site.

There are two primary methods for reading data from a site:
1. [Reading Instrument Data](#reading-instrument-data) - Data for each instrument at a site is read individually and stored in a dictionary with the instrument name as the key.
2. [Getting Observations](#getting-observations) - Finalized observations from all instruments at a site are aggregated into a single dataframe.

> `Site.read_data` and `Site.get_obs` have been wrapped in `uataq.read_data` and `uataq.get_obs` respectively for convenience with an added `SID` parameter.

## Reading Instrument Data

Using a `Site` object we can read the data from each instrument at the site for a specified processing lvl and time range:

```python
data = wbb.read_data(instruments='all', lvl='qaqc', time_range='2024')
```

The data is returned as a dictionary of pandas dataframes, one for each instrument. The dataframes are indexed by time and have columns for each variable:

```python
lgr_ugga = data['lgr_ugga']
lgr_ugga.head()
```
| Time_UTC            |   CH4_ppm |   CH4_ppm_sd |   H2O_ppm |   H2O_ppm_sd |   CO2_ppm |   CO2_ppm_sd |   CH4d_ppm |   CH4d_ppm_sd |   CO2d_ppm |   CO2d_ppm_sd |   Cavity_P_torr |   Cavity_P_torr_sd |   Cavity_T_C |   Cavity_T_C_sd |   Ambient_T_C |   Ambient_T_C_sd |   RD0_us |   RD0_us_sd |   RD1_us |   RD1_us_sd |   Fit_Flag | ID                   |   ID_CO2 |   ID_CH4 |   QAQC_Flag |
|:--------------------|----------:|-------------:|----------:|-------------:|----------:|-------------:|-----------:|--------------:|-----------:|--------------:|----------------:|-------------------:|-------------:|----------------:|--------------:|-----------------:|---------:|------------:|---------:|------------:|-----------:|:---------------------|---------:|---------:|------------:|
| 2024-01-01 00:00:00 |   2.39506 |   0.00255038 |   4986.81 |      60.8709 |   476.078 |     0.289434 |    2.40706 |    0.00268069 |    478.464 |      0.26705  |         140.206 |          0.0281737 |      24.2379 |     0.000296853 |       26.2806 |       0.00274172 |  9.56621 |  0.00704373 |  11.2176 |  0.0070454  |          3 | ~atmospher~atmospher |      -10 |      -10 |           0 |
| 2024-01-01 00:00:10 |   2.39479 |   0.0027215  |   4986.79 |      32.2532 |   475.82  |     0.243144 |    2.40679 |    0.00275197 |    478.205 |      0.249527 |         140.2   |          0.0256384 |      24.2371 |     0.000296624 |       26.2727 |       0.00503834 |  9.56404 |  0.0025944  |  11.2166 |  0.00534878 |          3 | ~atmospher~atmospher |      -10 |      -10 |           0 |
| 2024-01-01 00:00:20 |   2.39244 |   0.0014658  |   4995.93 |      65.1538 |   475.643 |     0.302123 |    2.40445 |    0.00150599 |    478.031 |      0.292054 |         140.216 |          0.0228126 |      24.2365 |     0.000374716 |       26.2632 |       0.00206392 |  9.56353 |  0.00292836 |  11.215  |  0.00612329 |          3 | ~atmospher~atmospher |      -10 |      -10 |           0 |
| 2024-01-01 00:00:30 |   2.39194 |   0.00294245 |   4997.65 |      60.0363 |   475.499 |     0.245503 |    2.40395 |    0.00302519 |    477.887 |      0.228847 |         140.215 |          0.0210461 |      24.2358 |     0.000361223 |       26.2635 |       0.00312633 |  9.5668  |  0.0051438  |  11.2152 |  0.0058915  |          3 | ~atmospher~atmospher |      -10 |      -10 |           0 |
| 2024-01-01 00:00:39 |   2.3915  |   0.0023981  |   5003.47 |      71.197  |   475.613 |     0.25103  |    2.40352 |    0.00252292 |    478.005 |      0.226779 |         140.203 |          0.0234361 |      24.2353 |     0.000507074 |       26.2581 |       0.0027034  |  9.564   |  0.00773102 |  11.2143 |  0.00730825 |          3 | ~atmospher~atmospher |      -10 |      -10 |           0 |

## Getting Observations

Or we can only get the finalized observations for a site which aggregates the instruments into a single dataframe:

```python
obs = wbb.get_obs(pollutants=['CO2', 'CH4', 'O3', 'NO2', 'NO', 'CO'],
                  time_range=['2024-02-08', None])
obs.head()
```
| Time_UTC                   |   NO_ppb |   NO2_ppb |   NOx_ppb |   CO2d_ppm_cal |   CO2d_ppm_raw |   CH4d_ppm_cal |   CH4d_ppm_raw |   O3_ppb |   CO_ppb |
|:---------------------------|---------:|----------:|----------:|---------------:|---------------:|---------------:|---------------:|---------:|---------:|
| 2024-02-08 00:00:01        |    nan   |       nan |     nan   |        nan     |        nan     |       nan      |      nan       |     24.6 |  nan     |
| 2024-02-08 00:00:01.010000 |    nan   |       nan |     nan   |        nan     |        nan     |       nan      |      nan       |    nan   |  529.643 |
| 2024-02-08 00:00:01.040000 |      2.1 |        10 |      12.1 |        nan     |        nan     |       nan      |      nan       |    nan   |  nan     |
| 2024-02-08 00:00:02        |    nan   |       nan |     nan   |        431.211 |        420.703 |         2.0114 |        2.00252 |    nan   |  nan     |
| 2024-02-08 00:00:03.140000 |    nan   |       nan |     nan   |        nan     |        nan     |       nan      |      nan       |     24.5 |  nan     |

Finalized observations only include data which has passed QAQC (`QAQC_Flag >= 0`) and that are measurements of the ambient atmosphere (`ID == -10`). The observations dataframe is indexed by time and aggregates pollutants into a single dataframe. Two formats are available: `wide` or `long`. The `wide` format has columns for each pollutant and the `long` format has a `pollutant` column with the pollutant name and a `value` column with the measurement value.

```python
obs_long = wbb.get_obs(pollutants=['CO2', 'CH4', 'O3', 'NO2', 'NO', 'CO'],
                       time_range=['2024-02-08', None],
                       format='long')
obs_long.head(10)
```
| Time_UTC                   | pollutant    |     value |
|:---------------------------|:-------------|----------:|
| 2024-02-08 00:00:01        | O3_ppb       |  24.6     |
| 2024-02-08 00:00:01.010000 | CO_ppb       | 529.643   |
| 2024-02-08 00:00:01.040000 | NO2_ppb      |  10       |
| 2024-02-08 00:00:01.040000 | NOx_ppb      |  12.1     |
| 2024-02-08 00:00:01.040000 | NO_ppb       |   2.1     |
| 2024-02-08 00:00:02        | CH4d_ppm_raw |   2.00252 |
| 2024-02-08 00:00:02        | CH4d_ppm_cal |   2.0114  |
| 2024-02-08 00:00:02        | CO2d_ppm_raw | 420.703   |
| 2024-02-08 00:00:02        | CO2d_ppm_cal | 431.211   |
| 2024-02-08 00:00:03.140000 | O3_ppb       |  24.5     |

### Mobile Sites & Observations

Included as part of UATAQ is the TRAX/eBus project, which collects data from mobile sites. The `MobileSite` object is a subclass of the `Site` object. The `laboratory` determines whether to build a `Site` or `MobileSite` object based on the `is_mobile` attribute in the configuration file.

Mobile sites provide the same functionality as fixed sites, but merge location data with observations when using the `get_obs` method and return a geodataframe.

```python
trx01 = lab.get_site('TRX01')
mobile_data = trx01.get_obs(group='horel', time_range=['2019', '2021'])
mobile_data.head()
```
| Time_UTC            |   O3_ppb |   PM2.5_ugm3 |   Latitude_deg |   Longitude_deg | geometry                      |
|:--------------------|---------:|-------------:|---------------:|----------------:|:------------------------------|
| 2019-01-01 00:00:00 |     30.4 |          nan |        40.7696 |        -111.839 | POINT (-111.838913 40.769608) |
| 2019-01-01 00:00:00 |    nan   |            2 |        40.7696 |        -111.839 | POINT (-111.838913 40.769608) |
| 2019-01-01 00:00:02 |    nan   |            2 |        40.7696 |        -111.839 | POINT (-111.838913 40.769608) |
| 2019-01-01 00:00:02 |     27.6 |          nan |        40.7696 |        -111.839 | POINT (-111.838913 40.769608) |
| 2019-01-01 00:00:04 |     27.2 |          nan |        40.7696 |        -111.839 | POINT (-111.838913 40.769608) |

Or in the long format:
```python
mobile_data_long = trx01.get_obs(group='horel', time_range=['2019', '2021'], format='long')
mobile_data_long.head()
```
| Time_UTC            | pollutant   |   value |   Latitude_deg |   Longitude_deg | geometry                      |
|:--------------------|:------------|--------:|---------------:|----------------:|:------------------------------|
| 2019-01-01 00:00:00 | O3_ppb      |    30.4 |        40.7696 |        -111.839 | POINT (-111.838913 40.769608) |
| 2019-01-01 00:00:00 | PM2.5_ugm3  |     2   |        40.7696 |        -111.839 | POINT (-111.838913 40.769608) |
| 2019-01-01 00:00:02 | PM2.5_ugm3  |     2   |        40.7696 |        -111.839 | POINT (-111.838913 40.769608) |
| 2019-01-01 00:00:02 | O3_ppb      |    27.6 |        40.7696 |        -111.839 | POINT (-111.838913 40.769608) |
| 2019-01-01 00:00:04 | O3_ppb      |    27.2 |        40.7696 |        -111.839 | POINT (-111.838913 40.769608) |

## Naming Convention

I chose `UATAQ` as the name for the package because it is the most encompassing name for the groups currently involved in the project.

Designing a user-friendly package is a challenge because the data is collected by multiple research groups, each with their own naming conventions and data formats. The package must be able to handle all of these different formats and provide a consistent interface for the user.

I have defined a set of [standardized column names](/lair/uataq/columns.md) that each groupspace module must define a `column_mapping` dictionary that maps the group's column names to the standardized names when using the `GroupSpace.standardize_data` method.

## Input Parameters

### SID
`SID` is the site identifier for a UATAQ site. It is a capitalized string that corresponds to a key in the configuration file.

### Instruments
`instruments` is a single instrument name or a list of instrument names. The instrument name is a string that corresponds to a key in the `lab.instruments` list.

### Pollutants
`pollutants` is a single pollutant name or a list of pollutant names. The pollutant name is a capitalized molecule abbreviation.

### Research Group
`group` is the research group that collected the data. It is a string that corresponds to a key in the `uataq.filesystem.groups` dictionary.

### Processing Level
`lvl` is the processing level of the data.
Available levels are:
 - `raw` : Raw data.
 - `qaqc` : [QAQC flags](https://github.com/uataq/data-pipeline/blob/main/QAQC_Flags.md) applied to data.
 - `calibrated` : Calibrated data. (Only available for instruments which receive calibration in post-processing.)
 - `final` : Finalized data. (Flagged data and measurements of calibration tanks are dropped.)

### Time Range
`time_range` filters the returned data to the specified time range. 

There are three primary formats for `time_range`:

1. `None`: Returns all available data.

2. Single string in ISO8601 format down to the hour:
    - The string is interpreted as a range from the start of the string to the start of the next time unit. 
    - Examples: 
      - '2020' represents the year 2020.
      - '2020-01' represents January 2020 to February 2020.
      - '2020-01-01' represents January 1st, 2020 to January 2nd, 2020.
      - '2020-01-01T12' represents January 1st, 2020 from 12:00 to 13:00.

3. List, tuple, or slice of two datetime-like objects:
    - Datetime-like objects include datetime objects, Timestamp objects, and strings in ISO8601 format.
    - The first object is the start of the range and the second object is the end of the range. The range is inclusive of the start and exclusive of the end.
    - The use of `None` in place of a datetime-like object will set the range to be unbounded in that direction.

### Number of Processes
`num_processes` is the number of processes to use when reading data from each instrument. The default is 1.
 - If `num_processes` is set to 1, the data is read serially.
 - Setting `num_processes` to a number greater than 1 will read the data in parallel using the minimum of `num_processes` and the number of files for an instrument.
 - Setting `num_processes` to 'max' will use the minimum of the number of files for an instrument and the number of available CPU cores.
 > Warning: Frequent use of `num_processes='max'` may upset your fellow node users.

### File Pattern
`file_pattern` is a string that is used to filter the files. The primary use for this parameter is to filter raw lin gps data by nmea sentence type. The default is `None`.
