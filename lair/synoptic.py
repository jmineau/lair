"""
Surface station observations from the Synoptic Data API (the MesoWest successor).

Three calls cover most uses:

- :func:`metadata` -- which stations exist for a query (``stid``, ``bbox``,
  ``state``, ``network``, ``vars``, ...), with location and period of record.
- :func:`timeseries` -- the native observations (often 5-min) for a query and
  time window, one row per station and time.
- :func:`hourly_mean` -- hourly means of a :func:`timeseries` frame, with wind
  averaged as a *vector* (``Uwind``/``Vwind``) and the direction taken from the
  mean vector. Averaging wind direction in degrees is wrong whenever an hour
  straddles north (350 and 10 deg average to 180).

Query keywords are passed straight to the API, so anything the Synoptic docs list
works (https://docs.synopticdata.com/services/). The token comes from ``token=``
or the ``SYNOPTIC_TOKEN`` environment variable, read at call time. Requires the
``requests`` extra. Long periods should be requested in chunks (e.g. a month at
a time); the API caps the size of one response.
"""

from __future__ import annotations

import os

import pandas as pd

from lair._optional import import_optional_dependency
from lair.air import wind_components, wind_direction

BASE_URL = "https://api.synopticdata.com/v2"


class SynopticError(RuntimeError):
    """The API answered, but with an error (bad token, no access, no stations...)."""


def _token(token: str | None) -> str:
    token = token or os.environ.get("SYNOPTIC_TOKEN")
    if not token:
        raise ValueError(
            "No Synoptic token: pass token= or set the SYNOPTIC_TOKEN environment variable."
        )
    return token


def _time(t) -> str:
    """Format a time as the API takes it: YYYYmmddHHMM in UTC."""
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y%m%d%H%M")


def _get(endpoint: str, params: dict, token: str | None, timeout: float) -> dict:
    requests = import_optional_dependency("requests")
    params = {k: v for k, v in params.items() if v is not None}
    params["token"] = _token(token)
    response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    summary = payload.get("SUMMARY", {})
    if summary.get("RESPONSE_CODE") != 1:
        raise SynopticError(summary.get("RESPONSE_MESSAGE", "unknown Synoptic error"))
    return payload


def metadata(token: str | None = None, timeout: float = 120, **query) -> pd.DataFrame:
    """
    Stations matching a query, one row each.

    Parameters
    ----------
    token : str, optional
        API token; defaults to ``$SYNOPTIC_TOKEN``.
    timeout : float
        Seconds to wait for the response.
    **query
        API query, e.g. ``bbox="-112.2,40.4,-111.7,40.9"``, ``stid="WBB,TRJO"``,
        ``vars="wind_speed"``, ``complete=1``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``stid``: ``name``, ``latitude``, ``longitude``,
        ``elevation_ft``, ``network_id``, ``status``, ``record_start``,
        ``record_end`` (UTC, naive).

    See Also
    --------
    timeseries : The observations of the stations it finds.

    Examples
    --------
    Needs a token and the internet:

    >>> from lair.synoptic import metadata
    >>> stations = metadata(bbox="-112.2,40.4,-111.7,40.9")  # doctest: +SKIP
    >>> stations[["name", "latitude", "longitude"]].head()  # doctest: +SKIP
    """
    payload = _get("stations/metadata", query, token, timeout)
    rows = []
    for st in payload.get("STATION", []):
        period = st.get("PERIOD_OF_RECORD") or {}
        rows.append(
            {
                "stid": st.get("STID"),
                "name": st.get("NAME"),
                "latitude": pd.to_numeric(st.get("LATITUDE"), errors="coerce"),
                "longitude": pd.to_numeric(st.get("LONGITUDE"), errors="coerce"),
                "elevation_ft": pd.to_numeric(st.get("ELEVATION"), errors="coerce"),
                "network_id": st.get("MNET_ID"),
                "status": st.get("STATUS"),
                "record_start": period.get("start"),
                "record_end": period.get("end"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("record_start", "record_end"):
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
    return df.set_index("stid").sort_index()


def timeseries(
    start,
    end,
    token: str | None = None,
    timeout: float = 300,
    units: str = "metric",
    **query,
) -> pd.DataFrame:
    """
    Native observations for every station matching a query, in one long frame.

    Parameters
    ----------
    start, end
        Window, anything ``pd.Timestamp`` parses; naive times follow
        :ref:`the UTC rule <utc-rule>`.
    token : str, optional
        API token; defaults to ``$SYNOPTIC_TOKEN``.
    timeout : float
        Seconds to wait for the response.
    units : str
        API ``units`` (``"metric"``: m/s, deg C).
    **query
        API query, e.g. ``stid="TRJO"`` or ``bbox=...``, and
        ``vars="wind_speed,wind_direction"``.

    Returns
    -------
    pd.DataFrame
        Columns ``stid``, ``Time`` (UTC, tz-aware) and one column per returned
        variable in the API's naming (``wind_speed_set_1``, ...). Stations with no
        observations in the window are omitted.

    See Also
    --------
    metadata : Which stations exist for a query.
    hourly_mean : Averages the frame by hour, with the wind as a vector.

    Examples
    --------
    Needs a token and the internet:

    >>> from lair.synoptic import hourly_mean, timeseries
    >>> obs = timeseries(
    ...     "2024-01-15 12:00",
    ...     "2024-01-15 18:00",
    ...     stid="TRJO",
    ...     vars="wind_speed,wind_direction",
    ... )  # doctest: +SKIP
    >>> hourly = hourly_mean(obs)  # doctest: +SKIP
    """
    params = dict(query, start=_time(start), end=_time(end), units=units)
    payload = _get("stations/timeseries", params, token, timeout)
    frames = []
    for st in payload.get("STATION", []):
        obs = st.get("OBSERVATIONS") or {}
        if not obs.get("date_time"):
            continue
        df = pd.DataFrame(obs).rename(columns={"date_time": "Time"})
        df["Time"] = pd.to_datetime(df["Time"], utc=True)
        for col in df.columns.drop("Time"):
            df[col] = _numeric_if_possible(df[col])
        df.insert(0, "stid", st.get("STID"))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stid", "Time"])
    return pd.concat(frames, ignore_index=True)


def _numeric_if_possible(values: pd.Series) -> pd.Series:
    """
    Numbers where every non-missing value parses as one.

    Some stations send numeric strings, e.g. '0.51', mixed with floats, and '' for
    missing. Text columns such as cardinal wind directions are left alone.
    """
    blank = values.map(lambda x: isinstance(x, str) and not x.strip())
    values = values.where(~blank.astype(bool))
    converted = pd.to_numeric(values, errors="coerce")
    if converted.notna().sum() == values.notna().sum():
        return converted
    return values


def hourly_mean(
    df: pd.DataFrame,
    speed: str = "wind_speed_set_1",
    direction: str = "wind_direction_set_1",
) -> pd.DataFrame:
    """
    Hourly means of a :func:`timeseries` frame, with wind averaged as a vector.

    Numeric columns are averaged over each clock hour (labeled by its start).
    Where ``speed`` and ``direction`` exist, ``Uwind``/``Vwind`` are computed per
    observation and averaged, and ``direction`` is replaced by the direction of
    the mean vector (NaN for a dead-calm hour); ``speed`` stays the scalar mean.
    ``n_obs`` counts the wind observations in the hour. Hours with no data are
    dropped. Works per station when a ``stid`` column is present.

    Parameters
    ----------
    df : pd.DataFrame
        Observations as :func:`timeseries` returns them, with a ``Time``
        column.
    speed : str, optional
        Column of wind speed. Default ``"wind_speed_set_1"``.
    direction : str, optional
        Column of wind direction (degrees, blowing from). Default
        ``"wind_direction_set_1"``.

    Returns
    -------
    pd.DataFrame
        ``stid`` (if given), ``Time`` and the averaged columns.

    See Also
    --------
    timeseries : Returns the frame it takes.
    lair.air.wind_components : Splits a wind into u and v.
    lair.air.wind_direction : Turns u and v back into a direction.

    Examples
    --------
    Two observations in one hour, from 350 and 10 degrees, average to a wind
    from the north (0 degrees), where a mean of the two directions gives 180:

    >>> import pandas as pd
    >>> from lair.synoptic import hourly_mean
    >>> obs = pd.DataFrame(
    ...     {
    ...         "Time": pd.to_datetime(
    ...             ["2024-01-15 12:00", "2024-01-15 12:30"], utc=True
    ...         ),
    ...         "wind_speed_set_1": [4.0, 4.0],
    ...         "wind_direction_set_1": [350.0, 10.0],
    ...     }
    ... )
    >>> hourly = hourly_mean(obs)
    >>> hourly[["wind_speed_set_1", "wind_direction_set_1", "n_obs"]].round(1)
       wind_speed_set_1  wind_direction_set_1  n_obs
    0               4.0                   0.0      2
    """
    if "stid" in df.columns:
        parts = [
            _hourly_one(g.drop(columns="stid"), speed, direction).assign(stid=stid)
            for stid, g in df.groupby("stid", sort=True)
        ]
        if not parts:
            return pd.DataFrame(columns=["stid", "Time"])
        out = pd.concat(parts, ignore_index=True)
        return out[["stid"] + [c for c in out.columns if c != "stid"]]
    return _hourly_one(df, speed, direction)


def _hourly_one(df: pd.DataFrame, speed: str, direction: str) -> pd.DataFrame:
    data = df.set_index("Time").sort_index()
    data = data.apply(pd.to_numeric, errors="coerce")
    has_wind = speed in data.columns and direction in data.columns
    if has_wind:
        data["Uwind"], data["Vwind"] = wind_components(data[speed], data[direction])
    hourly = data.resample("1h").mean()
    if has_wind:
        hourly[direction] = wind_direction(hourly["Uwind"], hourly["Vwind"])
        hourly["n_obs"] = data[speed].resample("1h").count()
    values = hourly.drop(columns="n_obs", errors="ignore")
    hourly = hourly[values.notna().any(axis=1)]
    return hourly.reset_index()


__all__ = ["BASE_URL", "SynopticError", "hourly_mean", "metadata", "timeseries"]
