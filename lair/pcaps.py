"""
Valley heat deficit and persistent cold air pools.

:func:`valleyheatdeficit` integrates a sounding's potential-temperature deficit
from the valley floor to a height (Whiteman et al., 2014), and
:func:`determine_pcap_events` finds the persistent cold air pool (PCAP) events,
the periods when the deficit stays above a threshold. :func:`build_pcap_mask`
and :func:`filter_pcap_events` mark or drop the times inside them.
"""

import numpy as np
import pandas as pd
import xarray as xr

from lair.constants import Rd as _Rd
from lair.constants import cp as _cp
from lair.meteorology import hypsometric, ideal_gas_law, poisson

# Plain SI floats (lair.meteorology is plain SI in, plain SI out)
Rd = float(_Rd.m_as("J / kg / K"))
cp = float(_cp.m_as("J / kg / K"))


def valleyheatdeficit(data: xr.Dataset, integration_height=2200) -> xr.DataArray:
    """
    Calculate the valley heat deficit.

    Whiteman, C. David, et al. “Relationship between Particulate Air Pollution
    and Meteorological Variables in Utah's Salt Lake Valley.”
    Atmospheric Environment, vol. 94, Sept. 2014, pp. 742-53.
    DOI.org (Crossref), https://doi.org/10.1016/j.atmosenv.2014.06.012.

    Parameters
    ----------
    data : xr.Dataset
        The soundings, as :func:`lair.soundings.get_soundings` returns them.
    integration_height : int
        The height to integrate to [m].

    Returns
    -------
    pd.Series
        The valley heat deficit [MJ/m^2] named ``VHD_MJ_m2``, indexed by
        ``Time_UTC``.

    See Also
    --------
    determine_pcap_events : Finds the events where it stays high.
    lair.soundings.get_soundings : Reads the soundings it takes.
    lair.meteorology.poisson : The potential temperature it integrates.

    Notes
    -----
    ``data`` needs temperature in degrees Celsius, pressure in hPa, a ``height``
    dimension, and the ``elevation`` and ``interpolation_interval`` attributes
    that ``get_soundings`` sets.

    Examples
    --------
    Two made-up soundings, a cold pool (temperature falling 2 K per kilometer)
    and a well-mixed one (6.5 K per kilometer), 12 hours apart:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import xarray as xr
    >>> from lair.pcaps import valleyheatdeficit
    >>> height = np.arange(1289.0, 3300.0, 10.0)
    >>> temperature = np.vstack(
    ...     [-5 - 2.0 * (height - 1289) / 1000, 0 - 6.5 * (height - 1289) / 1000]
    ... )
    >>> pressure = np.tile(870 * np.exp(-(height - 1289) / 8000), (2, 1))
    >>> soundings = xr.Dataset(
    ...     {
    ...         "temperature": (("time", "height"), temperature),
    ...         "pressure": (("time", "height"), pressure),
    ...     },
    ...     coords={
    ...         "time": pd.to_datetime(["2024-01-15 00:00", "2024-01-15 12:00"]),
    ...         "height": height,
    ...     },
    ...     attrs={"elevation": 1289.0, "interpolation_interval": 10},
    ... )
    >>> valleyheatdeficit(soundings).round(1)
    Time_UTC
    2024-01-15 00:00:00    3.6
    2024-01-15 12:00:00    1.5
    Name: VHD_MJ_m2, dtype: float64
    """
    h0 = data.elevation

    # Subset to the heights between the surface and the integration height
    data = data.sel(height=slice(h0, integration_height))

    # lair.meteorology works in plain SI: convert the sounding units here
    T = data.temperature + 273.15  # degC -> K
    p = data.pressure * 100  # hPa -> Pa

    # Calculate potential temperature using poisson's equation
    theta = poisson(T=T, p=p, p0=1e5)  # K
    theta_h = theta.sel(height=integration_height, method="nearest")

    # Calculate virtual temperature to account for water vapor: the
    # hypsometric equation gives the mean Tv of each layer between levels
    Tv_layer = hypsometric(
        p1=p.isel(height=slice(0, -1)).values,
        p2=p.isel(height=slice(1, None)).values,
        deltaz=data.interpolation_interval,  # m
    )  # K
    # Layer means sit at the layer midpoints. Average adjacent layers to get Tv
    # at the levels (linear in height), and use the end layers' means at the
    # surface and top levels so they aren't lost
    Tv = np.concatenate(
        [
            Tv_layer[:, :1],
            (Tv_layer[:, :-1] + Tv_layer[:, 1:]) / 2,
            Tv_layer[:, -1:],
        ],
        axis=1,
    )
    Tv = xr.DataArray(Tv, coords=T.coords, dims=T.dims)

    # Calculate the density using the ideal gas law
    rho = ideal_gas_law(solve_for="rho", p=p, T=Tv, R=Rd)  # kg/m3

    # Calculate the heat deficit by integrating using the trapezoid method
    integrand = (cp * rho * (theta_h - theta)).dropna("height", how="all")
    heat_deficit = integrand.integrate("height")  # J/m2

    heat_deficit = heat_deficit * 1e-6  # J/m2 -> MJ/m2
    # A sounding with no valid levels integrates to 0; report it as missing
    heat_deficit = heat_deficit.where(integrand.notnull().any("height"))

    vhd = heat_deficit.to_series()
    vhd.name = "VHD_MJ_m2"
    vhd.index.name = "Time_UTC"
    return vhd


def determine_pcap_events(
    vhd: pd.Series, threshold: float, min_periods: int = 3
) -> pd.DataFrame:
    """
    Determine persistent cold air pool (PCAP) events from the valley heat deficit.

    An event is a run of 12-hourly soundings whose valley heat deficit (VHD) is
    above ``threshold``, lasting at least ``min_periods`` soundings. The run is
    counted in time, on the 12-hourly grid from the first time in ``vhd`` to the
    last, not in rows: a sounding missing from ``vhd`` is a gap in the run, as
    is a VHD that is NaN.

    A gap of one sounding between two soundings above the threshold is taken as
    above, and counts toward ``min_periods``. A longer gap, or one next to a
    sounding at or below the threshold, ends the event.

    Parameters
    ----------
    vhd : pd.Series
        The valley heat deficit [MJ/m^2], indexed by time, one value every 12
        hours (the output of :func:`valleyheatdeficit`). Soundings may be
        missing or NaN.
    threshold : float
        A sounding is in an event when its VHD is above this value [MJ/m^2].
        Whiteman et al. (2014) use 4.04.
    min_periods : int, optional
        The fewest soundings an event lasts, a gap of one included. Default
        is 3 (36 hours).

    Returns
    -------
    pd.DataFrame
        One row per event, with its ``start`` (the first sounding) and ``end``
        (12 hours after the last sounding, less one second), indexed by
        ``event_id`` from 0.

    Raises
    ------
    ValueError
        If a time in ``vhd`` is not a whole number of 12-hour steps from the
        first, or a time is repeated.

    Examples
    --------
    The missing 2024-01-02 12:00 sounding is one gap between two soundings
    above the threshold, so the event runs through it:

    >>> import pandas as pd
    >>> times = pd.to_datetime(
    ...     ["2024-01-01 12:00", "2024-01-02 00:00", "2024-01-03 00:00"]
    ... )
    >>> vhd = pd.Series([5.0, 6.0, 5.5], index=times)
    >>> determine_pcap_events(vhd, threshold=4.04)  # doctest: +NORMALIZE_WHITESPACE
                           start                 end
    event_id
    0        2024-01-01 12:00:00 2024-01-03 11:59:59
    """
    if vhd.empty:
        return pd.DataFrame(
            {"start": pd.DatetimeIndex([]), "end": pd.DatetimeIndex([])}
        ).rename_axis("event_id")
    if vhd.index.has_duplicates:
        repeated = vhd.index[vhd.index.duplicated()][0]
        raise ValueError(f"vhd has more than one value at {repeated}.")
    vhd = vhd.sort_index()
    step = pd.Timedelta(hours=12)  # the time between two soundings

    # Put the series on the 12-hourly grid, so a missing sounding is a NaN
    times = pd.date_range(vhd.index.min(), vhd.index.max(), freq=step)
    off_grid = vhd.index.difference(times)
    if len(off_grid) > 0:
        raise ValueError(
            "vhd must be 12-hourly: "
            f"{off_grid[0]} is not a whole number of 12-hour steps "
            f"from the first time, {vhd.index.min()}."
        )
    vhd = vhd.reindex(times)

    above = vhd > threshold  # NaN is not above
    # One missing or NaN sounding between two above the threshold is taken as
    # above. Both neighbors above means neither is missing, so the gap is one.
    gap = (
        vhd.isna()
        & above.shift(1, fill_value=False)
        & above.shift(-1, fill_value=False)
    )
    in_event = above | gap

    # Number the runs of consecutive grid times in or out of an event
    run = (in_event != in_event.shift()).cumsum()
    event_times = times.to_series()[in_event].groupby(run[in_event].to_numpy())
    pcap_events = pd.DataFrame(
        {
            "start": event_times.min(),
            "end": event_times.max(),
            "periods": event_times.size(),
        }
    )
    pcap_events = pcap_events[pcap_events["periods"] >= min_periods]
    pcap_events = pcap_events[["start", "end"]].reset_index(drop=True)
    # inclusive of the following 12 hours, less a second
    pcap_events["end"] += step - pd.Timedelta(seconds=1)
    pcap_events.index.name = "event_id"

    return pcap_events


def _naive_utc(t):
    """Convert a tz-aware Timestamp/DatetimeIndex to naive UTC (naive passes through)."""
    return t.tz_convert("UTC").tz_localize(None) if t.tz is not None else t


def build_pcap_mask(index: pd.Index, events: pd.DataFrame) -> pd.Series:
    """
    Build a boolean mask for PCAP events.

    Parameters
    ----------
    index : pd.Index
        The datetime-like index to build the mask for. Naive times follow
        :ref:`the UTC rule <utc-rule>`; tz-aware times are converted.
    events : pd.DataFrame
        The DataFrame of PCAP events (naive UTC or tz-aware start/end).

    Returns
    -------
    pd.Series
        A boolean mask where True indicates a time within a PCAP event.

    See Also
    --------
    determine_pcap_events : Finds the events.
    filter_pcap_events : Drops the rows the mask marks.

    Examples
    --------
    >>> import pandas as pd
    >>> from lair.pcaps import build_pcap_mask
    >>> events = pd.DataFrame(
    ...     {
    ...         "start": pd.to_datetime(["2024-01-01 12:00"]),
    ...         "end": pd.to_datetime(["2024-01-03 11:59:59"]),
    ...     }
    ... )
    >>> times = pd.date_range("2024-01-01 00:00", periods=6, freq="12h")
    >>> build_pcap_mask(times, events).astype(int).tolist()
    [0, 1, 1, 1, 1, 0]
    """
    # Ensure we can compare the index values to the event timestamps.
    # Use pd.to_datetime so the function is robust to Index types that
    # are datetime-like but not necessarily a DatetimeIndex object.
    # Naive times are taken as UTC (valleyheatdeficit's events are naive UTC),
    # so tz-aware times are converted to naive UTC before comparing.
    idx = _naive_utc(pd.DatetimeIndex(pd.to_datetime(index)))

    mask = pd.Series(False, index=index)

    for _, event in events.iterrows():
        start = _naive_utc(pd.Timestamp(event["start"]))
        end = _naive_utc(pd.Timestamp(event["end"]))
        # Build boolean selection using vectorized comparisons. This
        # avoids label-based slicing (mask.loc[start: end]) which can
        # raise KeyError for non-monotonic indexes or when exact labels
        # are not present in the index.
        sel = (idx >= start) & (idx <= end)
        if sel.any():
            mask.loc[sel] = True

    return mask


def filter_pcap_events(
    data: pd.DataFrame, events: pd.DataFrame, level=None
) -> pd.DataFrame:
    """
    Filter a DataFrame to exclude rows within PCAP events.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to filter.
    events : pd.DataFrame
        The DataFrame of PCAP events.
    level : str, optional
        If the DataFrame has a MultiIndex, the level to apply the mask to.

    Returns
    -------
    pd.DataFrame
        The DataFrame without the rows that fall within PCAP events.

    See Also
    --------
    determine_pcap_events : Finds the events.
    build_pcap_mask : The mask it applies.

    Examples
    --------
    >>> import pandas as pd
    >>> from lair.pcaps import filter_pcap_events
    >>> events = pd.DataFrame(
    ...     {
    ...         "start": pd.to_datetime(["2024-01-01 12:00"]),
    ...         "end": pd.to_datetime(["2024-01-03 11:59:59"]),
    ...     }
    ... )
    >>> times = pd.date_range("2024-01-01 00:00", periods=6, freq="12h")
    >>> data = pd.DataFrame({"ch4": range(6)}, index=times)
    >>> filter_pcap_events(data, events)["ch4"].tolist()
    [0, 5]
    """
    index = data.index.get_level_values(level) if level is not None else data.index
    mask = build_pcap_mask(index, events)
    return data[~mask.to_numpy(dtype=bool)].copy()
