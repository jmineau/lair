"""
Functions for calculating the valley heat deficit (VHD) and determining persistent cold air pool (PCAP) events.
"""


import pandas as pd
import xarray as xr

from lair import units
from lair.constants import Rd, cp
from lair.meteorology import ideal_gas_law, hypsometric, poisson


def valleyheatdeficit(data: xr.Dataset, integration_height=2200) -> xr.DataArray:
    """
    Calculate the valley heat deficit.

    Whiteman, C. David, et al. “Relationship between Particulate Air Pollution
    and Meteorological Variables in Utah's Salt Lake Valley.”
    Atmospheric Environment, vol. 94, Sept. 2014, pp. 742-53.
    DOI.org (Crossref), https://doi.org/10.1016/j.atmosenv.2014.06.012.

    Parameters
    ----------
    data : xr.DataSet
        The sounding data.
    integration_height : int
        The height to integrate to [m].
    
    Returns
    -------
    xr.DataArray
        The valley heat deficit [MJ/m^2].
    """
    h0 = data.elevation

    # Subset to the heights between the surface and the integration height
    data = data.sel(height=slice(h0, integration_height))
    
    T = data.temperature.pint.quantify('degC').pint.to('degK')
    p = data.pressure.pint.quantify('hPa').pint.to('Pa')

    # Calculate potential temperature using poisson's equation
    theta = poisson(T=T, p=p, p0=1e5 * units('Pa'))
    theta_h = theta.sel(height=integration_height, method='nearest')

    # Calculate virtual temperature to account for water vapor
    Tv = hypsometric(p1=p.isel(height=slice(0, -1)).values,
                     p2=p.isel(height=slice(1, None)).values,
                     deltaz=data.interpolation_interval * units('m'))
    layer_heights = T.height.values[:-1] + data.interpolation_interval / 2
    Tv = xr.DataArray(Tv, coords=[data.time, layer_heights],
                      dims=['time', 'height'])\
            .interp_like(T, method='linear')\
            .pint.quantify('degK')

    # Calculate the density using the ideal gas law
    rho = ideal_gas_law(solve_for='rho', p=p, T=Tv, R=Rd)
    # Set pint units - Setting units to kg/m3 doesnt change the numbers
    # pint-xarray hasnt implemented .to_base_units() yet
    # when they do, we can change this to .pint.to_base_units()
    rho = rho.pint.to('kg/m^3')

    # Calculate the heat deficit by integrating using the trapezoid method
    heat_deficit = (cp * rho * (theta_h - theta)).dropna('height', how='all')\
        .integrate('height') * (1 * units('m'))  # J/m2

    heat_deficit = heat_deficit.pint.to('MJ/m^2').pint.dequantify()

    vhd = heat_deficit.to_series()
    vhd.name = 'VHD_MJ_m2'
    vhd.index.name = 'Time_UTC'
    return vhd


def determine_pcap_events(vhd: pd.Series, threshold: float, min_periods: int = 3
                          ) -> pd.DataFrame:
    """
    Determine the periods of persistent cold air pool (PCAP) events
    using the valley heat deficit (VHD) metric.
    """
    # Create a boolean mask for values above the threshold
    above_thres = vhd > threshold
    
    # Group consecutive values above the threshold and count the number of periods in each group
    groups = (above_thres != above_thres.shift()).cumsum()
    
    # Create a mask for groups that have at least min_periods consecutive values above the threshold
    event_mask = above_thres.groupby(groups).transform('sum') >= min_periods
    
    # Subset the original series to include only the values that are part of a PCAP event
    pcap_vhd = vhd[above_thres & event_mask]
    
    # Add event id to the series as a new column in a DataFrame
    pcap_vhd = pcap_vhd.to_frame()
    pcap_vhd['event_id'] = groups.loc[pcap_vhd.index].values
    
    # Group by event_id and determine the start and end time of each event
    pcap_events = (
    pcap_vhd.groupby('event_id')
    .apply(lambda g: pd.Series({
        'start': g.index.min(),
        'end': g.index.max() + pd.Timedelta(hours=11, minutes=59, seconds=59)  # inclusive of following 12 hours
    }))
    .reset_index(drop=True)
)
    pcap_events.index.name = 'event_id'

    return pcap_events