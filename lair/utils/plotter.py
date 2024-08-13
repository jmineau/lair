"""
This module provides utility functions for plotting data.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from cartopy.io.img_tiles import GoogleWTS
from matplotlib.legend_handler import HandlerLineCollection


def log10formatter(x, pos, deci=0) -> str:
    """
    Format ticks to log 10 format with deci number of decimals.

    Parameters
    ----------
    x : float
        Tick value.
    deci : int, optional
        Number of decimals to display. Defaults to 0.

    Returns
    -------
    str
        Formatted tick label.

    Examples
    --------
    >>> from functools import partial
    >>> import numpy as np
    >>> data: xr.DataArray  # some data, in this case, 3D (time, lat, lon)
    >>> np.log10(data).plot(cbar_kwargs={'format': partial(log10formatter, deci=2)})
    """

    return f'$10^{{{x:.{deci}f}}}$'


def truncate_colormap(cmap: str | mcolors.Colormap, minval: float=0.0, maxval: float=1.0, n: int=100) -> mcolors.LinearSegmentedColormap:
    """
    Truncate matplotlib colormaps using min and max vals from 0 to 1,
    and then linearly build a new colormap

    Parameters
    ----------
    cmap : str | matplotlib.colors.Colormap
        Colormap to be truncated.
    minval : float, optional
        Minimum value to truncate the colormap. Defaults to 0.0.
    maxval : float, optional
        Maximum value to truncate the colormap. Defaults to 1.0.
    n : int, optional
        Number of colors in the new colormap. Defaults to 100.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Truncated colormap.
    """
    if isinstance(cmap, str):
        cmap: mcolors.Colormap = plt.get_cmap(cmap)

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap


def NCL_cmap(table_name: str) -> mcolors.LinearSegmentedColormap:
    """
    Generate matplotlib colormap from NCL color table.

    Parameters
    ----------
    table_name : str

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        matplotlib colormap from NCL color table.
    """
    import pandas as pd

    # Get NCL table link
    tables = 'https://www.ncl.ucar.edu/Document/Graphics/ColorTables/Files'
    table_link = f'{tables}/{table_name}.rgb'

    # Convert table rgb file to pandas dataframe
    # TODO might be a better way to do this
    colortab = pd.read_csv(table_link, delim_whitespace=True, skiprows=1)
    colortab = colortab.drop('b', axis=1)  # Fix columns
    colortab = colortab.rename({'#': 'r', 'r': 'g', 'g': 'b'}, axis=1)

    # Create linear matplotlib cmap from pandas dataframe
    cmap = mcolors.LinearSegmentedColormap.from_list(table_name, colortab.values/255, N=100)
    return cmap


def terrain_cmap(minval: float=0.42, maxval: float=1.0, n: int=256) -> mcolors.LinearSegmentedColormap:
    """
    Matplotlib terrain cmap.

    Parameters
    ----------
    minval : float, optional
        Minimum value to truncate the colormap. Defaults to 0.42.
    maxval : float, optional
        Maximum value to truncate the colormap. Defaults to 1.0.
    n : int, optional
        Number of colors in the new colormap. Defaults to 256.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        Matplotlib terrain colormap.
    """
    return truncate_colormap('terrain', minval=minval, maxval=maxval, n=n)


def diurnalPlot(data: pd.DataFrame, param: str, stats: str | list[str]=['std', 'median', 'mean'],
                units: str | None=None, tz: str='UTC', freq: str='1H', ax: plt.Axes | None=None,
                colors: str | dict[str, str]={'mean': 'black', 'median': 'blue', 'std': 'gray'},
                min_count: int = 0) -> plt.Axes:
    """
    Plot the diurnal cycle of data.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot.
    param : str
        Parameter to plot.
    stats : str | list[str], optional
        Statistics to plot. Defaults to ['std', 'median', 'mean'].
    units : str | None, optional
        Units of the parameter for the ylabel. Defaults to None.
    tz : str, optional
        Timezone of the data. Defaults to 'UTC'.

        .. warning::
            DOES NOT CONVERT TIMEZONES.

    freq : str, optional
        Frequency of the data. Defaults to '1H'.
    ax : plt.Axes | None, optional
        Axis to plot on. Defaults to None.
    colors : str | dict[str, str], optional
        Colors of the statistics. Defaults to {'mean': 'black', 'median': 'blue', 'std': 'gray'}.
    min_count : int, optional
        Minimum count to plot. Defaults to 0.

    Returns
    -------
    plt.Axes
        Axis with the plot
    """
    import datetime as dt
    import matplotlib.dates as mdates
    from lair.utils.clock import diurnal

    # Check for count in stats
    if 'count' in stats:
        plot_count = True
    else:
        plot_count = False
        stats.append('count')

    # Calculate diurnal cycle
    agg = diurnal(data, freq, stats)[param]

    # Assign dummy date so locator isn't confused
    agg.index = agg.index.map(lambda t:
                              dt.datetime.combine(dt.date.today(), t))

    # Filter by count
    if min_count > 0:
        agg.loc[agg['count'] < min_count] = np.nan

    # Plot data
    if ax is None:
        fig, ax = plt.subplots()

    legend_elements = {
    }

    for stat in stats:
        if stat == 'std' and 'mean' in stats:
            # Plot standard deviation as fill_between only if 'mean' is also plotted
            handle = ax.fill_between(agg.index,
                                  agg['mean'] - agg['std'],
                                  agg['mean'] + agg['std'],
                                  color=colors.get(stat, 'gray'), alpha=0.2, edgecolor='none')
        else:
            if stat == 'count' and not plot_count:
                continue
            # Plot other statistics as lines
            handle, = ax.plot(agg.index, agg[stat], c=colors.get(stat, 'black'), lw=3)
        legend_elements[stat] = handle

    # Format x axis
    locator = mdates.HourLocator(byhour=range(3, 24, 3))
    formatter = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlim(mdates.date2num(agg.index[0])-0.03, mdates.date2num(agg.index[-1])+0.03)

    # Build legend
    if 'std' in stats and 'mean' in stats:
        std = legend_elements.pop('std')
        mean = legend_elements.pop('mean')
        legend_elements[r'mean $\pm$1$\sigma$'] = (std, mean)
    handles = [value for value in legend_elements.values()]
    labels = [key for key in legend_elements.keys()]
    ax.legend(handles, labels)

    ylabel = f'{param}'
    if units is not None:
        ylabel += f' [{units}]'
    ax.set(xlabel=f'Time [{tz}]',
           ylabel=ylabel)

    return ax


def seasonalPlot(data: pd.DataFrame, param: str='CH4', units: str='ppm', ax: plt.Axes | None=None
                 ) -> plt.Axes:
    """
    Plot the seasonal cycle of data by year.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot.
    param : str, optional
        Parameter to plot. Defaults to 'CH4'.
    units : str, optional
        Units of the parameter for the ylabel. Defaults to 'ppm'.
    ax : plt.Axes | None, optional
        Axis to plot on. Defaults to None.

    Returns
    -------
    plt.Axes
        Axis with the plot
    """
    # TODO need to add a year or int x formatter
    colors = {'DJF': '#e7298a', 
              'MAM': '#1b9e77', 
              'JJA': '#d95f02', 
              'SON': '#7570b3'}
    from lair.utils.clock import seasonal
    
    # Calculate seasonal cycle
    agg = seasonal(data, ['mean', 'std'])[param].unstack(level=0)

    # Plot data
    if ax is None:
        fig, ax = plt.subplots()

    mean = agg['mean'].plot(ax=ax, style=colors, lw=4)
    
    for season in agg.columns.levels[1]:
        ax.fill_between(agg.index, agg['mean', season] - agg['std', season],
                        agg['mean', season] + agg['std', season],
                        color=colors[season], alpha=0.2, edgecolor='none')

    ax.set(xlabel='Year',
           ylabel=f'{param} [{units}]')

    return ax


def create_polar_ax() -> plt.Axes:
    """
    Create a polar axis with North at the top.

    Returns
    -------
    plt.Axes
        Polar axis.
    """
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['N', 'E', 'S', 'W'])

    return ax


def format_radial_axis(ax: plt.Axes, x: str, scale_angle: float) -> None:
    """
    Format radial axis of polar plot.

    Parameters
    ----------
    ax : plt.Axes
        Axis to format.
    x : str
        Label of the radial axis.
    scale_angle : float
        Angle to position the label.

    Returns
    -------
    None
    """
    if scale_angle is not None:
        ax.set_rlabel_position(scale_angle)
    ha = 'right' if ax.get_rlabel_position() > 180 else 'left'
    ax.text(np.deg2rad(ax.get_rlabel_position()), max(ax.get_yticks()),
            x, ha=ha)
    # TODO
        # auto.text
        # when xbins is a list, x label is not positioned well
    return None


def polarPlot(data: pd.DataFrame, param: str='CH4', x: str='ws', wd: str='wd',
              statistic: str='mean', units: str='ppm', min_bin: int=1, xbins: int=30,
              scale_angle: float | None=None) -> plt.Axes:
    """
    Plot polar contour of data.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot.
    param : str, optional
        Parameter to plot. Defaults to 'CH4'.
    x : str, optional
        Variable to bin in the radial axis. Defaults to 'ws'.
    wd : str, optional
        Variable to bin in the angular axis. Defaults to 'wd'.
    statistic : str, optional
        Statistic to plot. Defaults to 'mean'.
    units : str, optional
        Units of the parameter for the colorbar label. Defaults to 'ppm'.
    min_bin : int, optional
        Minimum count in each bin to plot. Defaults to 1.
    xbins : int, optional
        Number of bins in the radial axis. Defaults to 30.
    scale_angle : float | None, optional
        Angle to position the radial axis label. Defaults to None.

    Returns
    -------
    plt.Axes
        Axis with the plot
    """
    from lair.air.air import bin_polar, circularize_radial_data

    binned_data = bin_polar(data, wd, x, xbins)

    agg = binned_data[[param, 'radian_bin', 'x_bin']]\
        .groupby(['radian_bin', 'x_bin']).agg([statistic, 'count'])[param]\
        .unstack()

    # Filter by count in each bin
    bins_n = agg['count']
    agg = agg[statistic].where(bins_n > min_bin)

    theta, r, c = circularize_radial_data(agg)

    ax = create_polar_ax()

    p = ax.contourf(theta, r, c, cmap='YlOrRd')
    cb = plt.colorbar(p, pad=0.07,
                 label=f'{statistic.capitalize()} {param} [{units}]')
    ax.colorbar = cb

    format_radial_axis(ax, x, scale_angle)

    return ax


def polarFreq(data: pd.DataFrame, x: str='ws', wd: str='wd', xbins: int=30,
              scale_angle: float | None =None) -> plt.Axes:
    """
    Plot the contoured frequency of the data.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot.
    x : str, optional
        Variable to bin in the radial axis. Defaults to 'ws'.
    wd : str, optional
        Variable to bin in the angular axis. Defaults to 'wd'.
    xbins : int, optional
        Number of bins in the radial axis. Defaults to 30.
    scale_angle : float | None, optional
        Angle to position the radial axis label. Defaults to None.

    Returns
    -------
    plt.Axes
        Axis with the plot
    """

    binned_data = bin_polar(data, wd, x, xbins)

    binned_data['count'] = 1
    counts = binned_data[['count', 'radian_bin', 'x_bin']]\
        .groupby(['radian_bin', 'x_bin'])['count'].sum()\
        .unstack()

    theta, r, c = circularize_contour_data(counts)

    # Convert to percent and set 0 to nan
    c = c / c.sum() * 100
    c[c == 0] = np.nan

    ax = create_polar_ax()

    p = ax.contourf(theta, r, c, cmap='binary')
    plt.colorbar(p, pad=0.07,
                 label='Frequency [%]')

    format_radial_axis(ax, x, scale_angle)

    return ax


def windvectorPlot(data: pd.DataFrame, wd: str='WD', ws: str='WS',
                   ax: plt.Axes | None =None, unit_length: bool=False,
                   **kwargs) -> plt.Axes:
    """
    Plot wind vectors.

    Parameters
    ----------
    data : pd.DataFrame
        Data to plot.
    wd : str, optional
        Wind direction variable. Defaults to 'WD'.
    ws : str, optional
        Wind speed variable. Defaults to 'WS'.
    ax : plt.Axes | None, optional
        Axis to plot on. Defaults to None.
    unit_length : bool, optional
        Normalize vectors to unit length. Defaults to False.
    **kwargs
        Additional arguments to pass to plt.quiver.

    Returns
    -------
    plt.Axes
        Axis with the plot
    """
    from lair.air.air import wind_components
    if ax is None:
        fig, ax = plt.subplots()

    u, v = wind_components(data[ws], data[wd])
    
    if unit_length:
        # Normalize vectors to unit length
        u = u / data[ws]
        v = v / data[ws]

    # Plot wind vectors
    ax.quiver(data.index, data[ws], u, v, **kwargs)
    ax.get_figure().autofmt_xdate()

    return ax


class HandlerDashedLines(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.

    https://stackoverflow.com/questions/31544489/two-line-styles-in-legend

    There's a potentially better version in the above link

    This needs a new names and better documentation.
    """

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        import numpy as np
        from matplotlib.lines import Line2D

        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[1] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines
