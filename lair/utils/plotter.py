"""
lair.utils.plotter
~~~~~~~~~~~~~~~~~~

This module provides utility functions for plotting data.
"""

import matplotlib.colors as mcolors
import numpy as np
from cartopy.io.img_tiles import GoogleWTS
from matplotlib.legend_handler import HandlerLineCollection


def log10formatter(x, pos, deci=0):
    """
    Format ticks to log 10 format with deci number of decimals
    """

    return f'$10^{{{x:.{deci}f}}}$'


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """
    Truncate matplotlib colormaps using min and max vals from 0 to 1,
    and then linearly building a new colormap
    """

    new_cmap = mcolors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))

    return new_cmap


def NCL_cmap(table_name):
    '''Generate matplotlib colormap from NCL color table'''
    import pandas as pd
    from matplotlib.colors import LinearSegmentedColormap as LSC

    # Get NCL table link
    tables = 'https://www.ncl.ucar.edu/Document/Graphics/ColorTables/Files'
    table_link = f'{tables}/{table_name}.rgb'

    # Convert table rgb file to pandas dataframe
    # TODO might be a better way to do this
    colortab = pd.read_csv(table_link, delim_whitespace=True, skiprows=1)
    colortab = colortab.drop('b', axis=1)  # Fix columns
    colortab = colortab.rename({'#': 'r', 'r': 'g', 'g': 'b'}, axis=1)

    # Create linear matplotlib cmap from pandas dataframe
    cmap = LSC.from_list(table_name, colortab.values/255, N=100)
    return cmap


def get_terrain_cmap(minval=0.42, maxval=1.0, n=256):
    '''Matplotlib terrain map truncated using min and max values between
    0 and 1'''

    import matplotlib.pyplot as plt
    terrain = truncate_colormap(plt.get_cmap('terrain'),
                                minval=minval, maxval=maxval, n=n)
    return terrain

def diurnalPlot(data, param='CH4', units='ppm', tz='MST', freq='1H', ax=None):
    # Calculate diurnal cycle
    agg = diurnal(data, ['mean', 'median', 'std'], freq)[param]

    # Assign dummy date so locator isn't confused
    agg.index = agg.index.map(lambda t:
                              dt.datetime.combine(dt.date.today(), t))

    # Plot data
    if ax is None:
        fig, ax = plt.subplots()

    mean, = ax.plot(agg.index, agg['mean'], c='black')
    std = ax.fill_between(agg.index, agg['mean'] - agg['std'],
                          agg['mean'] + agg['std'],
                          color='gray', alpha=0.2, edgecolor='none')
    median, = ax.plot(agg.index, agg['median'], c='blue', lw=4)

    # Format x axis
    locator = mdates.HourLocator(byhour=range(3, 24, 3))
    formatter = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlim(mdates.date2num(agg.index[0])-0.03, mdates.date2num(agg.index[-1])+0.03)

    ax.legend([(std, mean), median], [r'mean $\pm$1$\sigma$', 'median'])
    ax.set(xlabel=f'Time [{tz}]',
           ylabel=f'{param} [{units}]')

    return ax


def seasonalPlot(data, param='CH4', units='ppm', ax=None):
    colors = {'DJF': '#e7298a', 
              'MAM': '#1b9e77', 
              'JJA': '#d95f02', 
              'SON': '#7570b3'}
    
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


def create_polar_ax():
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    plt.xticks([0, np.pi/2, np.pi, 3*np.pi/2], ['N', 'E', 'S', 'W'])

    return ax


def format_radial_axis(ax, x, scale_angle):
    if scale_angle is not None:
        ax.set_rlabel_position(scale_angle)
    ha = 'right' if ax.get_rlabel_position() > 180 else 'left'
    ax.text(np.deg2rad(ax.get_rlabel_position()), max(ax.get_yticks()),
            x, ha=ha)
    # TODO
        # auto.text
        # when xbins is a list, x label is not positioned well
    return None


def polarPlot(data, param='CH4', x='ws', wd='wd', statistic='mean',
              units='ppm', min_bin=1, xbins=30, scale_angle=None):

    binned_data = bin_polar(data, wd, x, xbins)

    agg = binned_data[[param, 'radian_bin', 'x_bin']]\
        .groupby(['radian_bin', 'x_bin']).agg([statistic, 'count'])[param]\
        .unstack()

    # Filter by count in each bin
    bins_n = agg['count']
    agg = agg[statistic].where(bins_n > min_bin)

    theta, r, c = circularize_contour_data(agg)

    ax = create_polar_ax()

    p = ax.contourf(theta, r, c, cmap='YlOrRd')
    cb = plt.colorbar(p, pad=0.07,
                 label=f'{statistic.capitalize()} {param} [{units}]')
    ax.colorbar = cb

    format_radial_axis(ax, x, scale_angle)

    return ax


def polarFreq(data, x='ws', wd='wd', xbins=30, scale_angle=None):

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


def windvectorPlot(data, wd='WD', ws='WS', ax=None, unit_length=False, **kwargs):
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


class StadiaMapsTiles(GoogleWTS):
    """
    Retrieves tiles from stadiamaps.com.

    For a full reference on the styles available please see
    https://docs.stadiamaps.com/themes/. A few of the specific styles
    that are made available are ``alidade_smooth``, ``stamen_terrain`` and
    ``osm_bright``.

    Using the Stadia Maps API requires including an attribution. Please see
    https://docs.stadiamaps.com/attribution/ for details.

    For most styles that means including the following attribution:

    `© Stadia Maps <https://www.stadiamaps.com/>`_
    `© OpenMapTiles <https://openmaptiles.org/>`_
    `© OpenStreetMap contributors <https://www.openstreetmap.org/about/>`_

    with Stamen styles *additionally* requiring the following attribution:

    `© Stamen Design <https://stamen.com/>`_

    Parameters
    ----------
    apikey : str, required
        The authentication key provided by Stadia Maps to query their APIs
    style : str, optional
        Name of the desired style. Defaults to ``alidade_smooth``.
        See https://docs.stadiamaps.com/themes/ for a full list of styles.
    resolution : str, optional
        Resolution of the images to return. Defaults to an empty string,
        standard resolution (256x256). You can also specify "@2x" for high
        resolution (512x512) tiles.
    cache : bool or str, optional
        If True, the default cache directory is used. If False, no cache is
        used. If a string, the string is used as the path to the cache.
    """
    
    # COPIED FROM CARTOPY GITHUB ON 2023-11-03
    # HOPEFULLY THIS WILL BE IN THE NEXT RELEASE

    def __init__(self,
                 apikey,
                 style="alidade_smooth",
                 resolution="",
                 cache=False):
        super().__init__(cache=cache, desired_tile_form="RGBA")
        self.apikey = apikey
        self.style = style
        self.resolution = resolution
        if style == "stamen_watercolor":
            # Known style that has the jpg extension
            self.extension = "jpg"
        else:
            self.extension = "png"

    def _image_url(self, tile):
        x, y, z = tile
        return ("http://tiles.stadiamaps.com/tiles/"
                f"{self.style}/{z}/{x}/{y}{self.resolution}.{self.extension}"
                f"?api_key={self.apikey}")