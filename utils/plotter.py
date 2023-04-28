#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 14:17:19 2023

@author: James Mineau (James.Mineau@utah.edu)

Module to help generate plots
"""
from matplotlib.legend_handler import HandlerLineCollection


def log10formatter(x, pos, deci=0):
    '''Format ticks to log 10 format with deci number of decimals'''

    return f'$10^{{{x:.{deci}f}}}$'


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    '''Truncate matplotlib colormaps using min and max vals from 0 to 1,
    and then linearly building a new colormap'''

    import matplotlib.colors as mcolors
    import numpy as np

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
