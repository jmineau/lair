#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 13:32:57 2023

@author: James Mineau (James.Mineau@utah.edu)

Module to read and process all levels of UUCON LGR UGGA CO2 & CH4 data
"""

import datetime as dt
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from dataclasses import dataclass
from . import pipeline as pipe

import sys
sys.path.insert(1, '/uufs/chpc.utah.edu/common/home/u6036966/wkspace/scripts')
from helper.clock import UTC2MTN
from helper.clock import seasons as SEASONS

# Directories
UATAQ_DIR = '/uufs/chpc.utah.edu/common/home/u6036966/wkspace/data/uataq'
CONFIG_DIR = os.path.join(UATAQ_DIR, 'config')

# UATAQ pipeline config
site_config = pd.read_csv(os.path.join(CONFIG_DIR, 'site_config.csv'),
                          sep=', ', engine='python', index_col='stid')
data_config = pd.read_json(os.path.join(CONFIG_DIR, 'data_config.json'))

LGR_sites = ['csp', 'fru', 'hdp', 'hpl', 'roo', 'wbb']

SPECIES = ('CO2', 'CH4')

specie_latex = {'CO2': 'CO$_2$',
                'CH4': 'CH$_4$'}


def generate_baseline(data, window=dt.timedelta(hours=24), q=0.1):
    # data must have a datetime index
    baseline = (data.rolling(window=window, center=True).quantile(q)
                    .rolling(window=window, center=True).mean())

    return baseline


def generate_background(background_ID='hdp'):
    background_site = Site(background_ID)

    background = background_site.hourly
    return background


@dataclass
class Site():
    ID: str
    lvl: str = 'calibrated'
    species: tuple[str] = SPECIES
    excess_method: str = None

    instrument = 'lgr_ugga'

    afternoon = np.arange(18, 23)  # UTC

    def __post_init__(self):
        if not isinstance(self.species, tuple):
            self.species = tuple([self.species])

        self.data = self.read(self.lvl)

        self.simple = self.simplify(self.data, self.lvl)

        if self.lvl == 'calibrated':

            # Resample to hourly averages
            self.hourly = self.simple.dropna(how='all').resample('1H').mean()

            # Calculate excess if method is given
            if self.excess_method is not None:
                excess = self.calculate_excess(self.excess_method)

                self.hourly = pd.concat([self.hourly, excess], axis=1)

            # Subset to local afternoons
            self.afternoons = self.get_afternoons(self.hourly)

            # Average afternoons
            self.well_mixed = self.get_well_mixed(self.hourly)

    def read(self, lvl):
        lvl_funcs = {'raw': pipe.lgr_ugga.RAW,
                     'qaqc': pipe.lgr_ugga.QAQC,
                     'calibrated': pipe.lgr_ugga.CALIBRATED}

        data = lvl_funcs[lvl](self.ID)

        return data

    def simplify(self, data, lvl):

        if lvl == 'calibrated':
            # Apply QAQC Filter
            data = data[data.QAQC_Flag >= 0]  # Doesn't work for LICORs

        simple_cols = []
        for specie in self.species:
            data_col = f'{specie.upper()}d_ppm'
            if lvl == 'calibrated':
                data_col += '_cal'
            simple_cols.append(data_col)

        data = data[simple_cols]
        data.columns = data.columns.str.removesuffix(data_col[3:])

        return data

    def calculate_excess(self, method=excess_method,
                         window=dt.timedelta(hours=24), q=0.1,
                         background_ID='hdp'):

        total = self.hourly

        if method == 'baseline':
            baseline = generate_baseline(total, window=window, q=q)

            excess = total - baseline

        elif method == 'background':
            background = generate_background(background_ID)

            excess = total - background

        excess.columns += 'ex'

        return excess

    @staticmethod
    def get_afternoons(data, hours=afternoon):
        afternoons = data[data.index.hour.isin(hours)]

        return afternoons

    @staticmethod
    def get_well_mixed(data, hours=afternoon):
        # mean_hour = self.afternoon.mean()

        afternoons = Site.get_afternoons(data)
        well_mixed = afternoons.resample('1d').mean()
        # well_mixed.index += pd.TimedeltaIndex([mean_hour] * len(well_mixed),
        #                                       unit='h')

        return well_mixed


# %% PLOT

def plot_sites(ax, sites='active', **kwargs):
    import cartopy.crs as ccrs
    import geopandas as gpd

    coords = ['lati', 'long']

    if sites == 'all':
        df = site_config.loc[:, coords]
    elif sites == 'active':
        df = site_config.loc[site_config.active, coords]
    else:
        assert isinstance(sites, list)
        sites = list(map(str.lower, sites))
        df = site_config.loc[sites, coords]

    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.long,
                                                           df.lati))
    # gdf.set_crs(epsg=4326, inplace=True)  # changes figsize

    gdf.plot(ax=ax, transform=ccrs.PlateCarree(), c='None', edgecolor='black',
             **kwargs)

    return None


def timeseries(data, title, species=['CH4'], excess=True):

    def handle_excess(excess, specie):
        latex = specie_latex[specie]
        label = f'{latex} [ppm]'
        col = specie

        if excess:
            label = 'Excess ' + label
            col += 'ex'

        return col, label

    def plot_one(ax, specie, excess=False, title=title):
        col, label = handle_excess(excess, specie)

        data[col].plot(ax=ax, c='black')

        ax.set(title=title,
               xlabel='Time',
               ylabel=label)

    def plot_both(ax, excess=False, title=title):
        ax2 = ax.twinx()

        for specie, y_ax in zip(SPECIES, [ax, ax2]):
            col, label = handle_excess(excess, specie)
            data[col].plot(ax=y_ax, label=specie_latex[specie])

            y_ax.set(ylabel=label)

        ax.set(title=title,
               xlabel='Time')

    nrows = 2 if excess is True else 1

    fig, axes = plt.subplots(nrows=nrows, sharex=True)

    ax = axes[0] if excess is True else axes

    if len(species) == 1:
        plot_one(ax, species[0])

    elif len(species) == 2:
        plot_both(ax)

    if excess is True:
        ax = axes[1]

        if len(species) == 1:
            plot_one(ax, species[0], excess=excess, title=None)

        elif len(species) == 2:
            plot_both(ax, excess=excess, title=None)

    plt.show()



def diurnal(data_allhours):
    df = data_allhours

    df = UTC2MTN(df)
    df.index.rename('Time_MST', inplace=True)

    df['season'] = df.index.month.map(seasons)

    df_diurnal = df.groupby([df.index.hour, df.season]).mean(numeric_only=True)

    for site in sites:
        rel = sns.relplot(data=df_diurnal[site].reset_index(level=[1]),
                          col='season', x='Time_MST', y='excess', kind='line',
                          col_wrap=2)

        rel.fig.suptitle(site)

        plt.show()


def diurnal_years(data_allhours):
    df = data_allhours.copy()

    df = UTC2MST(df)
    df.index.rename('Time_MST', inplace=True)

    df['season'] = df.index.month.map(SEASONS)
    df['year'] = df.index.year
    df['hour'] = df.index.hour

    # df_diurnal = df.groupby([df.index.hour, df.year, df.season]).mean(numeric_only=True)
    # counts = df.groupby([df.index.hour, df.year, df.season]).count()

    # data = df_diurnal.reset_index(level=[1, 2])

    rel = sns.relplot(data=df, col='season', x='hour', y='CH4_ex',
                      hue='year', kind='line', col_wrap=2, errorbar='se',
                      palette=sns.color_palette("hls", 5))

    rel.set(xlabel='Hour [MST]',
            ylabel='CH4$_{4ex}$ [ppm]')

    plt.show()

    return df


# def seasonal_trends(data, plot_col, figsize=None):
#     from scipy.stats import linregress

#     df = data.resample('QS-DEC').mean(numeric_only=True)
#     df['season'] = df.index.month.map(SEASONS)
#     df = df.groupby(['season', df.index.year]).mean()

#     colors = ['#e7298a', '#d95f02', '#1b9e77', '#7570b3']

#     fig, ax = plt.subplots(figsize=figsize)

#     for season, color in zip(['DJF', 'MAM', 'JJA', 'SON'], colors):
#         season_df = df.loc[season].dropna(subset=plot_col)
#         season_df.reset_index(inplace=True)  # remove year from index
#         season_df.rename(columns={'Time_UTC': 'Year'}, inplace=True)

#         x = season_df.Year.astype(int)
#         y = season_df[plot_col]

#         slope, intercept, r_value, p_value, std_err = linregress(x, y)

#         season_df.plot(x='Year', y=plot_col, ax=ax, marker='s', c=color,
#                        label=f'{season}: Slope={slope:.3}; '
#                        f'P-value={p_value:.3}')

#         ax.plot(x, intercept + slope * x, color, linestyle='--', lw=4)

#     return ax

# %%
def seasonal_trends(data, plot_col, figsize=None, table_loc='upper right',
                    table_colWidth=0.15):
    from scipy.stats import linregress

    df = data.resample('QS-DEC').mean(numeric_only=True)
    df['season'] = df.index.month.map(SEASONS)
    df = df.groupby(['season', df.index.year]).mean()

    seasons = ['DJF', 'MAM', 'JJA', 'SON']
    colors = ['#e7298a', '#1b9e77', '#d95f02', '#7570b3']

    fig, ax = plt.subplots(figsize=figsize)

    cell_text = []
    for season, color in zip(seasons, colors):
        season_df = df.loc[season].dropna(subset=plot_col)
        season_df.reset_index(inplace=True)  # remove year from index
        season_df.rename(columns={'Time_UTC': 'Year'}, inplace=True)

        x = season_df.Year.astype(int)
        y = season_df[plot_col]

        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        season_df.plot(x='Year', y=plot_col, ax=ax, marker='s', c=color,
                       legend=False, zorder=0)

        ax.plot(x, intercept + slope * x, color, linestyle='--', lw=4,
                zorder=1)

        cell_text.append([round(slope, 3), round(p_value, 4)])

    table = plt.table(np.array(cell_text).T, rowLabels=['Slope', 'P-Value'],
                      colLabels=seasons, colColours=colors, loc=table_loc,
                      colWidths=[table_colWidth]*4, cellLoc='center')

    return ax, table
