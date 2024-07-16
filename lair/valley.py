"""
Module for creating a map of the Salt Lake Valley.

.. warning::
    This module was more or less useable until I moved it to the lair package, now it is not.
    I will need to update the imports and paths to get it working again.
"""

import cartopy.crs as ccrs
import geopandas as gpd
from matplotlib import pyplot as plt
import os
import pandas as pd
from shapely.geometry import Polygon


# TODO
#   Need to keep data in one location
#       Can I store with package or will it just need to sit on a lin-group dir
#       Data:
#           TRAX, MesoWest, UUCON, borders, interstates
#       Maybe can download them?

#   Labels probably too hard, just do them in powerpoint after
#   Try to put colorbar over ochres for inventory or census
wkspace = '/uufs/chpc.utah.edu/common/home/u6036966/wkspace'
DATA_DIR = os.path.join(wkspace, 'data')

# TODO
#   create tilers to select from

#: Salt Lake Valley bounding box [lonmin, latmin, lonmax, latmax]
BOUNDS = (-112.25, 40.4, 
          -111.62, 40.95)


class SaltLake:
    """
    Salt Lake Valley map class.

    Methods
    -------
    add_tiler(tiler, tiler_zoom)
        Add a tiler background to the map.
    add_Inventory(Inventory, alpha)
        Add an inventory to the map.
    add_census(census, alpha)
        Add census data to the map.
    add_border(lvl)
        Add state or county borders to the map.
    add_interstates()
        Add interstate highways to the map.
    add_TRAX(lines)
        Add TRAX lines to the map.
    add_UUCON(sites)
        Add UUCON sites to the map.
    add_MesoWest(status, networks)
        Add MesoWest stations to the map.
    add_north_arrow()
        Add a north arrow to the map.
    add_legend(legend_kws, legend_mapper)
        Add a legend to the map.
    add_extent_map()
        Add an extent map to the map.
    """
    def __init__(self, bounds=BOUNDS,
                 ax=None, crs=None, figsize=(6, 6),
                 tiler=None, tiler_zoom=9,
                 Inventory=False,  Inventory_cmap=None,
                 census=False,  # options for pop density, income, etc
                 background_alpha=0.3,
                 state_borders=False, county_borders=False,
                 interstates=False,
                 TRAX=False, UUCON=False,
                 MesoWest=False, Meso_status='active', Meso_networks=['UUNET'],
                 radiosonde=False, helicopter=False, DAQ=False,
                 scale_bar=False, north_arrow=False,
                 legend=True, legend_kws=None, legend_mapper=None,  # TODO
                 extent_map=False,  # TODO
                 latlon_ticks=True, more_lon_ticks=1, more_lat_ticks=0):

        self.features = []

        self._map_background(bounds, ax, crs, figsize,
                             tiler, tiler_zoom,
                             Inventory, census, background_alpha,
                             state_borders, county_borders,
                             latlon_ticks, more_lon_ticks, more_lat_ticks)

        if interstates:
            self.add_interstates()

        if TRAX:
            if TRAX is True:
                TRAX = ['r', 'g']
            self.TRAX_lines = TRAX  # save attr for legend
            self.add_TRAX(TRAX)

        if UUCON:
            if UUCON is True:
                UUCON = 'active'
            self.add_UUCON(UUCON)

        if MesoWest:
            self.add_MesoWest(Meso_status, Meso_networks)

        if radiosonde:
            self.add_radiosonde()

        if helicopter:
            self.add_helicopter()

        if DAQ:
            self.add_DAQ()

        if scale_bar:
            self.add_scale_bar()

        if north_arrow:
            self.add_north_arrow()

        if legend:
            self.add_legend(legend_kws, legend_mapper)

        if extent_map:
            self.add_extent_map()

    def __repr__(self):  # TODO

        # tiler used, bounds, inventory or census
        background = {'crs', self.crs}

        return f'SaltLakeValley(features={self.features})'

    def collect_feature(add_func):
        def wrapper(self, *args, **kwargs):
            feature = add_func(self, *args, **kwargs)
            self.features.append(feature)

        return wrapper

    def _map_background(self, bounds, ax, crs, figsize,
                        tiler, tiler_zoom,
                        Inventory, census, background_alpha,
                        state_borders, county_borders,
                        latlon_ticks, more_lon_ticks, more_lat_ticks):
        from utils.grid import bbox2extent, add_latlon_ticks

        # Matplotlib axes
        if not ax:  # if an ax is not given

            if bool(crs) & bool(tiler):
                # Need to use the tiler's crs
                raise ValueError("crs & tiler cannot both be supplied!")

            elif not bool(crs):  # if crs is not given
                if tiler:  # use tiler crs if tiler is given
                    crs = tiler.crs

                else:  # otherwise use PlateCaree (lat/lon)
                    crs = ccrs.PlateCarree()

            fig, ax = plt.subplots(subplot_kw={'projection': crs},
                                   figsize=figsize)

        elif bool(ax) & bool(crs):
            # ax has already been createdd, too late for a crs
            raise ValueError('ax & crs cannot both be supplied!')

        # Set attributes here so adding methods can use them
        self.ax = ax
        self.bounds = bounds
        self.crs = crs

        self.extent = bbox2extent(bounds)
        ax.set_extent(self.extent)

        # Cartopy tiler
        if tiler:
            self.add_tiler(tiler, tiler_zoom)

        # Xarray raster basemap
        if bool(Inventory) & bool(census):
            raise ValueError("Inventory & census cannot both be supplied!")

        elif Inventory:
            self.add_Inventory(Inventory, background_alpha)
            self.Inventory = Inventory

        elif census:
            self.add_census(census, background_alpha)

        # Geopandas vector boundaries
        if state_borders:
            self.add_border('state')

        if county_borders:
            self.add_border('county')

        # Format lat and lon ticks
        if latlon_ticks:
            add_latlon_ticks(ax, self.extent, x_rotation=30,
                             more_lon_ticks=more_lon_ticks,
                             more_lat_ticks=more_lat_ticks)
            ax.tick_params(axis='both', which='major', labelsize=15)

        return None

    @collect_feature
    def add_tiler(self, tiler, tiler_zoom):
        style_template = '{source}:{style}'

        def get_tiler(style):
            # TODO
            pass

        if isinstance(tiler, str):
            style = tiler
            tiler = get_tiler(style)
        else:
            source = tiler.__class__.__name__
            style = style_template.format(source=source,
                                          style=tiler.style)

        self.ax.add_image(tiler, tiler_zoom, zorder=0)

        return style

    @collect_feature
    def add_Inventory(self, Inventory, alpha):
        # TODO
        #   Probably should add plot method to Inventory super class
        Inventory.add2map(ax=self.ax, alpha=alpha, zorder=1)

        return str(Inventory)

    @collect_feature
    def add_census(self, census, alpha):
        import matplotlib.patches as patches
        import numpy as np

        def thous_formatter(x, pos):
            if x == 0:
                return str(x)
            return f'{x}k'

        if census == 'population':
            # http://doi.org/10.18128/D050.V17.0

            file = os.path.join(DATA_DIR,
                                'census/block_groups/utah_2020_pop.geojson')
            gdf = gpd.read_file(file)

            crs = ccrs.AlbersEqualArea()
            gdf.to_crs(crs, inplace=True)

            gdf['pop_km2_thous'] = gdf.pop_km2 / 1000
            gdf[gdf.pop_km2_thous < 0.1] = np.nan

            ax_pos = self.ax.get_position()

            cax = self.ax.get_figure().add_axes([ax_pos.x0 + 0.015,
                                                 ax_pos.y0 + 0.016,
                                                 0.03, 0.3], zorder=1.1)

            cax_frame = patches.Rectangle((ax_pos.x0 + 0.005,
                                           ax_pos.y0 + 0.003), 0.13, 0.33,
                                          edgecolor='black', facecolor='white',
                                          zorder=1.05, alpha=1,
                                          transform=self.ax.figure.transFigure)
            self.ax.add_patch(cax_frame)

            gdf.plot(ax=self.ax, column='pop_km2_thous', transform=crs,
                     cmap='pink_r', alpha=alpha, vmin=0, vmax=4, zorder=1,
                     lw=0.5, edgecolor='None', legend=True, cax=cax,
                     legend_kwds={'label': 'Population km$^{-2}$',
                                  'ticks': [0, 1, 2, 3, 4],
                                  'format': thous_formatter})

        else:
            raise NotImplementedError(f'Census data {census} not implemented!')

        return census

    @collect_feature
    def add_border(self, lvl):
        # TODO state borders should include all states
        #   possible option to subset
        borders_dir = os.path.join(DATA_DIR, 'Utah_boundaries')
        lvl_file = {'state': os.path.join(borders_dir, 'Utah.shp'),
                    'county': os.path.join(borders_dir, 'Counties.shp')}

        border = gpd.read_file(lvl_file[lvl], bbox=self.bounds)
        # border.set_crs(epsg=4326, inplace=True)

        border.plot(ax=self.ax, transform=ccrs.PlateCarree(),
                    facecolor='none', edgecolor='black', zorder=2)

        return f'{lvl} borders'

    @collect_feature
    def add_interstates(self):
        # FIXME interstates not plotting
        file = os.path.join(DATA_DIR, 'Utah_boundaries', 'Roads.shp')

        roads = gpd.read_file(file, bbox=self.bounds)

        # Filter out non-major roads
        major_roads = [1, 2, 3, 4, 5, 6]
        roads = roads[roads.CARTOCODE.isin(major_roads)]

        roads.plot(ax=self.ax)
        return 'interstates'

    @collect_feature
    def add_TRAX(self, lines):
        from utils.records import read_kml
        tracks_dir = os.path.join(wkspace, 'mobile/trax/transect/tracks')

        line_dict = {'r': {'lw': 4,
                           'z': 4.9},
                     'g': {'lw': 6,
                           'z': 4.6},
                     'b': {'lw': 7,
                           'z': 4.3}}

        for line in lines:
            file = f'TRAX_{line}_mask.kml'

            track = read_kml(os.path.join(tracks_dir, file))
            track_poly = Polygon(track._features[0]._features[0].geometry)

            self.ax.plot(*track_poly.exterior.xy, c=line,
                         transform=ccrs.PlateCarree(),
                         lw=line_dict[line]['lw'], zorder=line_dict[line]['z'])

        # from uataq.mobile import trax
        # TODO add plot method to trax class

        return 'TRAX'

    @collect_feature
    def add_UUCON(self, sites):
        from uataq import uucon

        uucon.plot_sites(self.ax, sites, zorder=5, markersize=200, lw=3)

        return 'UUCON'

    @collect_feature
    def add_MesoWest(self, status='active', networks=['UUNET']):
        # TODO create MesoWest module

        file = os.path.join(DATA_DIR, 'MesoWest',
                            'MesoWest_Utah_stations_20221017.csv')

        # Read MesoWest data
        df = pd.read_csv(file)
        stations = gpd.GeoDataFrame(df,
                                    geometry=gpd.points_from_xy(df.Longitude,
                                                                df.Latitude))
        # stations.set_crs(epsg=4326, inplace=True)  # changes figsize

        # Filter MesoWest data
        if status is not None:
            stations = stations[stations.Status == status.upper()]

        if networks is not None:
            stations = stations[stations.Mesonet.isin(networks)]

        # Add to axis
        stations.plot(ax=self.ax, transform=ccrs.PlateCarree(), c='black',
                      zorder=6, marker='x')

        return 'MesoWest'

    @collect_feature
    def add_radiosonde(self):
        # TODO
        raise ValueError('radiosonde not implemented!')
        return 'radiosonde'

    @collect_feature
    def add_helicopter(self):
        # TODO
        raise ValueError('helicopter not implemented!')
        return 'helicopter'

    @collect_feature
    def add_DAQ(self):
        # TODO
        raise ValueError('DAQ not implemented!')
        return 'DAQ'

    @collect_feature
    def add_scale_bar(self):
        # TODO
        # Addingg a scale bar with cartopy is difficult (2023-04-13)
        # Check: https://github.com/SciTools/cartopy/issues/490

        # Otherwise: https://stackoverflow.com/questions/32333870/how-can-i-show-a-km-ruler-on-a-cartopy-matplotlib-plot
        raise ValueError('scale bar not implemented!')
        return 'scale bar'

    @collect_feature
    def add_north_arrow(self):
        import matplotlib.patheffects as pe
        # TODO
        #   add path_effects=buffer
        north_arrow = u'\u25B2\nN'

        self.ax.text(0.965, 0.06, north_arrow, transform=self.ax.transAxes,
                     fontsize=16, ha='center', va='center', zorder=8,
                     path_effects=[pe.withStroke(linewidth=3,
                                                 foreground='white')])
        return 'north arrow'

    @collect_feature
    def add_legend(self, legend_kws, legend_mapper):
        import matplotlib.collections as mcol
        from matplotlib.lines import Line2D
        from utils.plotter import HandlerDashedLines

        def TRAX_legend():
            line = [[(0, 0)]]

            lc = mcol.LineCollection(len(self.TRAX_lines) * line,
                                     colors=self.TRAX_lines, linewidth=4)

            return lc

        def UUCON_legend():
            handle = Line2D([0], [0], marker='o', color='black', markersize=13,
                            markerfacecolor='None', linestyle='None',
                            markeredgewidth=2.8)
            return handle

        def MesoWest_legend():
            handle = Line2D([0], [0], marker='x', color='black', markersize=8,
                            linestyle='None', markeredgewidth=1.5)
            return handle

        legend_features = {'TRAX': TRAX_legend,
                           'UUCON': UUCON_legend,
                           'MesoWest': MesoWest_legend}

        handles, labels = [], []
        for label, handle in legend_features.items():
            if label not in self.features:
                continue

            handles.append(legend_features[label]())

            # If map is specified for a label, use mapped label,
            #   otherwise use original label
            if legend_mapper:
                label = legend_mapper.get(label, label)
            labels.append(label)

        self.ax.legend(handles, labels, loc='upper left',
                       handler_map={mcol.LineCollection:
                                    HandlerDashedLines()},
                       handlelength=2.5, handleheight=3, labelspacing=0.1)

        return 'legend'

    @collect_feature
    def add_extent_map(self):
        from utils.grid import add_extent_map

        fig = self.ax.get_figure()

        extent_map_extent = [-125, -105, 30, 50]
        extent_map_proj = ccrs.AlbersEqualArea(central_longitude=-111)
        extent_map_rect = [0.65, 0.67, 0.2, 0.2]

        extent_map_ax = add_extent_map(fig, self.extent, ccrs.PlateCarree(),
                                       extent_map_rect, extent_map_extent,
                                       extent_map_proj, 'red', 3, zorder=8)
