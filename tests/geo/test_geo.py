"""Tests for lair.geo.

Requires the `geo` extra (cartopy/shapely/pyproj/rasterio/rioxarray) — skipped
when not installed (run in the conda lair-dev env for full coverage). The pure
geometry/coordinate helpers are tested here; the plotting and regridding paths
are not.
"""

import numpy as np
import pytest

# exc_type=ImportError: lair._optional re-raises missing extras as a plain
# ImportError (not ModuleNotFoundError), which importorskip ignores by default.
geo = pytest.importorskip(
    "lair.geo", reason="requires the `geo` extra", exc_type=ImportError
)


class TestBboxExtent:
    def test_bbox2extent(self):
        # bbox [W, S, E, N] -> extent [W, E, S, N]
        assert geo.bbox2extent([-112, 40, -111, 41]) == [-112, -111, 40, 41]

    def test_extent2bbox(self):
        assert geo.extent2bbox([-112, -111, 40, 41]) == [-112, 40, -111, 41]

    def test_round_trip(self):
        bbox = [-112, 40, -111, 41]
        assert geo.extent2bbox(geo.bbox2extent(bbox)) == bbox


class TestDMS:
    def test_dms2dd(self):
        assert geo.dms2dd(40, 30, 0) == pytest.approx(40.5)

    def test_dms2dd_seconds(self):
        assert geo.dms2dd(0, 0, 3600) == pytest.approx(1.0)

    def test_dms2dd_negative_degrees(self):
        # d + m/60 + s/3600 (sign carried by d only).
        assert geo.dms2dd(-40, 30, 0) == pytest.approx(-39.5)


class TestWrapLons:
    def test_wraps_into_180_range(self):
        out = np.asarray(geo.wrap_lons(np.array([10.0, 190.0, 350.0])))
        np.testing.assert_allclose(out, [10.0, -170.0, -10.0])


class TestDistanceHelpers:
    def test_bearing_due_north(self):
        assert geo.bearing(40, -111, 41, -111) == pytest.approx(0.0)

    def test_haversine_one_degree_latitude(self):
        # ~111 km per degree of latitude.
        assert geo.haversine(40, -111, 41, -111) == pytest.approx(111.19, abs=0.1)

    def test_earth_radius_equator(self):
        assert geo.earth_radius(0.0) == pytest.approx(6378.137, abs=1e-3)

    def test_cosine_weights(self):
        np.testing.assert_allclose(
            geo.cosine_weights(np.array([0.0, 60.0])), [1.0, 0.5]
        )

    def test_bearing_final_bearing(self):
        # The final bearing along a great circle differs from the initial.
        b = geo.bearing(40, -111, 41, -110, final=True)
        assert 0.0 <= b < 360.0

    def test_haversine_radius_argument(self):
        # One degree of arc on a unit sphere is 1 degree in radians.
        assert geo.haversine(40, -111, 41, -111, R=1.0) == pytest.approx(
            np.deg2rad(1.0), abs=1e-6
        )

    def test_earth_radius_decreases_to_poles(self):
        r = np.asarray(geo.earth_radius(np.array([0.0, 90.0])))
        assert r[0] == pytest.approx(6378.137, abs=1e-3)  # equatorial
        assert r[1] < r[0]  # polar radius is smaller


class TestCRS:
    def test_epsg_roundtrip(self):
        crs = geo.CRS(4326)
        assert crs.epsg == 4326

    def test_proj4_and_wkt_are_strings(self):
        crs = geo.CRS(4326)
        assert isinstance(crs.proj4, str) and crs.proj4
        assert isinstance(crs.wkt, str) and crs.wkt

    def test_repr_and_str_mention_epsg(self):
        crs = geo.CRS(4326)
        assert "4326" in str(crs)
        assert "4326" in repr(crs)

    def test_conversions_return_objects(self):
        crs = geo.CRS(4326)
        assert crs.to_cartopy() is not None
        assert crs.to_rasterio() is not None
        assert crs.to_pyproj() is not None


def test_write_rio_crs_sets_crs():
    grid = geo.generate_regular_grid(-112, -108, 1.0, 40, 44, 1.0)
    assert grid.rio.crs is None
    out = geo.write_rio_crs(grid, 4326)
    assert out.rio.crs is not None


def test_basegrid_copy_is_independent():
    grid = geo.write_rio_crs(
        geo.generate_regular_grid(-112, -110, 1.0, 40, 42, 1.0), 4326
    )
    bgrid = geo.BaseGrid(grid, crs=geo.CRS(4326))
    assert isinstance(bgrid.copy(), geo.BaseGrid)


def test_points_along_line():
    from shapely import LineString

    pts = geo.points_along_line(LineString([(0, 0), (0, 1)]), spacing=0.25)
    # 0, 0.25, 0.5, 0.75, 1.0 -> 5 points along the segment.
    assert len(pts) == 5


def test_generate_regular_grid_returns_dataarray():
    from xarray import DataArray

    grid = geo.generate_regular_grid(-112, -110, 1.0, 40, 42, 1.0)
    assert isinstance(grid, DataArray)
    assert grid.dims == ("y", "x")


def test_round_latlon():
    import numpy as np
    import xarray as xr

    da = xr.DataArray(
        np.zeros((2, 2)),
        coords={"lat": [40.001, 41.004], "lon": [-112.002, -111.006]},
        dims=["lat", "lon"],
    )
    out = geo.round_latlon(da, lat_deci=2, lon_deci=2)
    assert out.lat.values.tolist() == [40.0, 41.0]
    assert out.lon.values.tolist() == [-112.0, -111.01]


def test_basegrid_construction():
    grid = geo.generate_regular_grid(-112, -110, 1.0, 40, 42, 1.0)
    bgrid = geo.BaseGrid(grid, crs=geo.CRS(4326))
    assert isinstance(bgrid, geo.BaseGrid)


@pytest.fixture
def latlon_grid():
    """A 5x6 regular lat/lon Dataset (1 deg) with rio CRS + cf-recognisable axes.

    This is the shape the clip/regrid/resample helpers expect: a Dataset (not a
    DataArray — cf.add_bounds is Dataset-only) with 1D lat/lon carrying
    degrees_north/east units, rio spatial dims set, and an EPSG:4326 CRS.
    """
    import xarray as xr

    lat = np.arange(40.0, 45.0, 1.0)
    lon = np.arange(-114.0, -108.0, 1.0)
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"emis": (("lat", "lon"), rng.random((len(lat), len(lon))))},
        coords={"lat": lat, "lon": lon},
    )
    ds.lat.attrs["units"] = "degrees_north"
    ds.lon.attrs["units"] = "degrees_east"
    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    return geo.write_rio_crs(ds, 4326)


class TestClip:
    def test_clip_bbox_is_inclusive(self, latlon_grid):
        clipped = geo.clip(latlon_grid, bbox=(-113, 41, -110, 43), crs=4326)
        assert clipped.sizes == {"lat": 3, "lon": 4}
        # The clipped grid is a strict subset of the original extent.
        assert clipped.lat.min() >= 41 and clipped.lat.max() <= 43

    def test_clip_extent_matches_bbox(self, latlon_grid):
        by_bbox = geo.clip(latlon_grid, bbox=(-113, 41, -110, 43), crs=4326)
        by_extent = geo.clip(latlon_grid, extent=(-113, -110, 41, 43), crs=4326)
        assert by_bbox.sizes == by_extent.sizes

    def test_clip_geom(self, latlon_grid):
        from shapely.geometry import box

        clipped = geo.clip(latlon_grid, geom=box(-113, 41, -110, 43), crs=4326)
        # geom clipping is exclusive of the outer bounds -> smaller than bbox.
        assert clipped.sizes["lat"] <= 3 and clipped.sizes["lon"] <= 4

    def test_requires_exactly_one_selector(self, latlon_grid):
        with pytest.raises(AssertionError):
            geo.clip(
                latlon_grid, bbox=(-113, 41, -110, 43), extent=(-113, -110, 41, 43)
            )
        with pytest.raises(AssertionError):
            geo.clip(latlon_grid)


class TestGridcellArea:
    def test_areas_positive_and_shaped(self, latlon_grid):
        area = geo.gridcell_area(latlon_grid)
        assert area.sizes == {"lat": 5, "lon": 6}
        assert float(area.min()) > 0

    def test_area_decreases_toward_poles(self, latlon_grid):
        area = geo.gridcell_area(latlon_grid)
        # Cell area shrinks with latitude (cos weighting).
        assert float(area.isel(lat=0).mean()) > float(area.isel(lat=-1).mean())


class TestResampleRegrid:
    def test_resample_coarsens(self, latlon_grid):
        coarse = geo.resample(latlon_grid, 2.0)
        # Coarser resolution -> fewer cells than the 1-degree input.
        assert coarse.sizes["lat"] < latlon_grid.sizes["lat"]
        assert coarse.sizes["lon"] < latlon_grid.sizes["lon"]
        assert coarse.rio.crs is not None

    def test_regrid_to_target_grid(self, latlon_grid):
        out_grid = geo.generate_regular_grid(
            -114, -108, 2.0, 40, 45, 2.0, x_label="lon", y_label="lat"
        )
        out_grid.lat.attrs["units"] = "degrees_north"
        out_grid.lon.attrs["units"] = "degrees_east"
        regridded = geo.regrid(latlon_grid, out_grid)
        assert regridded.sizes["lat"] == out_grid.sizes["lat"]
        assert regridded.sizes["lon"] == out_grid.sizes["lon"]


class TestBaseGridOperations:
    def test_clip_returns_new_grid(self, latlon_grid):
        bgrid = geo.BaseGrid(latlon_grid, crs=4326)
        clipped = bgrid.clip(bbox=(-113, 41, -110, 43))
        assert clipped is not bgrid
        assert clipped.data.sizes == {"lat": 3, "lon": 4}
        # Original is untouched (not inplace).
        assert bgrid.data.sizes == {"lat": 5, "lon": 6}

    def test_clip_inplace(self, latlon_grid):
        bgrid = geo.BaseGrid(latlon_grid, crs=4326)
        same = bgrid.clip(bbox=(-113, 41, -110, 43), inplace=True)
        assert same is bgrid
        assert bgrid.data.sizes == {"lat": 3, "lon": 4}

    def test_resample(self, latlon_grid):
        bgrid = geo.BaseGrid(latlon_grid, crs=4326)
        coarse = bgrid.resample(2.0)
        assert coarse.data.sizes["lat"] < latlon_grid.sizes["lat"]

    def test_regrid_to_target(self, latlon_grid):
        out_grid = geo.generate_regular_grid(
            -114, -108, 2.0, 40, 45, 2.0, x_label="lon", y_label="lat"
        )
        out_grid.lat.attrs["units"] = "degrees_north"
        out_grid.lon.attrs["units"] = "degrees_east"
        bgrid = geo.BaseGrid(latlon_grid, crs=4326)
        regridded = bgrid.regrid(out_grid)
        assert regridded.data.sizes["lat"] == out_grid.sizes["lat"]

    def test_gridcell_area_property(self, latlon_grid):
        bgrid = geo.BaseGrid(latlon_grid, crs=4326)
        assert float(bgrid.gridcell_area.min()) > 0

    def test_reproject_rejects_latlon(self, latlon_grid):
        # reproject asserts the source CRS is not already 4326.
        bgrid = geo.BaseGrid(latlon_grid, crs=4326)
        with pytest.raises(AssertionError):
            bgrid.reproject(2.0)


class TestTickHelpers:
    """Smoke-exercise the cartopy lat/lon tick formatters on a headless axis."""

    @pytest.fixture(autouse=True)
    def _close_figures(self):
        yield
        import matplotlib.pyplot as plt

        plt.close("all")

    @staticmethod
    def _geo_ax(extent):
        import matplotlib

        matplotlib.use("Agg")
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})[1]
        # Give cartopy a concrete extent so tick computation is deterministic.
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        return ax

    def test_add_latlon_ticks(self):
        ax = self._geo_ax([-113, -110, 39, 42])
        geo.add_latlon_ticks(ax, extent=[-113, -110, 39, 42])

    def test_add_lat_and_lon_ticks(self):
        ax = self._geo_ax([-113, -110, 39, 42])
        geo.add_lat_ticks(ax, ylims=[39, 42])
        geo.add_lon_ticks(ax, xlims=[-113, -110])
