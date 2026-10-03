"""Tests for lair.geo.

Requires the `geo` extra (cartopy/shapely/pyproj/rasterio/rioxarray) — skipped
when not installed (run in the conda lair-dev env for full coverage). The pure
geometry/coordinate helpers are tested here; the plotting and regridding paths
are not.
"""

import numpy as np
import pytest
from xarray import DataArray

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
        # The sign of d applies to the whole angle: -40 deg 30' is -40.5, not -39.5.
        assert geo.dms2dd(-40, 30, 0) == pytest.approx(-40.5)
        assert geo.dms2dd(-111, 30, 36) == pytest.approx(-111.51)

    def test_dms2dd_negative_zero_degrees(self):
        # -0 deg 30' (a float -0.0 keeps its sign)
        assert geo.dms2dd(-0.0, 30) == pytest.approx(-0.5)

    def test_dms2dd_strings(self):
        assert geo.dms2dd("40", "30", "0") == pytest.approx(40.5)

    @pytest.mark.parametrize("bad", ["abc", None])
    def test_dms2dd_unparseable_is_nan(self, bad):
        assert np.isnan(geo.dms2dd(40, bad))


class TestWrapLons:
    def test_interval_is_half_open(self):
        # [base, base + period): 180 wraps to -180.
        out = np.asarray(geo.wrap_lons(np.array([-180.0, 180.0])))
        np.testing.assert_allclose(out, [-180.0, -180.0])

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

    def test_bearing_final_bearing_in_range(self):
        b = geo.bearing(40, -111, 41, -110, final=True)
        assert 0.0 <= b < 360.0

    def test_haversine_near_antipodal_is_not_nan(self):
        # Rounding can push the haversine term a just above 1 -> sqrt(1 - a) NaN.
        d = geo.haversine(-87.5, -180.0, 87.5, 0.0)  # exact antipodes
        assert not np.isnan(d)
        assert d == pytest.approx(np.pi * 6371, rel=1e-6)

    def test_haversine_broadcasts_scalar_against_array(self):
        # A fixed point against many points (e.g. distance from a site).
        d = geo.haversine(40, -111, np.array([41.0, 42.0]), np.array([-111.0, -111.0]))
        np.testing.assert_allclose(d, [111.19, 222.39], atol=0.1)


# Worked example from https://www.movable-type.co.uk/scripts/latlong.html:
# Land's End (50 03 59N, 005 42 53W) -> John o' Groats (58 38 38N, 003 04 12W)
LANDS_END = (50 + 3 / 60 + 59 / 3600, -(5 + 42 / 60 + 53 / 3600))
JOHN_O_GROATS = (58 + 38 / 60 + 38 / 3600, -(3 + 4 / 60 + 12 / 3600))


class TestBearing:
    """lair#23: verify bearing against published and independent references."""

    @pytest.mark.parametrize(
        "p2, expected",
        [
            ((41, -111), 0.0),  # north
            ((40, -110), 89.678),  # east (not exactly 90: great circle)
            ((39, -111), 180.0),  # south
            ((40, -112), 270.322),  # west
        ],
    )
    def test_cardinal_directions(self, p2, expected):
        assert geo.bearing(40, -111, *p2) == pytest.approx(expected, abs=1e-3)

    def test_movable_type_initial(self):
        # 009 07 11 on the reference page
        b = geo.bearing(*LANDS_END, *JOHN_O_GROATS)
        assert b == pytest.approx(9 + 7 / 60 + 11 / 3600, abs=1 / 3600)

    def test_movable_type_final(self):
        # 011 16 31 on the reference page. The old implementation returned
        # initial + 180 (~189 deg) here.
        b = geo.bearing(*LANDS_END, *JOHN_O_GROATS, final=True)
        assert b == pytest.approx(11 + 16 / 60 + 31 / 3600, abs=1 / 3600)

    def test_final_equals_reverse_of_return_leg(self):
        fwd_final = geo.bearing(35, 45, 35, 135, final=True)
        back_initial = geo.bearing(35, 135, 35, 45)
        assert fwd_final == pytest.approx((back_initial + 180) % 360)

    def test_meridian_initial_equals_final(self):
        assert geo.bearing(40, -111, 45, -111, final=True) == pytest.approx(0.0)

    def test_radians_input_gives_degrees_output(self):
        b = geo.bearing(*np.deg2rad([35, 45, 35, 135]), deg=False)
        assert b == pytest.approx(geo.bearing(35, 45, 35, 135))

    def test_coincident_points(self):
        assert geo.bearing(40, -111, 40, -111) == 0.0

    def test_broadcasts_scalar_against_array(self):
        b = geo.bearing(40, -111, np.array([41.0, 39.0]), np.array([-111.0, -111.0]))
        np.testing.assert_allclose(b, [0.0, 180.0], atol=1e-9)

    def test_matches_pyproj_sphere(self):
        """Initial and final bearings agree with pyproj's geodesic on a sphere."""
        pyproj = pytest.importorskip("pyproj")
        g = pyproj.Geod(a=6371000, b=6371000)
        rng = np.random.default_rng(0)
        lat1, lat2 = rng.uniform(-80, 80, (2, 200))
        lon1, lon2 = rng.uniform(-180, 180, (2, 200))
        az12, az21, _ = g.inv(lon1, lat1, lon2, lat2)
        init = geo.bearing(lat1, lon1, lat2, lon2)
        final = geo.bearing(lat1, lon1, lat2, lon2, final=True)

        # compare on the circle (359.9999 == 0.0000)
        def angdiff(a, b):
            return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)

        assert angdiff(init, az12 % 360).max() < 1e-6
        assert angdiff(final, (az21 + 180) % 360).max() < 1e-6
        assert ((init >= 0) & (init < 360)).all()

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


def _points_along_line_cases():
    from shapely import LineString, MultiLineString

    curve = [(np.cos(t), np.sin(t)) for t in np.linspace(0, np.pi, 25)]
    lasso = [(0, 0), (1, 0), (1.5, 0.5), (1, 1), (1, 0), (1.2, -0.7)]
    return {
        "straight": (LineString([(0, 0), (0, 1)]), 0.25),
        "diagonal": (LineString([(0, 0), (1, 1)]), 0.3),
        "branch": (MultiLineString([[(0, 0), (2, 0)], [(1, 0), (1, 1.5)]]), 0.4),
        "cross": (MultiLineString([[(-1, 0), (1, 0)], [(0, -1), (0, 1)]]), 0.35),
        "loop": (LineString([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]), 0.3),
        "curve": (LineString(curve), 0.2),
        "disjoint": (MultiLineString([[(0, 0), (1, 0)], [(0, 0.1), (1, 0.1)]]), 0.25),
        "lasso": (LineString(lasso), 0.3),
    }


# Output of the original (state-level BFS) implementation, before the #35
# rewrite. The rewrite must reproduce it exactly, except where float error in
# the old exact distance comparisons skipped a node exactly `spacing` away
# (branch: 1.2 - 0.8 = 0.3999999999999999 < 0.4 put the point at 1.24).
_POINTS_ALONG_LINE_EXPECTED = {
    "straight": [
        (0.0, 0.0),
        (0.0, 0.25),
        (0.0, 0.5),
        (0.0, 0.75),
        (0.0, 1.0),
    ],
    "diagonal": [
        (0.0, 0.0),
        (0.22917, 0.22917),
        (0.45833, 0.45833),
        (0.6875, 0.6875),
        (0.91667, 0.91667),
    ],
    "branch": [
        (0.0, 0.0),
        (0.4, 0.0),
        (0.8, 0.0),
        (1.2, 0.0),
        (1.0, 0.35526),
        (1.6, 0.0),
        (1.0, 0.78947),
        (2.0, 0.0),
        (1.0, 1.22368),
    ],
    "cross": [
        (-1.0, 0.0),
        (-0.62069, 0.0),
        (-0.24138, 0.0),
        (0.13793, 0.0),
        (0.0, -0.34483),
        (0.0, 0.34483),
        (0.51724, 0.0),
        (0.0, -0.72414),
        (0.0, 0.72414),
        (0.89655, 0.0),
    ],
    "loop": [
        (0.0, 0.0),
        (0.32353, 0.0),
        (0.0, 0.32353),
        (0.64706, 0.0),
        (0.0, 0.64706),
        (0.97059, 0.0),
        (0.0, 0.97059),
        (1.0, 0.32353),
        (0.32353, 1.0),
        (1.0, 0.64706),
        (0.64706, 1.0),
        (1.0, 0.97059),
    ],
    "curve": [
        (1.0, 0.0),
        (0.97686, 0.20384),
        (0.91561, 0.39944),
        (0.81412, 0.57769),
        (0.67901, 0.73175),
        (0.51554, 0.85564),
        (0.3296, 0.9419),
        (0.13053, 0.99144),
        (-0.07459, 0.99511),
        (-0.27651, 0.95992),
        (-0.46648, 0.88256),
        (-0.63686, 0.76871),
        (-0.78103, 0.62281),
        (-0.89082, 0.44972),
        (-0.96593, 0.25882),
        (-0.99633, 0.05594),
    ],
    "disjoint": [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.75, 0.0),
        (1.0, 0.0),
    ],
    "lasso": [
        (0.0, 0.0),
        (0.32353, 0.0),
        (0.64706, 0.0),
        (0.97059, 0.0),
        (1.20833, 0.20833),
        (1.08, -0.28),
        (1.0, 0.44118),
        (1.4375, 0.4375),
        (1.168, -0.588),
        (1.0, 0.76471),
    ],
}


class TestPointsAlongLine:
    @pytest.mark.parametrize("case", list(_POINTS_ALONG_LINE_EXPECTED))
    def test_matches_reference(self, case):
        line, spacing = _points_along_line_cases()[case]
        pts = geo.points_along_line(line, spacing=spacing)
        assert [(p.x, p.y) for p in pts] == _POINTS_ALONG_LINE_EXPECTED[case]

    @pytest.mark.parametrize("case", list(_POINTS_ALONG_LINE_EXPECTED))
    def test_points_at_least_spacing_apart(self, case):
        line, spacing = _points_along_line_cases()[case]
        xy = np.array([(p.x, p.y) for p in geo.points_along_line(line, spacing)])
        d = np.hypot(*(xy[:, None, :] - xy[None, :, :]).transpose(2, 0, 1))
        # Up to float error: distances of exactly `spacing` are allowed
        assert d[np.triu_indices(len(xy), k=1)].min() >= spacing - 1e-12

    def test_rejects_nonpositive_spacing(self):
        from shapely import LineString

        with pytest.raises(ValueError, match="spacing"):
            geo.points_along_line(LineString([(0, 0), (1, 0)]), spacing=0)

    def test_small_spacing_is_even(self):
        # Sub-1e-5 steps used to collapse onto a fixed 5-decimal grid, giving
        # 19 points spaced 5e-5 or 6e-5 (#35)
        from shapely import LineString

        pts = geo.points_along_line(LineString([(0, 0), (0.001, 0)]), spacing=5e-5)
        x = np.array([p.x for p in pts])
        assert len(pts) == 21
        np.testing.assert_allclose(np.diff(np.sort(x)), 5e-5)

    def test_long_line(self):
        # Runtime is near-linear in the number of points; this took tens of
        # minutes with the original state-level BFS (#35)
        from shapely import LineString

        pts = geo.points_along_line(LineString([(0, 0), (1000, 0)]), spacing=1.0)
        assert len(pts) == 1001
        np.testing.assert_allclose([p.x for p in pts], np.arange(1001))


def test_generate_regular_grid_returns_dataarray():
    from xarray import DataArray

    grid = geo.generate_regular_grid(-112, -110, 1.0, 40, 42, 1.0)
    assert isinstance(grid, DataArray)
    assert grid.dims == ("y", "x")


class TestGenerateRegularGrid:
    def test_quarter_degree_centres_not_rounded(self):
        grid = geo.generate_regular_grid(0, 1, 0.25, 0, 1, 0.25)
        np.testing.assert_allclose(grid.x.values, [0.125, 0.375, 0.625, 0.875])
        np.testing.assert_allclose(np.diff(grid.y.values), 0.25)

    def test_large_spacing_centres_not_rounded(self):
        grid = geo.generate_regular_grid(0, 100, 25, 0, 100, 25)
        np.testing.assert_allclose(grid.x.values, [12.5, 37.5, 62.5, 87.5])

    def test_offset_origin(self):
        grid = geo.generate_regular_grid(-112.0125, -111.0, 0.1, 40.0, 41.0, 0.1)
        assert grid.x.values[0] == pytest.approx(-111.9625, abs=1e-12)
        np.testing.assert_allclose(np.diff(grid.x.values), 0.1)
        assert grid.y.values.tolist()[:3] == [40.05, 40.15, 40.25]

    def test_cell_count(self):
        # Cells whose centre falls inside [min, max)
        grid = geo.generate_regular_grid(-112.2, -111.75, 0.05, 40.45, 40.93, 0.05)
        assert grid.sizes == {"y": 10, "x": 9}

    def test_chunks(self):
        pytest.importorskip("dask")
        grid = geo.generate_regular_grid(0, 4, 1, 0, 4, 1, chunks={"x": 2, "y": 2})
        assert grid.chunks == ((2, 2), (2, 2))


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

    This is the shape the clip/regrid/resample helpers expect: 1D lat/lon
    carrying degrees_north/east units, rio spatial dims set, and an EPSG:4326
    CRS. (``latlon_grid.emis`` is the DataArray version.)
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

    @pytest.mark.parametrize("container", ["list", "ndarray", "geoseries"])
    def test_clip_geom_collections(self, latlon_grid, container):
        from shapely.geometry import box

        geoms = [box(-113, 41, -110, 43)]
        if container == "ndarray":
            geoms = np.array(geoms)
        elif container == "geoseries":
            gpd = pytest.importorskip("geopandas")
            geoms = gpd.GeoSeries(geoms)
        clipped = geo.clip(latlon_grid, geom=geoms, crs=4326)
        single = geo.clip(latlon_grid, geom=box(-113, 41, -110, 43), crs=4326)
        assert clipped.sizes == single.sizes

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

    def test_radius_array(self, latlon_grid):
        # An array R (e.g. per-latitude radius) must not be truth-tested.
        R = geo.earth_radius(latlon_grid["lat"])
        area = geo.gridcell_area(latlon_grid, R=R)
        np.testing.assert_allclose(area, geo.gridcell_area(latlon_grid))

    def test_radius_scalar(self, latlon_grid):
        a1 = geo.gridcell_area(latlon_grid, R=1.0)
        a2 = geo.gridcell_area(latlon_grid, R=2.0)
        np.testing.assert_allclose(a2, 4 * a1)

    @pytest.mark.parametrize("descending_y", [False, True])
    def test_metre_grid(self, descending_y):
        # A 3 x 4 grid of 2 km x 1 km cells in UTM 12N; north-up rasters store
        # y descending, which must not give negative areas.
        import xarray as xr

        x = 4e5 + 2000.0 * np.arange(4)
        y = 4.5e6 + 1000.0 * np.arange(3)
        if descending_y:
            y = y[::-1]
        ds = xr.Dataset(
            {"v": (("y", "x"), np.ones((3, 4)))}, coords={"y": y, "x": x}
        ).rio.set_spatial_dims(x_dim="x", y_dim="y")
        ds = geo.write_rio_crs(ds, 32612)
        area = geo.gridcell_area(ds)
        assert area.dims == ("y", "x")
        np.testing.assert_allclose(area.values, 2.0)

    def test_from_latlon(self, latlon_grid):
        area = geo.gridcell_area_from_latlon(latlon_grid.lat, latlon_grid.lon)
        assert isinstance(area, np.ndarray)
        np.testing.assert_allclose(area, geo.gridcell_area(latlon_grid).values)


class TestResampleRegrid:
    def test_resample_centres_not_rounded(self, latlon_grid):
        # 0.25 deg cells starting at the 39.5 edge are centred on 39.625, not 39.62
        fine = geo.resample(latlon_grid, 0.25)
        assert fine.lat.values[0] == pytest.approx(39.625, abs=1e-12)
        np.testing.assert_allclose(np.diff(fine.lat.values), 0.25)

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


class TestDataArrayInput:
    """DataArray in -> DataArray out for the cf-bounds helpers (#34).

    cf_xarray's bounds helpers are Dataset-only, so these used to raise
    AttributeError on a DataArray.
    """

    @pytest.fixture
    def da(self, latlon_grid):
        da = latlon_grid.emis
        da.attrs["units"] = "kg m-2 s-1"
        return da

    def test_gridcell_area(self, latlon_grid, da):
        area = geo.gridcell_area(da)
        np.testing.assert_allclose(area, geo.gridcell_area(latlon_grid))

    def test_gridcell_area_metre_grid(self):
        import xarray as xr

        x = 4e5 + 2000.0 * np.arange(4)
        y = 4.5e6 + 1000.0 * np.arange(3)
        da = xr.DataArray(
            np.ones((3, 4)), coords={"y": y, "x": x}, dims=("y", "x"), name="v"
        )
        da = geo.write_rio_crs(da.rio.set_spatial_dims(x_dim="x", y_dim="y"), 32612)
        np.testing.assert_allclose(geo.gridcell_area(da).values, 2.0)

    def test_resample(self, latlon_grid, da):
        out = geo.resample(da, 2.0)
        assert isinstance(out, DataArray)
        assert out.name == "emis"
        assert out.attrs["units"] == "kg m-2 s-1"
        assert out.rio.crs is not None
        np.testing.assert_allclose(out, geo.resample(latlon_grid, 2.0).emis)

    def test_resample_unnamed(self, da):
        out = geo.resample(da.rename(None), 2.0)
        assert isinstance(out, DataArray)
        assert out.name is None

    def test_regrid(self, latlon_grid, da):
        out_grid = geo.generate_regular_grid(
            -114, -108, 2.0, 40, 45, 2.0, x_label="lon", y_label="lat"
        )
        out_grid.lat.attrs["units"] = "degrees_north"
        out_grid.lon.attrs["units"] = "degrees_east"
        out = geo.regrid(da, out_grid)
        assert isinstance(out, DataArray)
        assert out.name == "emis"
        np.testing.assert_allclose(out, geo.regrid(latlon_grid, out_grid).emis)

    def test_basegrid(self, da):
        bgrid = geo.BaseGrid(da, crs=4326)
        coarse = bgrid.resample(2.0)
        assert isinstance(coarse.data, DataArray)
        assert coarse.data.sizes["lat"] < da.sizes["lat"]
        assert float(bgrid.gridcell_area.min()) > 0


class TestGridcellAreaCRS:
    """Any geographic CRS is lat/lon, not only EPSG:4326 (#34)."""

    @pytest.mark.parametrize("crs", ["OGC:CRS84", 4269])
    def test_geographic_crs(self, latlon_grid, crs):
        area = geo.gridcell_area(geo.write_rio_crs(latlon_grid, crs))
        np.testing.assert_allclose(area, geo.gridcell_area(latlon_grid))

    def test_no_crs_is_value_error(self, latlon_grid):
        grid = latlon_grid.drop_vars("spatial_ref")
        assert grid.rio.crs is None
        with pytest.raises(ValueError, match="no CRS"):
            geo.gridcell_area(grid)

    def test_non_metre_projected_crs(self, latlon_grid):
        # Utah Central in US survey feet: projected, but not metres
        grid = geo.write_rio_crs(latlon_grid, 3566)
        with pytest.raises(ValueError, match="Only lat-lon and meter grids"):
            geo.gridcell_area(grid)


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

    @pytest.mark.parametrize("dpi", [100, 300])
    def test_tick_count_ignores_dpi(self, dpi):
        ax = self._geo_ax([-113, -110, 39, 42])
        ax.figure.set_dpi(dpi)
        geo.add_latlon_ticks(ax, extent=[-113, -110, 39, 42])
        # Same as at the default 100 dpi
        ref = self._geo_ax([-113, -110, 39, 42])
        geo.add_latlon_ticks(ref, extent=[-113, -110, 39, 42])
        assert len(ax.get_xticks()) == len(ref.get_xticks())
        assert len(ax.get_yticks()) == len(ref.get_yticks())

    def test_add_lat_and_lon_ticks(self):
        ax = self._geo_ax([-113, -110, 39, 42])
        geo.add_lat_ticks(ax, ylims=[39, 42])
        geo.add_lon_ticks(ax, xlims=[-113, -110])


class TestPlotting:
    @pytest.fixture(autouse=True)
    def _agg(self):
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("Agg")
        yield
        plt.close("all")

    @pytest.mark.parametrize("crs", [None, 4326, "EPSG:4326", 5070])
    def test_plot_grid_crs_inputs(self, latlon_grid, crs):
        ax = geo.plot_grid(latlon_grid, crs=crs)
        assert ax is not None

    def test_plot_grid_accepts_crs_wrapper(self, latlon_grid):
        assert geo.plot_grid(latlon_grid, crs=geo.CRS(5070)) is not None

    def test_plot_grid_extent_is_lonlat(self, latlon_grid):
        # extent is lon/lat even when the map is projected
        import cartopy.crs as ccrs

        extent = [-113, -110, 41, 43]
        ax = geo.plot_grid(latlon_grid, extent=extent, crs=ccrs.LambertConformal())
        x0, x1, y0, y1 = ax.get_extent(crs=ccrs.PlateCarree())
        # The projected view encloses the requested lon/lat box (not metres)
        assert -115 < x0 <= -113 and -110 <= x1 < -108
        assert 39 < y0 <= 41 and 43 <= y1 < 45

    def test_add_extent_map_outlines_box(self):
        import cartopy.crs as ccrs
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = geo.add_extent_map(
            fig,
            main_extent=[-113, -110, 39, 42],
            main_extent_crs=ccrs.PlateCarree(),
            extent_map_rect=(0.0, 0.0, 0.3, 0.3),
            extent_map_extent=[-125, -100, 30, 50],
            extent_map_crs=ccrs.PlateCarree(),
            color="red",
            linewidth=2,
        )
        box_artist = ax.collections[-1]
        # An outline: red edge, no fill
        np.testing.assert_allclose(box_artist.get_edgecolor(), [[1, 0, 0, 1]])
        face = np.asarray(box_artist.get_facecolor())
        assert face.size == 0 or np.all(face[:, 3] == 0)
