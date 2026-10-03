"""Tests for lair.hrrr.

Requires the `requests`/`formats`/`geo` extras (boto3/numcodecs/cartopy) —
skipped when not installed (run in the conda lair-dev env for full coverage).
S3 URL construction, chunk decoding and the Winds download are tested against
in-memory chunks served by a fake boto3 resource; live HRRR fetches are not.
"""

import datetime as dt

import numpy as np
import pytest

hrrr = pytest.importorskip(
    "lair.hrrr", reason="requires boto3/numcodecs/cartopy", exc_type=ImportError
)


@pytest.fixture
def zarr_id():
    return hrrr.ZarrId(
        run_hour=dt.datetime(2024, 7, 18, 0),
        level_type="sfc",
        var_level="surface",
        var_name="TMP",
        model_type="fcst",
    )


class TestS3Urls:
    def test_group_url(self, zarr_id):
        assert (
            hrrr.create_s3_group_url(zarr_id)
            == "s3://hrrrzarr/sfc/20240718/20240718_00z_fcst.zarr/surface/TMP"
        )

    def test_subgroup_url(self, zarr_id):
        assert (
            hrrr.create_s3_subgroup_url(zarr_id)
            == "s3://hrrrzarr/sfc/20240718/20240718_00z_fcst.zarr/surface/TMP/surface"
        )

    def test_group_url_without_prefix(self, zarr_id):
        url = hrrr.create_s3_group_url(zarr_id, prefix=False)
        assert not url.startswith("s3://")
        assert url.endswith("surface/TMP")

    def test_chunk_url(self, zarr_id):
        assert hrrr.create_s3_chunk_url(zarr_id, "0.0").endswith(
            "surface/TMP/surface/TMP/0.0.0"
        )


def test_format_chunk_id_prepends_time(zarr_id):
    # fcst chunks are addressed as "0.<chunk_id>".
    assert zarr_id.format_chunk_id("3") == "0.3"


def test_generate_zarr_ids():
    times = [dt.datetime(2024, 7, 18, 0), dt.datetime(2024, 7, 18, 1)]
    ids = hrrr.generate_zarr_ids(
        times, level_type="sfc", variables=[("surface", "TMP")], model_type="fcst"
    )
    assert isinstance(ids, dict)
    key = ("surface", "TMP")
    assert key in ids
    # One ZarrId per requested time.
    assert len(ids[key]) == 2
    assert all(isinstance(z, hrrr.ZarrId) for z in ids[key])


class TestWindsFrame:
    def test_speed_is_from_components(self):
        import numpy as np
        import pandas as pd

        times = pd.date_range("2024-01-01", periods=2, freq="h")
        out = hrrr.winds_frame([3.0, 0.0], [4.0, 2.0], lon=-111.9, times=times)
        assert list(out.columns) == ["u", "v", "ws", "wd"]
        # rotation to earth-relative preserves the speed
        np.testing.assert_allclose(out["ws"], [5.0, 2.0])
        np.testing.assert_allclose(out["ws"], np.hypot(out["u"], out["v"]))


# --- chunk download/decode, hermetic -----------------------------------------
#
# HRRR zarr chunks are 150 x 150 blosc-compressed float16 grids (float32 for
# surface pressure); forecasts stack one such grid per lead time. The tests
# below build chunks like that in memory and serve them from a fake boto3 S3
# resource, so the decode/indexing path runs without AWS.


def _blosc(arr):
    """Compress an array the way the hrrrzarr chunks are stored."""
    import numcodecs

    return numcodecs.Blosc().encode(arr)


class _FakeS3:
    """Minimal stand-in for ``boto3.resource('s3')``: serves bytes by key."""

    def __init__(self, objects):
        self.objects = objects
        self.requested = []

    def Object(self, bucket, key):
        import io

        self.requested.append((bucket, key))
        body = io.BytesIO(self.objects[key])
        return type("Obj", (), {"get": lambda self: {"Body": body}})()


def _anl_id(var_name="UGRD", var_level="10m_above_ground", model_type="anl"):
    return hrrr.ZarrId(
        run_hour=dt.datetime(2024, 1, 1, 6),
        level_type="sfc",
        var_level=var_level,
        var_name=var_name,
        model_type=model_type,
    )


def test_format_chunk_id_analysis_is_unchanged():
    # Analyses have no time dimension, so the chunk id is used as-is
    assert _anl_id().format_chunk_id("4.5") == "4.5"


def test_generate_zarr_ids_fields():
    times = [dt.datetime(2024, 1, 1, h) for h in (0, 1, 2)]
    ids = hrrr.generate_zarr_ids(
        times,
        level_type="sfc",
        variables=[("UGRD", "10m_above_ground"), ("VGRD", "10m_above_ground")],
        model_type="anl",
    )
    assert list(ids) == [("UGRD", "10m_above_ground"), ("VGRD", "10m_above_ground")]
    ugrd = ids[("UGRD", "10m_above_ground")]
    assert [z.run_hour for z in ugrd] == times
    assert {(z.var_name, z.var_level, z.level_type, z.model_type) for z in ugrd} == {
        ("UGRD", "10m_above_ground", "sfc", "anl")
    }


class TestDecompressChunk:
    def test_analysis_is_150x150_float16(self):
        grid = np.arange(150 * 150, dtype="<f2").reshape(150, 150) / 64
        out = hrrr.decompress_chunk(_anl_id(), _blosc(grid))
        assert out.shape == (150, 150)
        assert out.dtype == np.dtype("<f2")
        np.testing.assert_array_equal(out, grid)

    def test_surface_pressure_is_float32(self):
        # float16 can't hold ~85000 Pa to the pascal; PRES is stored as float32
        grid = np.full((150, 150), 85123.0, dtype="<f4")
        zid = _anl_id(var_name="PRES", var_level="surface")
        out = hrrr.decompress_chunk(zid, _blosc(grid))
        assert out.dtype == np.dtype("<f4")
        assert out[0, 0] == 85123.0

    def test_forecast_stacks_lead_times(self):
        grid = np.stack([np.full((150, 150), h, dtype="<f2") for h in range(3)])
        out = hrrr.decompress_chunk(_anl_id(model_type="fcst"), _blosc(grid))
        assert out.shape == (3, 150, 150)
        np.testing.assert_array_equal(out[:, 7, 9], [0, 1, 2])


def _nearest_point(chunk_id="4.5", iy=7, ix=9):
    import xarray as xr

    return xr.Dataset(
        {
            "chunk_id": ((), chunk_id),
            "in_chunk_y": ((), iy),
            "in_chunk_x": ((), ix),
        }
    )


class TestGetValue:
    def test_analysis_value_at_in_chunk_y_x(self):
        grid = np.zeros((150, 150), dtype="<f2")
        grid[7, 9] = 3.5  # row = y, column = x
        grid[9, 7] = -1.0  # what a swapped x/y would read
        zid = _anl_id()
        key = hrrr.create_s3_chunk_url(zid, "4.5")
        s3 = _FakeS3({key: _blosc(grid)})

        value = hrrr.get_value(s3, zid, "4.5", _nearest_point())
        assert value == 3.5
        # Analysis chunks are addressed without the forecast "0." prefix
        assert s3.requested == [
            (
                "hrrrzarr",
                "sfc/20240101/20240101_06z_anl.zarr/10m_above_ground/UGRD/"
                "10m_above_ground/UGRD/4.5",
            )
        ]

    def test_forecast_returns_one_value_per_lead_time(self):
        grid = np.zeros((2, 150, 150), dtype="<f2")
        grid[:, 7, 9] = [1.5, 2.5]
        zid = _anl_id(model_type="fcst")
        key = hrrr.create_s3_chunk_url(zid, "4.5")
        assert key.endswith("/UGRD/0.4.5")
        s3 = _FakeS3({key: _blosc(grid)})
        np.testing.assert_array_equal(
            hrrr.get_value(s3, zid, "4.5", _nearest_point()), [1.5, 2.5]
        )

    def test_retrieve_object_reads_bucket_key(self):
        s3 = _FakeS3({"a/b": b"payload"})
        assert hrrr.retrieve_object(s3, "a/b") == b"payload"
        assert s3.requested == [("hrrrzarr", "a/b")]


def _chunk_index(lon, lat, spacing=3000.0):
    """A 5 x 5 chunk-index grid on the HRRR projection centred near (lon, lat).

    Each cell's chunk_id/in_chunk_* encode its (row, col) so the selected cell
    can be identified.
    """
    import cartopy.crs as ccrs
    import xarray as xr

    x0, y0 = hrrr.PROJECTION.transform_point(lon, lat, ccrs.PlateCarree())
    # Offset the grid so the point is not on a cell centre; x and y offsets
    # differ so swapping them would pick a different cell
    x = x0 + spacing * (np.arange(5) - 2) + 0.2 * spacing
    y = y0 + spacing * (np.arange(5) - 2) - 0.4 * spacing
    rows, cols = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    return xr.Dataset(
        {
            "chunk_id": (("y", "x"), np.vectorize(lambda r, c: f"{r}.{c}")(rows, cols)),
            "in_chunk_y": (("y", "x"), rows * 10),
            "in_chunk_x": (("y", "x"), cols * 10),
        },
        coords={"x": x, "y": y},
    )


def test_get_nearest_point_on_hrrr_projection():
    lon, lat = -111.85, 40.77
    index = _chunk_index(lon, lat)
    point = hrrr.get_nearest_point(lon, lat, index)
    # x offset +0.2 cell -> nearest column is the centre one (2); y offset
    # -0.4 cell -> nearest row is still 2. A point 1 cell north lands on row 3.
    assert str(point.chunk_id.values) == "2.2"
    north = hrrr.get_nearest_point(lon, lat + 3000.0 / 111e3, index)
    assert str(north.chunk_id.values) == "3.2"


def test_winds_end_to_end(monkeypatch):
    """Winds wires chunk index -> nearest point -> chunk download -> frame."""
    import xarray as xr

    # At HRRR's reference longitude (-97.5 deg) grid and earth axes coincide,
    # so the earth-relative winds equal the grid-relative ones
    lon, lat = -97.5, 38.5
    index = _chunk_index(lon, lat)
    times = [dt.datetime(2024, 1, 1, 0), dt.datetime(2024, 1, 1, 1)]
    u_grid, v_grid = [3.0, -2.0], [4.0, 0.5]

    # Centre cell: chunk "2.2", in-chunk (y, x) = (20, 20)
    objects = {}
    for var, values in (("UGRD", u_grid), ("VGRD", v_grid)):
        for t, value in zip(times, values):
            grid = np.full((150, 150), 99.0, dtype="<f2")
            grid[20, 20] = value
            zid = hrrr.ZarrId(t, "sfc", "10m_above_ground", var, "anl")
            objects[hrrr.create_s3_chunk_url(zid, "2.2")] = _blosc(grid)
    s3 = _FakeS3(objects)

    opened = []
    monkeypatch.setattr(hrrr.s3fs, "S3FileSystem", lambda anon: "fs")
    monkeypatch.setattr(hrrr.s3fs, "S3Map", lambda url, s3: url)
    monkeypatch.setattr(
        xr, "open_zarr", lambda store: opened.append(store) or index, raising=True
    )
    monkeypatch.setattr(hrrr.boto3, "resource", lambda **kwargs: s3)

    winds = hrrr.Winds(lat=lat, lon=lon, times=times)

    assert opened == ["s3://hrrrzarr/grid/HRRR_chunk_index.zarr"]
    assert len(s3.requested) == 4  # 2 variables x 2 times
    assert list(winds.data.index) == times
    np.testing.assert_allclose(winds.data["u"], u_grid, atol=1e-6)
    np.testing.assert_allclose(winds.data["v"], v_grid, atol=1e-6)
    np.testing.assert_allclose(winds.data["ws"], [5.0, np.hypot(2.0, 0.5)])
    # (3, 4) blows toward the NE, i.e. from 216.87 deg; (-2, 0.5) blows toward
    # the WNW, i.e. from 104.04 deg
    np.testing.assert_allclose(winds.data["wd"], [216.87, 104.04], atol=0.01)
