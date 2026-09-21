"""Tests for lair.hrrr.

Requires the `requests`/`formats`/`geo` extras (boto3/numcodecs/cartopy) —
skipped when not installed (run in the conda lair-dev env for full coverage).
The pure S3 URL/chunk construction is tested here; live HRRR fetches are not.
"""

import datetime as dt

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
