"""Tests for lair.air (wind math + polar binning).

Meteorological wind-direction convention: direction is where the wind blows
*from*, in degrees clockwise from North. So a wind FROM the north (0 deg) blows
toward the south -> v component is negative.
"""

import numpy as np
import pandas as pd
import pytest

from lair import air


class TestWindComponents:
    @pytest.mark.parametrize(
        "direction, exp_u, exp_v",
        [
            (0.0, 0.0, -1.0),  # from N -> blows toward S (v < 0)
            (90.0, -1.0, 0.0),  # from E -> blows toward W (u < 0)
            (180.0, 0.0, 1.0),  # from S -> blows toward N (v > 0)
            (270.0, 1.0, 0.0),  # from W -> blows toward E (u > 0)
        ],
    )
    def test_unit_speed_directions(self, direction, exp_u, exp_v):
        u, v = air.wind_components(1.0, direction)
        assert u == pytest.approx(exp_u, abs=1e-9)
        assert v == pytest.approx(exp_v, abs=1e-9)

    def test_round_trip_with_wind_direction(self):
        # components -> direction should recover the original bearing.
        for direction in (0.0, 45.0, 130.0, 225.0, 359.0):
            u, v = air.wind_components(5.0, direction)
            assert air.wind_direction(u, v) == pytest.approx(direction, abs=1e-6)


class TestWindDirection:
    def test_known_vectors(self):
        # Wind blowing toward the east (u>0, v=0) comes FROM the west -> 270.
        assert air.wind_direction(1.0, 0.0) == pytest.approx(270.0)
        # Toward the north (v>0) comes FROM the south -> 180.
        assert air.wind_direction(0.0, 1.0) == pytest.approx(180.0)

    def test_zero_vector_has_no_direction(self):
        assert np.isnan(air.wind_direction(0.0, 0.0))
        wd = air.wind_direction(np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        assert np.isnan(wd[0]) and wd[1] == pytest.approx(270.0)

    def test_pandas_labels_kept(self):
        u = pd.Series([0.0, 1.0], index=["a", "b"])
        v = pd.Series([0.0, 0.0], index=["a", "b"])
        wd = air.wind_direction(u, v)
        assert isinstance(wd, pd.Series) and list(wd.index) == ["a", "b"]
        assert np.isnan(wd["a"]) and wd["b"] == pytest.approx(270.0)

    def test_vector_mean_across_north(self):
        # 350 and 10 deg average to north, not to the 180 a mean of degrees gives
        u, v = air.wind_components(np.array([5.0, 5.0]), np.array([350.0, 10.0]))
        assert air.wind_direction(u.mean(), v.mean()) % 360 == pytest.approx(0.0)

    def test_range_is_0_to_360(self):
        rng = np.random.default_rng(0)
        u = rng.normal(size=100)
        v = rng.normal(size=100)
        wd = air.wind_direction(u, v)
        assert np.all((wd >= 0) & (wd < 360))


class TestRotateWinds:
    def test_no_rotation_at_reference_longitude(self):
        # At the HRRR reference longitude (-97.5) the rotation angle is zero.
        u_out, v_out = air.rotate_winds(3.0, 4.0, lon=-97.5)
        assert u_out == pytest.approx(3.0)
        assert v_out == pytest.approx(4.0)

    def test_preserves_wind_speed(self):
        # Rotation is orthonormal -> magnitude is unchanged.
        u, v = 3.0, 4.0
        u_out, v_out = air.rotate_winds(u, v, lon=-110.0)
        assert np.hypot(u_out, v_out) == pytest.approx(np.hypot(u, v))

    @pytest.mark.parametrize("lon", [-112.0, 248.0, -472.0])
    def test_longitude_convention_does_not_matter(self, lon):
        # 248 E and -472 are the same meridian as 112 W; HRRR readers accept
        # either convention, so the rotation must not depend on it.
        u_ref, v_ref = air.rotate_winds(0.0, 10.0, lon=-112.0)
        u_out, v_out = air.rotate_winds(0.0, 10.0, lon=lon)
        assert u_out == pytest.approx(u_ref)
        assert v_out == pytest.approx(v_ref)
        assert v_out > 0  # a southerly grid wind stays southerly at 112 W

    def test_longitude_array(self):
        u_out, v_out = air.rotate_winds(
            np.zeros(2), np.full(2, 10.0), lon=np.array([-112.0, 248.0])
        )
        assert u_out[0] == pytest.approx(u_out[1])
        assert v_out[0] == pytest.approx(v_out[1])


def test_bin_polar_adds_expected_columns():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "ws": rng.uniform(0, 10, size=200),
            "wd": rng.uniform(0, 360, size=200),
        }
    )
    out = air.bin_polar(df, x="ws", wd="wd", xbins=10)
    for col in ("wd_bin", "radian_bin", "x_bin"):
        assert col in out.columns
    # Direction bins map onto the 16 compass sectors (N..NNW), no 'N2' leakage.
    assert "N2" not in set(out["wd_bin"].dropna().unique())


def test_bin_polar_does_not_mutate_caller():
    df = pd.DataFrame({"ws": [1.0, 3.0], "wd": [10.0, 100.0]})
    before = df.copy()
    out = air.bin_polar(df, xbins=[0, 2, 4])
    assert "wd_bin" in out.columns
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.filterwarnings("error::FutureWarning")
def test_bin_polar_wd_bin_is_16_ordered_sectors():
    # 350 deg falls in the last cut interval and wraps round to N
    df = pd.DataFrame({"ws": [1.0, 2.0, 3.0], "wd": [5.0, 350.0, 200.0]})
    out = air.bin_polar(df, xbins=[0, 2, 4])
    assert out["wd_bin"].tolist() == ["N", "N", "SSW"]
    cats = out["wd_bin"].cat
    assert cats.ordered
    assert list(cats.categories) == [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
    ]  # fmt: skip


def test_sector_radians_match_radian_bin_exactly():
    # The polar plots reindex their grid to sector_radians(), which only lines
    # up if radian_bin takes exactly these float values
    sectors = air.sector_radians()
    np.testing.assert_allclose(sectors, np.deg2rad(np.arange(16) * 22.5))
    df = pd.DataFrame({"ws": np.ones(16), "wd": np.arange(16) * 22.5 + 3.0})
    out = air.bin_polar(df, xbins=[0, 2])
    assert out["radian_bin"].tolist() == sectors.tolist()


def test_bin_polar_explicit_bin_edges():
    df = pd.DataFrame({"ws": [1, 2, 3, 4], "wd": [10, 100, 190, 280]})
    out = air.bin_polar(df, xbins=[0, 2, 4])
    # Compass sectors for the four cardinal-ish directions.
    assert out["wd_bin"].tolist() == ["N", "E", "S", "W"]
    # Speeds binned to the right edge of each explicit interval.
    assert out["x_bin"].tolist() == [2, 2, 4, 4]


def test_bin_polar_int_xbins_is_number_of_edges():
    # An int xbins is the number of evenly spaced edges from min to max, so it
    # gives xbins - 1 speed bins, labelled by their right edges.
    df = pd.DataFrame({"ws": [0.0, 1.0, 2.0, 3.0, 4.0], "wd": [10.0] * 5})
    out = air.bin_polar(df, xbins=5)
    assert list(out["x_bin"].cat.categories) == [1.0, 2.0, 3.0, 4.0]
    assert out["x_bin"].tolist() == [1.0, 1.0, 2.0, 3.0, 4.0]


def test_bin_polar_nan_direction():
    # A missing direction leaves that row unbinned instead of raising.
    df = pd.DataFrame({"ws": [1.0, 2.0, 3.0], "wd": [10.0, np.nan, 280.0]})
    out = air.bin_polar(df, xbins=[0, 2, 4])
    assert out["wd_bin"].iloc[0] == "N"
    assert pd.isna(out["wd_bin"].iloc[1])
    assert out["radian_bin"].iloc[0] == pytest.approx(0.0)
    assert np.isnan(out["radian_bin"].iloc[1])
    assert out["radian_bin"].iloc[2] == pytest.approx(np.deg2rad(270))


def test_bin_polar_nan_speed_first():
    # Bin edges come from the non-missing speeds, even when the first is NaN.
    df = pd.DataFrame({"ws": [np.nan, 0.0, 2.0, 4.0], "wd": [10.0] * 4})
    out = air.bin_polar(df, xbins=3)
    assert list(out["x_bin"].cat.categories) == [2.0, 4.0]
    assert pd.isna(out["x_bin"].iloc[0])
    assert out["x_bin"].iloc[1:].tolist() == [2.0, 2.0, 4.0]


def test_bin_polar_invalid_xbins_raises():
    df = pd.DataFrame({"ws": [1.0], "wd": [10.0]})
    with pytest.raises(ValueError):
        air.bin_polar(df, xbins="not-valid")


def test_circularize_radial_data_wraps_around():
    # 3 theta rows x 2 radius columns -> meshgrid gains a wraparound row.
    agg = pd.DataFrame(
        np.arange(6).reshape(3, 2), index=[0.0, 1.0, 2.0], columns=[10, 20]
    )
    theta, r, c = air.circularize_radial_data(agg)
    assert theta.shape == (4, 2)
    assert r.shape == (4, 2)
    assert c.shape == (4, 2)
    # The appended row closes the circle by repeating the first data row.
    assert np.array_equal(c[-1], c[0])
