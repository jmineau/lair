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
            (0.0, 0.0, -1.0),    # from N -> blows toward S (v < 0)
            (90.0, -1.0, 0.0),   # from E -> blows toward W (u < 0)
            (180.0, 0.0, 1.0),   # from S -> blows toward N (v > 0)
            (270.0, 1.0, 0.0),   # from W -> blows toward E (u > 0)
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


def test_bin_polar_explicit_bin_edges():
    df = pd.DataFrame({"ws": [1, 2, 3, 4], "wd": [10, 100, 190, 280]})
    out = air.bin_polar(df, xbins=[0, 2, 4])
    # Compass sectors for the four cardinal-ish directions.
    assert out["wd_bin"].tolist() == ["N", "E", "S", "W"]
    # Speeds binned to the right edge of each explicit interval.
    assert out["x_bin"].tolist() == [2, 2, 4, 4]


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
