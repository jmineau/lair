"""Tests for lair.transects (metrics on transit x point matrices)."""

import numpy as np
import pandas as pd
import pytest

from lair import transects


@pytest.fixture
def matrix():
    """30 transits x 50 points: flat 2.0 ppm background with a transit-varying offset,
    a persistent +0.2 ppm source at point 10, an intermittent +0.5 ppm source at point 30
    (on for 6 of 30 transits), and one transit with no data."""
    rng = np.random.default_rng(0)
    obs = np.full((30, 50), 2.0) + rng.normal(0, 0.001, (30, 50))
    obs += np.linspace(0, 0.3, 30)[:, None]  # boundary-layer-like offset per transit
    obs[:, 10] += 0.2
    on = np.zeros(30, bool)
    on[::5] = True
    obs[on, 30] += 0.5
    obs[7] = np.nan
    return obs, on


def test_enhancement_removes_per_transit_offset(matrix):
    obs, _ = matrix
    enh = transects.enhancement(obs, baseline_q=5)
    assert np.isnan(enh[7]).all()
    # background points sit near zero regardless of the transit offset
    assert np.nanmax(np.abs(enh[:, 0])) < 0.01
    assert np.allclose(np.nanmedian(enh[:, 10]), 0.2, atol=0.01)


def test_detection_frequency_and_magnitude(matrix):
    obs, on = matrix
    enh = transects.enhancement(obs)
    f = transects.detection_frequency(enh, threshold=0.1, min_transits=5)
    assert f[10] == pytest.approx(1.0)
    assert f[30] == pytest.approx(on[np.arange(30) != 7].mean(), abs=0.01)
    assert f[0] == pytest.approx(0.0)
    m = transects.magnitude(enh, threshold=0.1)
    assert m[30] == pytest.approx(0.5, abs=0.01)
    assert np.isnan(transects.magnitude(enh[:3], threshold=0.1, min_transits=10)).all()


def test_transit_times_and_profile(matrix):
    obs, on = matrix
    enh = transects.enhancement(obs)
    base = pd.Timestamp("2020-01-01 12:00", tz="UTC").timestamp()
    # transit k starts at hour k (mod 24) UTC; points spaced 10 s
    time = base + np.arange(30)[:, None] * 3600 + np.arange(50)[None, :] * 10.0
    time[7] = np.nan
    tt = transects.transit_times(time)
    assert tt[0] == pd.Timestamp("2020-01-01 12:04:05")  # median of 50 points, 10 s apart
    assert pd.isna(tt[7])
    bins, freq, mag = transects.profile(enh, tt, by="hour", threshold=0.1, min_transits=1)
    assert bins.shape == (24,) and freq.shape == (24, 50)
    # the persistent source is detected in every hour bin that has data
    assert np.nanmin(freq[:, 10]) == pytest.approx(1.0)


def test_along_route_distance():
    pytest.importorskip("pyproj")
    d = transects.along_route_distance(np.array([-111.9, -111.9]), np.array([40.7, 40.709]))
    assert d[0] == 0.0 and d[1] == pytest.approx(1.0, abs=0.01)


def test_merge_and_pool_routes():
    pytest.importorskip("scipy")
    # route A: 5 points on a line; route B shares A's points 2-3 (within 5 m) and adds 2 new ones
    a = np.c_[np.arange(5) * 100.0, np.zeros(5)]
    b = np.array([[201.0, 2.0], [303.0, -1.0], [400.0, 300.0], [400.0, 400.0]])
    net, idx = transects.merge_route_points([a, b], tol=10.0)
    assert len(net) == 7
    assert list(idx[0]) == [0, 1, 2, 3, 4]
    assert list(idx[1]) == [2, 3, 5, 6]
    ma = np.arange(10, dtype=float).reshape(2, 5)          # 2 transits on A
    mb = np.array([[1.0, 2.0, 3.0, 4.0]])                  # 1 transit on B
    pooled = transects.pool_routes([ma, mb], idx, len(net))
    assert pooled.shape == (3, 7)
    assert pooled[0, 2] == 2.0 and pooled[2, 2] == 1.0 and pooled[2, 5] == 3.0
    assert np.isnan(pooled[0, 5]) and np.isnan(pooled[2, 0])
    # shared network point 2 now has three transits: two from A, one from B
    assert np.isfinite(pooled[:, 2]).sum() == 3
