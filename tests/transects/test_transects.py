"""Tests for lair.transects (metrics on transit x point matrices)."""

import numpy as np
import pandas as pd
import pytest

from lair import transects


@pytest.fixture
def matrix():
    """
    30 transits x 50 points with two sources and a missing transit.

    A flat 2.0 ppm background with a transit-varying offset,
    a persistent +0.2 ppm source at point 10, an intermittent +0.5 ppm source at point 30
    (on for 6 of 30 transits), and one transit with no data.
    """
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
    assert tt[0] == pd.Timestamp(
        "2020-01-01 12:04:05"
    )  # median of 50 points, 10 s apart
    assert pd.isna(tt[7])
    bins, freq, mag = transects.profile(
        enh, tt, by="hour", threshold=0.1, min_transits=1
    )
    assert bins.shape == (24,) and freq.shape == (24, 50)
    # the persistent source is detected in every hour bin that has data
    assert np.nanmin(freq[:, 10]) == pytest.approx(1.0)


class TestProfileTimeZone:
    """A time-zone-aware time is the moment it names, shifted once by tz_offset_hours."""

    # three transits at 18:00 UTC, which is 11:00 in MST (UTC-7)
    utc = pd.DatetimeIndex(["2020-01-01 18:00", "2020-01-02 18:00", "2020-01-03 18:00"])

    @staticmethod
    def _hours_with_data(times, tz_offset_hours):
        enh = np.ones((3, 1))
        bins, freq, _ = transects.profile(
            enh, times, min_transits=1, tz_offset_hours=tz_offset_hours
        )
        return bins[~np.isnan(freq[:, 0])].tolist()

    @pytest.mark.parametrize(
        "times",
        [
            utc,
            utc.tz_localize("UTC"),
            utc.tz_localize("UTC").tz_convert("America/Denver"),
            utc.tz_localize("UTC").tz_convert("Asia/Kolkata"),
        ],
        ids=["naive UTC", "UTC", "Denver", "Kolkata"],
    )
    def test_the_offset_shifts_the_utc_hour_once(self, times):
        assert self._hours_with_data(times, 0) == [18]
        assert self._hours_with_data(times, -7) == [11]

    def test_other_bins_use_the_utc_moment_too(self):
        # 2020-01-01 10:00 UTC is 03:00 on Wednesday 1 January in MST; shifted
        # twice it would be Tuesday 31 December
        local = pd.DatetimeIndex(["2020-01-01 10:00"]).tz_localize("UTC")
        for by, expected in [("weekday", 2), ("month", 1)]:
            bins, freq, _ = transects.profile(
                np.ones((1, 1)),
                local.tz_convert("America/Denver"),
                by=by,
                min_transits=1,
                tz_offset_hours=-7,
            )
            assert bins[~np.isnan(freq[:, 0])].tolist() == [expected]


def test_along_route_distance():
    pytest.importorskip("pyproj")
    d = transects.along_route_distance(
        np.array([-111.9, -111.9]), np.array([40.7, 40.709])
    )
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
    ma = np.arange(10, dtype=float).reshape(2, 5)  # 2 transits on A
    mb = np.array([[1.0, 2.0, 3.0, 4.0]])  # 1 transit on B
    pooled = transects.pool_routes([ma, mb], idx, len(net))
    assert pooled.shape == (3, 7)
    assert pooled[0, 2] == 2.0 and pooled[2, 2] == 1.0 and pooled[2, 5] == 3.0
    assert np.isnan(pooled[0, 5]) and np.isnan(pooled[2, 0])
    # shared network point 2 now has three transits: two from A, one from B
    assert np.isfinite(pooled[:, 2]).sum() == 3


def test_robust_z_is_transit_relative(matrix):
    obs, on = matrix
    enh = transects.enhancement(obs)
    # add a uniform +1 ppm to one transit: absolute enhancement is unchanged (baseline
    # removes it), and so is the z-score, which is the property we want
    z = transects.robust_z(enh, min_points=10)
    assert np.isnan(z[7]).all()
    assert np.nanmedian(z[:, 0]) == pytest.approx(0.0, abs=0.5)
    assert np.nanmin(z[:, 10]) > 3.0  # persistent source stands out in every transit
    f = transects.detection_frequency(z, threshold=3.0, min_transits=5)
    assert f[10] == pytest.approx(1.0) and f[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# builder: lag_positions / snap_to_route / split_transits / transect_matrix
# ---------------------------------------------------------------------------


@pytest.fixture
def track():
    """
    A 5-km straight route sampled every second at 10 m/s, with a dwell and a gap.

    It goes out (s increasing), a
    120-s dwell at the far end, back, a 30-min gap, then out again but only 3 km.
    Obs = 2.0 ppm with a +0.3 ppm source at s = 2,000 m (sampled with a 10-s lag, i.e.
    the value shows up 100 m further along in the direction of travel).
    """
    v, L = 10.0, 5000.0
    out = np.arange(0, L + 1, v)
    s = np.r_[out, np.full(120, L), out[::-1]]
    t = np.arange(len(s), dtype=float)
    # second run after a gap, partial
    out2 = np.arange(0, 3000 + 1, v)
    s = np.r_[s, out2]
    t = np.r_[t, t[-1] + 1800 + np.arange(len(out2))]
    xy = np.c_[s, np.zeros_like(s)]
    lag = 10.0
    obs = np.full(len(s), 2.0)
    # source is at s = 2000; a lagged reading logs it at the position v*lag later
    s_true = np.interp(t - lag, t, s)  # where the air was taken in
    obs[np.abs(s_true - 2000.0) <= 25.0] += 0.3
    return t, xy, obs, lag


def test_lag_positions_moves_samples_back_along_the_track(track):
    t, xy, obs, lag = track
    lagged = transects.lag_positions(t, xy, lag, max_gap_s=600)
    # outbound: 100 m behind the logged position
    assert lagged[50, 0] == pytest.approx(xy[50, 0] - 100.0)
    # the first sample of a run is clamped to the run start, not interpolated across the gap
    i_run2 = len(t) - 301
    assert lagged[i_run2, 0] == pytest.approx(xy[i_run2, 0])
    assert lagged[i_run2 + 5, 0] == pytest.approx(
        0.0
    )  # 5 s in, lag 10 s -> still clamped


def test_split_transits_finds_reversal_and_gap(track):
    pytest.importorskip("scipy")
    t, xy, obs, lag = track
    transit, table = transects.split_transits(
        t, xy[:, 0], max_gap_s=600, reversal_m=500, min_span_m=1000
    )
    assert list(table["direction"]) == [1, -1, 1]
    assert table.loc[0, "s_max"] == pytest.approx(5000.0)
    assert table.loc[2, "s_max"] == pytest.approx(3000.0)
    # every kept sample belongs to exactly one transit; the dwell is split between 0 and 1
    assert (transit >= 0).all()
    assert (transit[:501] == 0).all()
    assert (transit[-301:] == 2).all()


def test_split_transits_ignores_jitter_and_short_shunts():
    pytest.importorskip("scipy")
    t = np.arange(600.0)
    s = t * 10.0
    s[100:110] -= 30.0  # GPS jitter
    s[300:340] = (
        s[300] - np.r_[np.arange(20), np.arange(20)[::-1]] * 10.0
    )  # 200-m shunt
    transit, table = transects.split_transits(t, s, reversal_m=500)
    assert len(table) == 1 and table.loc[0, "direction"] == 1
    assert (transit == 0).all()


def test_split_transits_drops_off_route_and_short():
    pytest.importorskip("scipy")
    t = np.arange(200.0)
    s = np.full(200, np.nan)
    s[:50] = np.arange(50) * 10.0  # 500 m only
    transit, table = transects.split_transits(t, s, min_span_m=1000)
    assert table.empty and (transit == -1).all()


def test_transect_matrix_with_lag_and_dwell_trim(track):
    pytest.importorskip("scipy")
    t, xy, obs, lag = track
    route_xy = np.c_[np.arange(0, 5001, 50.0), np.zeros(101)]
    lagged = transects.lag_positions(t, xy, lag)
    point, dist = transects.snap_to_route(lagged, route_xy, max_dist=60)
    assert (point >= 0).all() and dist.max() <= 25.0
    transit, table = transects.split_transits(t, lagged[:, 0], max_gap_s=600)
    m_obs, m_t, m_n = transects.transect_matrix(
        transit, point, obs, t, len(table), len(route_xy), max_dwell_s=30
    )
    assert m_obs.shape == (3, 101)
    # with the lag applied the source lands on point 40 (s = 2000) in both directions
    src = np.nanargmax(m_obs, axis=1)
    assert list(src) == [40, 40, 40]
    assert m_obs[0, 40] == pytest.approx(2.3, abs=0.05)
    assert m_obs[0, 10] == pytest.approx(2.0)
    # the partial third transit logs to 3 km, i.e. air taken in up to 2.9 km (point 58)
    assert np.isnan(m_obs[2, 59:]).all() and np.isfinite(m_obs[2, :59]).all()
    # the terminus dwell is trimmed to max_dwell_s: the end point holds ~30 s, not 60
    assert m_n[0, 100] <= 32 and m_n[1, 100] <= 32
    # time is the mean sample time of the cell
    assert m_t[0, 0] == pytest.approx(t[point == 0][transit[point == 0] == 0].mean())


def _build(t, xy, obs, lag):
    """Run the whole builder; returns (lagged, transit, table, matrix obs)."""
    route_xy = np.c_[np.arange(0, 5001, 50.0), np.zeros(101)]
    lagged = transects.lag_positions(t, xy, lag)
    point, _ = transects.snap_to_route(lagged, route_xy, max_dist=60)
    transit, table = transects.split_transits(t, lagged[:, 0], max_gap_s=600)
    m_obs, m_t, _ = transects.transect_matrix(
        transit, point, obs, t, len(table), len(route_xy), max_dwell_s=30
    )
    return lagged, transit, table, m_obs, m_t


def test_builder_accepts_tz_aware_times(track):
    pytest.importorskip("scipy")
    t, xy, obs, lag = track
    t0 = pd.Timestamp("2024-01-01 06:00", tz="America/Denver")
    when = t0 + pd.to_timedelta(t, unit="s")
    expected = _build(t, xy, obs, lag)
    for time in (when, pd.Series(when)):
        lagged, transit, table, m_obs, m_t = _build(time, xy, obs, lag)
        np.testing.assert_allclose(lagged, expected[0])
        np.testing.assert_array_equal(transit, expected[1])
        np.testing.assert_allclose(m_obs, expected[3])
        # times come back as POSIX seconds (UTC), not local wall-clock
        assert table.loc[0, "t_start"] == pytest.approx(t0.timestamp())
        np.testing.assert_allclose(m_t, expected[4] + t0.timestamp())


@pytest.mark.parametrize("kind", ["seconds", "datetime64"])
def test_builder_skips_samples_without_time(track, kind):
    # A sample without a time can't be placed in time: it gets no lagged position, no
    # transit and no cell, and the other samples come out as if it weren't there.
    pytest.importorskip("scipy")
    t, xy, obs, lag = track
    missing = np.zeros(len(t), bool)
    missing[[0, 200, 201, 700, len(t) - 1]] = True
    expected = _build(t[~missing], xy[~missing], obs[~missing], lag)
    if kind == "seconds":
        time = np.where(missing, np.nan, t)
    else:
        time = (t * 1e9).astype("datetime64[ns]")
        time[missing] = np.datetime64("NaT", "ns")
    lagged, transit, table, m_obs, m_t = _build(time, xy, obs, lag)
    assert np.isnan(lagged[missing]).all()
    np.testing.assert_allclose(lagged[~missing], expected[0])
    assert (transit[missing] == -1).all()
    np.testing.assert_array_equal(transit[~missing], expected[1])
    pd.testing.assert_frame_equal(table, expected[2])
    np.testing.assert_allclose(m_obs, expected[3])
    np.testing.assert_allclose(m_t, expected[4])


def test_lag_positions_tz_aware_and_nat(track):
    # No scipy needed, so this also runs in the lean env.
    t, xy, obs, lag = track
    expected = transects.lag_positions(t, xy, lag)
    when = pd.Series(pd.Timestamp("2024-07-01", tz="UTC") + pd.to_timedelta(t, "s"))
    when = when.dt.tz_convert("America/Denver")
    np.testing.assert_allclose(transects.lag_positions(when, xy, lag), expected)
    when[50] = pd.NaT
    lagged = transects.lag_positions(when, xy, lag)
    assert np.isnan(lagged[50]).all()
    np.testing.assert_allclose(lagged[51:], expected[51:])


def test_snap_to_route_nan_position():
    pytest.importorskip("scipy")
    route_xy = np.c_[np.arange(0, 201, 50.0), np.zeros(5)]
    xy = np.array([[1.0, 0.0], [np.nan, np.nan], [49.0, 0.0], [500.0, 0.0]])
    idx, d = transects.snap_to_route(xy, route_xy, max_dist=10)
    assert idx.tolist() == [0, -1, 1, -1]
    assert np.isnan(d[1]) and d[3] == pytest.approx(300.0)


def test_magnitude_without_threshold_uses_every_transit_with_data():
    # point 0: 1, 3 and 8 (mean 4, median 3); point 1: 2 and 6 (the NaN is no transit)
    enh = np.array([[1.0, 2.0], [3.0, np.nan], [8.0, 6.0]])
    np.testing.assert_allclose(
        transects.magnitude(enh, stat="mean", min_transits=2), [4, 4]
    )
    np.testing.assert_allclose(
        transects.magnitude(enh, stat="median", min_transits=2), [3, 4]
    )
    # a point needs min_transits transits with data: point 1 has only two
    out = transects.magnitude(enh, stat="mean", min_transits=3)
    assert out[0] == pytest.approx(4.0) and np.isnan(out[1])


@pytest.mark.parametrize(
    "stat, expected",
    [("median", "12:00:10"), ("min", "12:00:00"), ("max", "12:00:20")],
)
def test_transit_times_from_datetime64(stat, expected):
    day = np.datetime64("2024-01-01")
    row = day + np.array(
        [12 * 3600, 12 * 3600 + 10, 12 * 3600 + 20], dtype="timedelta64[s]"
    )
    time = np.vstack([row, np.full(3, np.datetime64("NaT", "s"), dtype=row.dtype)])
    tt = transects.transit_times(time, stat=stat)
    assert tt[0] == pd.Timestamp(f"2024-01-01 {expected}")
    assert pd.isna(tt[1])


def test_lag_positions_without_any_time_is_all_nan():
    time = np.full(4, np.datetime64("NaT", "s"), dtype="datetime64[s]")
    xy = np.arange(8.0).reshape(4, 2)
    out = transects.lag_positions(time, xy, lag_s=10.0)
    assert out.shape == (4, 2) and np.isnan(out).all()


def test_snap_to_route_with_no_position_is_off_route():
    pytest.importorskip("scipy")
    route_xy = np.c_[np.arange(0, 201, 50.0), np.zeros(5)]
    idx, d = transects.snap_to_route(np.full((3, 2), np.nan), route_xy, max_dist=10)
    assert idx.tolist() == [-1, -1, -1]
    assert np.isnan(d).all()


@pytest.mark.parametrize("which", ["s", "time"])
def test_split_transits_with_nothing_to_place_is_empty(which):
    pytest.importorskip("scipy")
    t = np.arange(100.0)
    s = t * 20.0
    if which == "s":
        s[:] = np.nan  # off the route throughout
    else:
        t[:] = np.nan  # no sample has a time
    transit, table = transects.split_transits(t, s)
    assert (transit == -1).all()
    assert table.empty and list(table.columns) == transects.TRANSIT_COLUMNS


def test_split_transits_skips_a_run_of_one_sample():
    pytest.importorskip("scipy")
    # 200 s along the route at 10 m/s (2 km), then an hour later one lone sample
    t = np.r_[np.arange(200.0), 5000.0]
    s = np.r_[np.arange(200.0) * 10.0, 500.0]
    transit, table = transects.split_transits(t, s, max_gap_s=600)
    assert transit[-1] == -1
    assert (transit[:-1] == 0).all()
    assert len(table) == 1 and table.loc[0, "n"] == 200


def test_transect_matrix_counts_every_sample_without_max_dwell_s():
    # Transit 0 sits at point 1 for three samples (obs 1, 3, 5 at t = 0, 10, 100 s)
    transit = np.zeros(3, dtype=int)
    point = np.ones(3, dtype=int)
    obs = np.array([1.0, 3.0, 5.0])
    time = np.array([0.0, 10.0, 100.0])
    m_obs, m_t, m_n = transects.transect_matrix(transit, point, obs, time, 1, 2)
    assert m_obs[0, 1] == pytest.approx(3.0)  # (1 + 3 + 5) / 3
    assert m_t[0, 1] == pytest.approx(110.0 / 3)
    assert m_n[0, 1] == 3
    assert np.isnan(m_obs[0, 0])
    # max_dwell_s=20 drops the sample 100 s after the first one
    m_obs, _, m_n = transects.transect_matrix(
        transit, point, obs, time, 1, 2, max_dwell_s=20
    )
    assert m_obs[0, 1] == pytest.approx(2.0) and m_n[0, 1] == 2
