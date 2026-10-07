"""
Metrics on mobile-platform transect matrices.

A *transect matrix* is a 2-D array ``obs[transit, point]`` of a species measured on
repeated transits of a fixed route, resampled onto fixed along-route points (NaN where a
transit has no data at a point). Persistence questions are answered per point:

- :func:`enhancement`        — obs minus each transit's own low percentile (removes the
                                boundary-layer / background cycle transit by transit)
- :func:`detection_frequency` — fraction of transits enhanced above a threshold
- :func:`magnitude`          — mean or median enhancement, over detected transits or all
- :func:`transit_times`      — one representative timestamp per transit
- :func:`profile`            — detection frequency and magnitude binned by hour / weekday / month
- :func:`along_route_distance` — cumulative geodesic distance of the points [km]

Building the matrix from a georeferenced time series (the *transect builder*, after
Mitchell et al. 2018's algorithm):

- :func:`lag_positions`  — put each sample where the sampled air was taken in (inlet lag)
- :func:`snap_to_route`  — nearest fixed route point of each sample
- :func:`split_transits` — cut the along-route coordinate into one-way transits at the
                            reversals of travel and at time gaps
- :func:`transect_matrix` — average the samples onto ``[transit, point]``

Everything is plain numpy; the platform-specific file formats live in the calling package
(e.g. ``slv.measurements.mobile`` for TRAX). Times are POSIX seconds (float) or
datetime-like (datetime64, or pandas times with or without a time zone; naive times are
taken as UTC). The builder skips samples without a time (NaN / NaT).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "enhancement",
    "robust_z",
    "detection_frequency",
    "magnitude",
    "transit_times",
    "profile",
    "along_route_distance",
    "merge_route_points",
    "pool_routes",
    "lag_positions",
    "snap_to_route",
    "split_transits",
    "transect_matrix",
]


def enhancement(obs: np.ndarray, baseline_q: float = 5.0) -> np.ndarray:
    """
    Enhancement above each transit's own ``baseline_q``-th percentile along the route.

    Transits with no finite values return NaN throughout.
    """
    obs = np.asarray(obs, dtype=float)
    ok = np.isfinite(obs).any(axis=1)
    base = np.full((obs.shape[0], 1), np.nan)
    base[ok, 0] = np.nanpercentile(obs[ok], baseline_q, axis=1)
    return obs - base


def robust_z(enh: np.ndarray, min_points: int = 20) -> np.ndarray:
    """
    Per-transit robust z-score of the enhancement.

    The score is ``(enh - median) / (1.4826 * MAD)``, with the median and MAD taken
    along each transit's route points.

    A point is then judged against the rest of *its own transit*, so a night with the whole
    route elevated does not read as detections everywhere. Transits with fewer than
    ``min_points`` finite points, or zero MAD, return NaN.
    """
    enh = np.asarray(enh, dtype=float)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(enh, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(enh - med), axis=1, keepdims=True) * 1.4826
    n = np.isfinite(enh).sum(axis=1, keepdims=True)
    ok = (n >= min_points) & (mad > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (enh - med) / mad
    return np.where(ok, z, np.nan)


def detection_frequency(
    enh: np.ndarray, threshold: float, min_transits: int = 10
) -> np.ndarray:
    """
    Fraction of transits (with data at the point) whose enhancement exceeds ``threshold``.

    Points sampled by fewer than ``min_transits`` transits return NaN.
    """
    enh = np.asarray(enh, dtype=float)
    n = np.isfinite(enh).sum(axis=0)
    det = (enh > threshold).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = det / n
    f = np.where(n >= min_transits, f, np.nan)
    return f


def magnitude(
    enh: np.ndarray,
    threshold: float | None = None,
    stat: str = "median",
    min_transits: int = 10,
) -> np.ndarray:
    """
    Per-point enhancement magnitude.

    It is ``stat`` ("median" | "mean") over detected transits (``enh > threshold``) or
    over all transits with data (``threshold=None``).

    Points sampled by fewer than ``min_transits`` transits (with data, before thresholding)
    return NaN; so do points with no detection at all.
    """
    enh = np.asarray(enh, dtype=float)
    n = np.isfinite(enh).sum(axis=0)
    if threshold is not None:
        enh = np.where(enh > threshold, enh, np.nan)
    func = {"median": np.nanmedian, "mean": np.nanmean}[stat]
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = func(enh, axis=0)
    return np.where(n >= min_transits, m, np.nan)


def transit_times(time: np.ndarray, stat: str = "median") -> pd.DatetimeIndex:
    """
    One timestamp per transit from a ``time[transit, point]`` matrix.

    The matrix holds POSIX seconds or datetime64; transits with no data get NaT.
    """
    t = np.asarray(time)
    if np.issubdtype(t.dtype, np.datetime64):
        t = t.astype("datetime64[s]").astype(float)
        t[np.isnat(np.asarray(time))] = np.nan
    func = {"median": np.nanmedian, "min": np.nanmin, "max": np.nanmax}[stat]
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rep = func(t.astype(float), axis=1)
    return pd.to_datetime(rep, unit="s")


def profile(
    enh: np.ndarray,
    times: pd.DatetimeIndex,
    by: str = "hour",
    threshold: float = 0.0,
    min_transits: int = 10,
    tz_offset_hours: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the detection frequency and median magnitude per point, binned by time.

    ``by`` is the "hour", "weekday" or "month" of the transit time. Returns ``(bins, freq[bin, point], mag[bin, point])``. ``tz_offset_hours`` shifts the
    (UTC) transit times to local time before binning (e.g. -7 for MST).
    """
    t = times + pd.Timedelta(hours=tz_offset_hours)
    key = {"hour": t.hour, "weekday": t.weekday, "month": t.month}[by]
    bins = {"hour": np.arange(24), "weekday": np.arange(7), "month": np.arange(1, 13)}[
        by
    ]
    key = np.asarray(key, dtype=float)
    freq = np.full((len(bins), enh.shape[1]), np.nan)
    mag = np.full_like(freq, np.nan)
    for i, b in enumerate(bins):
        sel = key == b
        if sel.sum() == 0:
            continue
        freq[i] = detection_frequency(enh[sel], threshold, min_transits)
        mag[i] = magnitude(enh[sel], threshold, "median", min_transits)
    return bins, freq, mag


def along_route_distance(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Cumulative geodesic distance [km] along the ordered route points (WGS84)."""
    from pyproj import Geod

    seg = Geod(ellps="WGS84").line_lengths(np.asarray(lon), np.asarray(lat))
    return np.concatenate([[0.0], np.cumsum(seg)]) / 1000.0


def merge_route_points(
    routes_xy: list[np.ndarray], tol: float
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Merge the fixed points of several routes into one network point set.

    ``routes_xy`` are ``(n_i, 2)`` arrays of projected coordinates (metres). Points of the
    first route are kept; a point of a later route is added only if it is farther than
    ``tol`` from every point already in the set, otherwise it is snapped to the nearest
    existing one. Returns ``(network_xy, index)`` where ``index[i][k]`` is the network
    point of route ``i``'s point ``k``. Shared track thus maps to shared network points.
    """
    from scipy.spatial import cKDTree

    network = np.asarray(routes_xy[0], dtype=float).copy()
    index = [np.arange(len(network))]
    for xy in routes_xy[1:]:
        xy = np.asarray(xy, dtype=float)
        d, near = cKDTree(network).query(xy)
        new = d > tol
        idx = near.copy()
        idx[new] = len(network) + np.arange(new.sum())
        network = np.vstack([network, xy[new]])
        index.append(idx)
    return network, index


def pool_routes(
    matrices: list[np.ndarray], index: list[np.ndarray], n_network: int
) -> np.ndarray:
    """
    Stack per-route ``[transit, point]`` matrices onto the network points.

    Returns a ``[sum of transits, n_network]`` matrix, NaN where a route has no point
    (or no data) at a network point. Where two route points of one route snap to the
    same network point, the last one wins.
    """
    rows = sum(m.shape[0] for m in matrices)
    out = np.full((rows, n_network), np.nan)
    r0 = 0
    for m, idx in zip(matrices, index, strict=True):
        m = np.asarray(m, dtype=float)
        out[r0 : r0 + m.shape[0], idx] = m
        r0 += m.shape[0]
    return out


# ---------------------------------------------------------------------------
# Transect builder: from a georeferenced time series to [transit, point]
# ---------------------------------------------------------------------------

TRANSIT_COLUMNS = ["direction", "t_start", "t_end", "s_min", "s_max", "n"]


def _time_seconds(time) -> np.ndarray:
    """
    POSIX seconds (float) from numeric or datetime-like input; NaN for NaT.

    Datetime-like input goes through pandas, which handles tz-aware times (an object
    array to numpy) and NaT. Naive times are taken as UTC.
    """
    t = np.asarray(time)
    if t.dtype.kind in "biuf":
        return t.astype(float)
    dt = pd.DatetimeIndex(pd.to_datetime(t, utc=True))
    return ((dt - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(float)


def _runs(t: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Run index per sample; a new run starts after a gap longer than ``max_gap_s``."""
    new = np.ones(len(t), bool)
    new[1:] = np.diff(t) > max_gap_s
    return np.cumsum(new) - 1


def lag_positions(time, xy: np.ndarray, lag_s, max_gap_s: float = 600.0) -> np.ndarray:
    """
    Positions where the air of each sample was actually taken in.

    A sample logged at time *t* is air that entered the inlet ``lag_s`` seconds earlier, so
    it belongs at the platform's position at ``t - lag``. Positions are interpolated
    linearly in time along ``xy`` (``(n, 2)``, planar or lon/lat; ``time`` sorted).
    ``lag_s`` is a scalar or one value per sample (the lag differs by instrument epoch).

    The record is cut into runs at gaps longer than ``max_gap_s``, and ``t - lag`` is not
    allowed to reach back before the start of the sample's own run: the first seconds of a
    run keep its first position instead of being interpolated across the gap to wherever
    the previous run ended.

    Samples without a time (NaN / NaT) get a NaN position and are left out of the
    interpolation for the others.
    """
    t = _time_seconds(time)
    xy = np.asarray(xy, dtype=float)
    lag = np.broadcast_to(np.asarray(lag_s, dtype=float), t.shape)
    out = np.full_like(xy, np.nan)
    has_t = np.isfinite(t)
    if not has_t.any():
        return out
    t, lag, xy_t = t[has_t], lag[has_t], xy[has_t]
    run = _runs(t, max_gap_s)
    starts = np.flatnonzero(np.r_[True, np.diff(run) != 0])
    t_lag = np.maximum(t - lag, t[starts][run])
    for k in range(xy.shape[1]):
        out[has_t, k] = np.interp(t_lag, t, xy_t[:, k])
    return out


def snap_to_route(
    xy: np.ndarray, route_xy: np.ndarray, max_dist: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Nearest route point of each sample: ``(index, distance)``, index -1 beyond ``max_dist``.

    Both arrays are ``(n, 2)`` in the same projected metres. A sample with a NaN position
    (e.g. no time in :func:`lag_positions`) gets index -1 and distance NaN.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(xy, dtype=float)
    d = np.full(len(xy), np.nan)
    idx = np.full(len(xy), -1)
    finite = np.isfinite(xy).all(axis=1)
    if finite.any():
        d[finite], idx[finite] = cKDTree(np.asarray(route_xy, dtype=float)).query(
            xy[finite]
        )
    return np.where(d <= max_dist, idx, -1), d


def _empty_transit_table() -> pd.DataFrame:
    return pd.DataFrame(columns=TRANSIT_COLUMNS, index=pd.Index([], name="transit"))


def split_transits(
    time,
    s: np.ndarray,
    max_gap_s: float = 600.0,
    reversal_m: float = 500.0,
    min_span_m: float = 1000.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Cut a platform's along-route coordinate into one-way transits.

    ``s`` is the along-route position of each sample (metres; NaN where the sample is not
    on this route) and ``time`` is sorted. A transit ends

    - at a time gap longer than ``max_gap_s`` (overnight, or the record stops), or
    - at a reversal of the direction of travel: a turning point of ``s`` with prominence
      of at least ``reversal_m`` (``scipy.signal.find_peaks`` on ``s`` and on ``-s``).
      GPS jitter and short shunts stay inside a transit; a turnaround at a terminus does
      not. A dwell at the terminus is cut at its middle, between the arriving and the
      departing transit (trim it with ``max_dwell_s`` in :func:`transect_matrix`).

    Transits covering less than ``min_span_m`` of route are discarded. Samples without a
    time (NaN / NaT) are skipped like samples off the route.

    Returns ``(transit, table)``: ``transit`` is the transit index per sample (-1 for
    samples not in a kept transit) and ``table`` has one row per transit, indexed by
    that number: ``direction`` (+1 with increasing ``s``, -1 against it), ``t_start``,
    ``t_end`` (POSIX s), ``s_min``, ``s_max`` and ``n`` samples.
    """
    from scipy.signal import find_peaks

    t = _time_seconds(time)
    s = np.asarray(s, dtype=float)
    transit = np.full(len(t), -1, dtype=int)
    idx_on = np.flatnonzero(np.isfinite(s) & np.isfinite(t))
    if len(idx_on) == 0:
        return transit, _empty_transit_table()
    t_on, s_on = t[idx_on], s[idx_on]
    run = _runs(t_on, max_gap_s)
    run_starts = np.flatnonzero(np.r_[True, np.diff(run) != 0])
    run_ends = np.r_[run_starts[1:], len(run)]
    rows = []
    k = 0
    for a0, b0 in zip(run_starts, run_ends, strict=True):
        sr = s_on[a0:b0]
        if len(sr) < 2:
            continue
        hi, _ = find_peaks(sr, prominence=reversal_m)
        lo, _ = find_peaks(-sr, prominence=reversal_m)
        cuts = np.unique(np.r_[0, hi, lo, len(sr)])
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            seg = idx_on[a0 + a : a0 + b]
            ss = s[seg]
            if ss.max() - ss.min() < min_span_m:
                continue
            transit[seg] = k
            rows.append(
                (
                    k,
                    1 if ss[-1] >= ss[0] else -1,
                    t[seg[0]],
                    t[seg[-1]],
                    ss.min(),
                    ss.max(),
                    len(seg),
                )
            )
            k += 1
    if not rows:
        return transit, _empty_transit_table()
    table = pd.DataFrame(rows, columns=["transit", *TRANSIT_COLUMNS]).set_index(
        "transit"
    )
    return transit, table


def transect_matrix(
    transit: np.ndarray,
    point: np.ndarray,
    obs: np.ndarray,
    time,
    n_transits: int,
    n_points: int,
    max_dwell_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Average samples onto a ``[transit, point]`` matrix.

    ``transit`` comes from :func:`split_transits` and ``point`` from :func:`snap_to_route`
    (-1 in either skips the sample); ``obs`` and ``time`` are per sample. With
    ``max_dwell_s``, samples at a point taken more than that long after the transit first
    reached the point are dropped, so a platform sitting at a stop contributes a pass, not
    a long time-average, to that point. Samples without a time (NaN / NaT) are skipped.

    Returns ``(obs, time, n)``: the mean observation, the mean POSIX time and the number of
    samples per cell, NaN where a transit has no sample at a point.
    """
    t = _time_seconds(time)
    obs = np.asarray(obs, dtype=float)
    ok = (transit >= 0) & (point >= 0) & np.isfinite(obs) & np.isfinite(t)
    df = pd.DataFrame(
        {"transit": transit[ok], "point": point[ok], "obs": obs[ok], "t": t[ok]}
    )
    if max_dwell_s is not None:
        first = df.groupby(["transit", "point"])["t"].transform("min")
        df = df[(df["t"] - first) <= max_dwell_s]
    agg = df.groupby(["transit", "point"]).agg(
        obs=("obs", "mean"), t=("t", "mean"), n=("obs", "size")
    )
    out_obs = np.full((n_transits, n_points), np.nan)
    out_t = np.full_like(out_obs, np.nan)
    out_n = np.full_like(out_obs, np.nan)
    ti = agg.index.get_level_values("transit").to_numpy()
    pi = agg.index.get_level_values("point").to_numpy()
    out_obs[ti, pi] = agg["obs"].to_numpy()
    out_t[ti, pi] = agg["t"].to_numpy()
    out_n[ti, pi] = agg["n"].to_numpy(dtype=float)
    return out_obs, out_t, out_n
