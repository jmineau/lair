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

Everything is plain numpy; the platform-specific file formats live in the calling package
(e.g. ``slv.measurements.mobile`` for TRAX). Times are POSIX seconds (float) or datetime64.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "enhancement",
    "detection_frequency",
    "magnitude",
    "transit_times",
    "profile",
    "along_route_distance",
    "merge_route_points",
    "pool_routes",
]


def enhancement(obs: np.ndarray, baseline_q: float = 5.0) -> np.ndarray:
    """Enhancement above each transit's own ``baseline_q``-th percentile along the route.

    Transits with no finite values return NaN throughout.
    """
    obs = np.asarray(obs, dtype=float)
    ok = np.isfinite(obs).any(axis=1)
    base = np.full((obs.shape[0], 1), np.nan)
    base[ok, 0] = np.nanpercentile(obs[ok], baseline_q, axis=1)
    return obs - base


def detection_frequency(
    enh: np.ndarray, threshold: float, min_transits: int = 10
) -> np.ndarray:
    """Fraction of transits (with data at the point) whose enhancement exceeds ``threshold``.

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
    """Per-point enhancement magnitude: ``stat`` ("median" | "mean") over detected transits
    (``enh > threshold``) or over all transits with data (``threshold=None``).

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
    """One timestamp per transit from a ``time[transit, point]`` matrix (POSIX seconds or
    datetime64); NaT for transits with no data."""
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
    """Detection frequency and median magnitude per point, binned by ``by``
    ("hour", "weekday" or "month") of the transit time.

    Returns ``(bins, freq[bin, point], mag[bin, point])``. ``tz_offset_hours`` shifts the
    (UTC) transit times to local time before binning (e.g. -7 for MST).
    """
    t = times + pd.Timedelta(hours=tz_offset_hours)
    key = {"hour": t.hour, "weekday": t.weekday, "month": t.month}[by]
    bins = {"hour": np.arange(24), "weekday": np.arange(7), "month": np.arange(1, 13)}[by]
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
    """Merge the fixed points of several routes into one network point set.

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
    """Stack per-route ``[transit, point]`` matrices onto the network points.

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
