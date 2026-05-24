"""Shared utilities for the npj data pipeline."""
from __future__ import annotations
import numpy as np

# Target radar grid (per DATA_HANDLING.md §2.3)
RADAR_LON = np.round(np.arange(113.00, 120.00 + 1e-6, 0.01), 2)  # 701
RADAR_LAT = np.round(np.arange(36.00, 42.60 + 1e-6, 0.01), 2)    # 661
RADAR_W = len(RADAR_LON)
RADAR_H = len(RADAR_LAT)


def dbz_to_rainrate(dbz):
    """Physical dBZ (xarray-decoded) -> rainrate mm/h via China operational
    Z-R relation Z = 300 R^1.4."""
    arr = np.asarray(dbz, dtype=np.float32)
    z = np.power(10.0, arr / 10.0)
    return np.power(z / 300.0, 1.0 / 1.4)


def bilinear_interp_to_radar(values, src_lat, src_lon, dst_lat=None, dst_lon=None):
    """Bilinear interp of a 2D or 3D field to dst grid.

    ``values`` shape:
      - (H_src, W_src) -> returns (H_dst, W_dst)
      - (T, H_src, W_src) -> returns (T, H_dst, W_dst)  (vectorized)

    src_lat may be decreasing (ERA5 convention).
    """
    if dst_lat is None:
        dst_lat = RADAR_LAT
    if dst_lon is None:
        dst_lon = RADAR_LON

    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    values = np.asarray(values, dtype=np.float32)

    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        if values.ndim == 2:
            values = np.ascontiguousarray(values[::-1, :])
        else:
            values = np.ascontiguousarray(values[:, ::-1, :])

    def _frac_idx(src, dst):
        idx = np.interp(dst, src, np.arange(len(src)))
        i0 = np.clip(np.floor(idx).astype(int), 0, len(src) - 2)
        f = (idx - i0).astype(np.float32)
        return i0, f

    ilat, flat_w = _frac_idx(src_lat, dst_lat)
    ilon, flon_w = _frac_idx(src_lon, dst_lon)
    flat2 = flat_w[:, None]
    flon2 = flon_w[None, :]

    if values.ndim == 2:
        v00 = values[np.ix_(ilat, ilon)]
        v10 = values[np.ix_(ilat + 1, ilon)]
        v01 = values[np.ix_(ilat, ilon + 1)]
        v11 = values[np.ix_(ilat + 1, ilon + 1)]
        out = (v00 * (1 - flat2) * (1 - flon2)
               + v10 * flat2 * (1 - flon2)
               + v01 * (1 - flat2) * flon2
               + v11 * flat2 * flon2)
        return out.astype(np.float32)

    v00 = values[:, ilat[:, None], ilon[None, :]]
    v10 = values[:, (ilat + 1)[:, None], ilon[None, :]]
    v01 = values[:, ilat[:, None], (ilon + 1)[None, :]]
    v11 = values[:, (ilat + 1)[:, None], (ilon + 1)[None, :]]
    out = (v00 * (1 - flat2) * (1 - flon2)
           + v10 * flat2 * (1 - flon2)
           + v01 * (1 - flat2) * flon2
           + v11 * flat2 * flon2)
    return out.astype(np.float32)


def linear_time_interp(t_src, values, t_dst):
    """Per-pixel linear interpolation in time. values shape (T_src, ...).

    Builds a sparse (T_dst, T_src) weight matrix and does a single BLAS
    matmul — fast even for ~500k pixel layers.
    """
    t_src = np.asarray(t_src, dtype="datetime64[ns]").astype("int64").astype(np.float64)
    t_dst = np.asarray(t_dst, dtype="datetime64[ns]").astype("int64").astype(np.float64)
    values = np.asarray(values, dtype=np.float32)
    T_src = t_src.shape[0]
    T_dst = t_dst.shape[0]

    idx_right = np.clip(np.searchsorted(t_src, t_dst, side="left"), 1, T_src - 1)
    idx_left = idx_right - 1
    t_l = t_src[idx_left]
    t_r = t_src[idx_right]
    w_r = np.clip((t_dst - t_l) / np.maximum(t_r - t_l, 1.0), 0.0, 1.0)
    w_l = 1.0 - w_r

    W = np.zeros((T_dst, T_src), dtype=np.float32)
    rows = np.arange(T_dst)
    W[rows, idx_left] += w_l.astype(np.float32)
    W[rows, idx_right] += w_r.astype(np.float32)

    rest = values.shape[1:]
    flat = values.reshape(T_src, -1)
    out = W @ flat
    return out.reshape((T_dst,) + rest)
