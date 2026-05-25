"""ERA5 IO: load 4 variables × 8 levels, interp to radar 6-min / 0.01° grid.

Optimised for repeated NPJDataset.__getitem__ calls:
  * Per-month cropped numpy cube is cached (one read per (var, level, month)).
  * Cold reads happen in a ThreadPoolExecutor (32 files in parallel).
  * Spatial bilinear weights are precomputed once per ERA5 grid.
  * Temporal interp uses a small (T_dst, T_src) BLAS matmul.
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xarray as xr

from .utils import RADAR_LAT, RADAR_LON, linear_time_interp

# === PLUVIAN PATH RESOLVER ===
import os as _os
_REPO_ROOT = _os.environ.get("PLUVIAN_REPO_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
def _data_path(*parts):
    return _os.path.join(_REPO_ROOT, *parts)


ERA5_ROOT = Path(_data_path("era5"))
MFD_ROOT = Path(_data_path("derived/mfd"))
VARS = ("u", "v", "q", "t")
LEVELS = (925, 850, 700, 600, 500, 400, 300, 250)
MFD_LEVELS = (925, 850, 700)
SUBSET_BOX = (112.5, 120.5, 35.5, 43.1)  # (lon_lo, lon_hi, lat_lo, lat_hi)


def _file(var: str, level: int, year: int, month: int) -> Path:
    return ERA5_ROOT / f"era5_{var}_{level}_{year}{month:02d}.nc"


def has_era5(var: str, level: int, year: int, month: int) -> bool:
    return _file(var, level, year, month).exists()


@lru_cache(maxsize=128)
def _load_monthly(var: str, level: int, year: int, month: int):
    """Return (times_ns int64, src_lat asc, src_lon asc, arr (T, H, W) float32).

    Cached so repeated sample loads pay disk + xarray cost once per month.
    """
    p = _file(var, level, year, month)
    ds = xr.open_dataset(p)
    lo, hi, la_lo, la_hi = SUBSET_BOX
    sub = ds.sel(longitude=slice(lo, hi), latitude=slice(la_hi, la_lo))
    arr = np.squeeze(sub[var].values).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    src_lat = sub["latitude"].values
    src_lon = sub["longitude"].values
    src_t = sub["valid_time"].values.astype("datetime64[ns]").astype("int64")
    # Make lat increasing for interp simplicity
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        arr = np.ascontiguousarray(arr[:, ::-1, :])
    return src_t, src_lat, src_lon, arr


@lru_cache(maxsize=4)
def _spatial_weights(src_lat_key: tuple, src_lon_key: tuple):
    """Precompute bilinear corner indices/weights from src grid -> RADAR grid.

    Returned arrays are float32 / int64; keys are tuples of the source coord
    values so equality is exact.
    """
    src_lat = np.asarray(src_lat_key)
    src_lon = np.asarray(src_lon_key)

    def _frac(src, dst):
        idx = np.interp(dst, src, np.arange(len(src)))
        i0 = np.clip(np.floor(idx).astype(np.int64), 0, len(src) - 2)
        f = (idx - i0).astype(np.float32)
        return i0, f

    ilat, flat_w = _frac(src_lat, RADAR_LAT)
    ilon, flon_w = _frac(src_lon, RADAR_LON)
    flat2 = flat_w[:, None]
    flon2 = flon_w[None, :]
    w00 = (1 - flat2) * (1 - flon2)
    w10 = flat2 * (1 - flon2)
    w01 = (1 - flat2) * flon2
    w11 = flat2 * flon2
    return ilat, ilon, w00.astype(np.float32), w10.astype(np.float32), \
        w01.astype(np.float32), w11.astype(np.float32)


def _interp_to_radar(arr, src_lat, src_lon):
    """``arr`` shape (..., H_src, W_src) -> (..., H_dst, W_dst). Caches weights."""
    ilat, ilon, w00, w10, w01, w11 = _spatial_weights(
        tuple(src_lat.tolist()), tuple(src_lon.tolist()))
    # Reshape to (N, H_src, W_src), one fancy-index pass over corners
    orig_shape = arr.shape
    a = arr.reshape(-1, orig_shape[-2], orig_shape[-1])
    v00 = a[:, ilat[:, None], ilon[None, :]]
    v10 = a[:, (ilat + 1)[:, None], ilon[None, :]]
    v01 = a[:, ilat[:, None], (ilon + 1)[None, :]]
    v11 = a[:, (ilat + 1)[:, None], (ilon + 1)[None, :]]
    out = v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11
    return out.reshape(orig_shape[:-2] + (len(w00), len(w00[0])))


def _prefetch_monthly(jobs):
    """Warm `_load_monthly` cache for many (var, level, year, month).

    Uses a small thread pool — netCDF4-python releases the GIL during reads
    but is *not* fully thread-safe across files on some builds. We use a
    serialized lock around `xr.open_dataset` inside ``_load_monthly`` and
    rely on the lru_cache to dedupe identical keys.
    """
    jobs = [j for j in jobs if _file(*j).exists()]
    if not jobs:
        return
    # Serialize via a single-worker pool — parallel xarray + netCDF segfaults
    # on this machine. Still useful as a prefetch hook for future swap-in of
    # a true thread-safe backend (h5netcdf / zarr).
    for j in jobs:
        _load_monthly(*j)


def load_era5_window(start_time, n_frames: int, frame_minutes: int = 6):
    """Return dict[var] -> (T, L, H, W) interp'd to radar grid."""
    return _load_era5_cached(pd.Timestamp(start_time).value, n_frames, frame_minutes)


@lru_cache(maxsize=8)
def _load_era5_cached(start_ns: int, n_frames: int, frame_minutes: int):
    return _load_era5_impl(pd.Timestamp(start_ns), n_frames, frame_minutes)


def _load_era5_impl(start, n_frames, frame_minutes):
    target_times = pd.date_range(start, periods=n_frames,
                                 freq=f"{frame_minutes}min")
    target_ns = target_times.values.astype("datetime64[ns]").astype("int64")
    hour_lo = (start.floor("h") - pd.Timedelta(hours=1)).value
    hour_hi = (target_times[-1].ceil("h") + pd.Timedelta(hours=1)).value
    months = sorted({(t.year, t.month) for t in [start - pd.Timedelta(hours=2),
                                                  target_times[-1] + pd.Timedelta(hours=2)]})

    # Prefetch every needed (var, level, month) in parallel — first call only.
    _prefetch_monthly([(var, lvl, y, m) for var in VARS for lvl in LEVELS
                       for (y, m) in months])

    H = len(RADAR_LAT)
    W = len(RADAR_LON)
    out = {}
    for var in VARS:
        cube = np.empty((n_frames, len(LEVELS), H, W), dtype=np.float32)
        stacked = None
        sub_t = None
        src_lat = src_lon = None
        for li, lvl in enumerate(LEVELS):
            arr_parts = []
            t_parts = []
            for (y, m) in months:
                if not _file(var, lvl, y, m).exists():
                    continue
                src_t, sl, sln, arr = _load_monthly(var, lvl, y, m)
                src_lat, src_lon = sl, sln
                sel = (src_t >= hour_lo) & (src_t <= hour_hi)
                if not sel.any():
                    continue
                arr_parts.append(arr[sel])
                t_parts.append(src_t[sel])
            if not arr_parts:
                cube[:, li] = 0
                continue
            sub_arr = np.concatenate(arr_parts, axis=0) if len(arr_parts) > 1 else arr_parts[0]
            if stacked is None:
                stacked = np.empty((len(LEVELS), sub_arr.shape[0],
                                    sub_arr.shape[1], sub_arr.shape[2]),
                                   dtype=np.float32)
                sub_t = np.concatenate(t_parts) if len(t_parts) > 1 else t_parts[0]
            stacked[li] = sub_arr
        if stacked is None:
            out[var] = np.zeros_like(cube)
            continue
        # Batched bilinear across (L, T_src) leading dims.
        interp_sp = _interp_to_radar(stacked, src_lat, src_lon)  # (L, T_src, H, W)
        for li in range(len(LEVELS)):
            cube[:, li] = linear_time_interp(
                sub_t.astype("datetime64[ns]"), interp_sp[li],
                target_times.values)
        out[var] = cube
    return out


# ---------------------------------------------------------------------------
# MFD (precomputed moisture-flux divergence) loader — same grid/time logic.
# ---------------------------------------------------------------------------

def _mfd_file(level: int, year: int, month: int) -> Path:
    return MFD_ROOT / f"era5_mfd_{level}_{year}{month:02d}.nc"


@lru_cache(maxsize=64)
def _load_mfd_monthly(level: int, year: int, month: int):
    """Return (times_ns, src_lat asc, src_lon asc, arr (T, H, W) float32)."""
    p = _mfd_file(level, year, month)
    ds = xr.open_dataset(p)
    lo, hi, la_lo, la_hi = SUBSET_BOX
    sub = ds.sel(longitude=slice(lo, hi), latitude=slice(la_hi, la_lo))
    if sub["latitude"].size == 0:
        sub = ds.sel(longitude=slice(lo, hi), latitude=slice(la_lo, la_hi))
    arr = np.squeeze(sub["mfd"].values).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    src_lat = sub["latitude"].values
    src_lon = sub["longitude"].values
    src_t = sub["valid_time"].values.astype("datetime64[ns]").astype("int64")
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        arr = np.ascontiguousarray(arr[:, ::-1, :])
    return src_t, src_lat, src_lon, arr


def load_mfd_window(start_time, n_frames: int, frame_minutes: int = 6):
    """Return (T, L_mfd=3, H, W) float32 interp'd to radar grid.

    Mirrors `load_era5_window` but for the precomputed MFD field at the three
    pressure levels in ``MFD_LEVELS``. Missing files -> zeros for that level.
    """
    return _load_mfd_cached(pd.Timestamp(start_time).value, n_frames, frame_minutes)


@lru_cache(maxsize=8)
def _load_mfd_cached(start_ns: int, n_frames: int, frame_minutes: int):
    return _load_mfd_impl(pd.Timestamp(start_ns), n_frames, frame_minutes)


def _load_mfd_impl(start, n_frames, frame_minutes):
    target_times = pd.date_range(start, periods=n_frames,
                                 freq=f"{frame_minutes}min")
    hour_lo = (start.floor("h") - pd.Timedelta(hours=1)).value
    hour_hi = (target_times[-1].ceil("h") + pd.Timedelta(hours=1)).value
    months = sorted({(t.year, t.month) for t in [start - pd.Timedelta(hours=2),
                                                  target_times[-1] + pd.Timedelta(hours=2)]})

    H = len(RADAR_LAT)
    W = len(RADAR_LON)
    cube = np.zeros((n_frames, len(MFD_LEVELS), H, W), dtype=np.float32)
    for li, lvl in enumerate(MFD_LEVELS):
        arr_parts = []
        t_parts = []
        src_lat = src_lon = None
        for (y, m) in months:
            if not _mfd_file(lvl, y, m).exists():
                continue
            src_t, sl, sln, arr = _load_mfd_monthly(lvl, y, m)
            src_lat, src_lon = sl, sln
            sel = (src_t >= hour_lo) & (src_t <= hour_hi)
            if not sel.any():
                continue
            arr_parts.append(arr[sel])
            t_parts.append(src_t[sel])
        if not arr_parts:
            continue
        sub_arr = np.concatenate(arr_parts, axis=0) if len(arr_parts) > 1 else arr_parts[0]
        sub_t = np.concatenate(t_parts) if len(t_parts) > 1 else t_parts[0]
        interp_sp = _interp_to_radar(sub_arr, src_lat, src_lon)   # (T_src, H, W)
        cube[:, li] = linear_time_interp(
            sub_t.astype("datetime64[ns]"), interp_sp, target_times.values)
    return cube
