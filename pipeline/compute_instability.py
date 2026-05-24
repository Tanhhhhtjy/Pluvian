"""Compute 6 instability indices on ERA5 8-level grid (T3).

Outputs per month (2023-05..09), shape (T, lat, lon):
  era5_cape_2023MM.nc     J/kg   (MU-CAPE)
  era5_cin_2023MM.nc      J/kg   (MU-CIN, negative)
  era5_kindex_2023MM.nc   degC
  era5_lcl_2023MM.nc      hPa
  era5_si_2023MM.nc       delta_degC (Showalter)
  era5_tt_2023MM.nc       degC   (Total Totals)

Grid: ERA5 native 0.25 deg over (110-122 E, 33-45 N) -> 49x49.
Levels: 925/850/700/600/500/400/300/250 hPa (decreasing required by metpy).

Parallelism: ProcessPoolExecutor across timesteps; each worker handles one
hour and loops 2401 gridpoints for MU-CAPE/CIN and Showalter (the per-profile
metpy calls). Vectorised numpy for K-index, Total Totals, and dewpoint;
vectorised metpy.lcl for LCL.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import xarray as xr

# Silence metpy/pint nan warnings during per-profile loops
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="metpy")

ERA5_ROOT = Path("/Data/tanh/npj/era5")
OUT_ROOT = Path("/Data/tanh/npj/derived/instability")
LOG_PATH = OUT_ROOT / "compute.log"

LEVELS = (925, 850, 700, 600, 500, 400, 300, 250)  # high -> low pressure
P_LEVELS = np.array(LEVELS, dtype=np.float64)
IDX_850 = LEVELS.index(850)
IDX_700 = LEVELS.index(700)
IDX_500 = LEVELS.index(500)
IDX_925 = LEVELS.index(925)


def _open_var_level(var: str, level: int, year: int, month: int) -> xr.DataArray:
    p = ERA5_ROOT / f"era5_{var}_{level}_{year}{month:02d}.nc"
    ds = xr.open_dataset(p)
    da = ds[var].squeeze(drop=True)
    return da


def load_month(year: int, month: int):
    """Return (times, lat, lon, T (T, L, lat, lon) K, q (T, L, lat, lon) kg/kg)."""
    t_layers, q_layers = [], []
    times = lat = lon = None
    for lvl in LEVELS:
        da_t = _open_var_level("t", lvl, year, month)
        da_q = _open_var_level("q", lvl, year, month)
        if times is None:
            times = da_t["valid_time"].values
            lat = da_t["latitude"].values
            lon = da_t["longitude"].values
        t_layers.append(da_t.values.astype(np.float32))
        q_layers.append(da_q.values.astype(np.float32))
    T = np.stack(t_layers, axis=1)  # (T, L, lat, lon) Kelvin
    q = np.stack(q_layers, axis=1)  # kg/kg
    return times, lat, lon, T, q


# ---------- Worker (per timestep) ----------
# Module-level so it pickles cleanly on Linux fork; we also avoid passing pint.

def _worker_init():
    # Import once per worker to amortise metpy import cost.
    global _mc, _units
    import metpy.calc as mc  # noqa: WPS433
    from metpy.units import units  # noqa: WPS433
    _mc = mc
    _units = units


def _profile_indices(p_hpa, T_C, Td_C):
    """Compute (MU-CAPE, MU-CIN, SI) for a single column. Returns floats (J/kg, J/kg, degC)."""
    u = _units
    p = p_hpa * u.hPa
    T = T_C * u.degC
    Td = Td_C * u.degC
    if (
        not np.all(np.isfinite(T_C))
        or not np.all(np.isfinite(Td_C))
        or np.any(Td_C > T_C + 0.01)
    ):
        # Clip slight super-saturation
        Td_C = np.minimum(Td_C, T_C)
        Td = Td_C * u.degC
    try:
        cape, cin = _mc.most_unstable_cape_cin(p, T, Td)
        cape_v = float(cape.to("J/kg").magnitude)
        cin_v = float(cin.to("J/kg").magnitude)
    except Exception:
        cape_v, cin_v = np.nan, np.nan
    try:
        si = _mc.showalter_index(p, T, Td)
        # showalter returns shape (1,) sometimes
        si_v = float(np.atleast_1d(si.magnitude)[0])
    except Exception:
        si_v = np.nan
    return cape_v, cin_v, si_v


def _process_timestep(args):
    """For one (t_idx, T_C(L,H,W), Td_C(L,H,W)) compute CAPE/CIN/SI grids."""
    t_idx, T_C, Td_C = args
    L, H, W = T_C.shape
    cape = np.full((H, W), np.nan, dtype=np.float32)
    cin = np.full((H, W), np.nan, dtype=np.float32)
    si = np.full((H, W), np.nan, dtype=np.float32)
    p_hpa = P_LEVELS
    for j in range(H):
        for i in range(W):
            Tcol = T_C[:, j, i]
            Tdcol = Td_C[:, j, i]
            if not np.isfinite(Tcol).all() or not np.isfinite(Tdcol).all():
                continue
            cape[j, i], cin[j, i], si[j, i] = _profile_indices(p_hpa, Tcol, Tdcol)
    return t_idx, cape, cin, si


# ---------- Vectorised helpers ----------

def compute_td_kelvin(T_K, q_kgkg):
    """Vectorised dewpoint (degC) from (T_K, q kg/kg) on (T, L, lat, lon).

    Uses Bolton 1980 via metpy formula but implemented in numpy for speed and
    to avoid pint overhead on the full cube.
    """
    # Mixing ratio r = q / (1 - q)
    r = q_kgkg / (1.0 - q_kgkg)
    # Vapor pressure e = p * r / (eps + r), eps=0.622; p needs to broadcast (L,)
    eps = 0.621981
    p_pa = P_LEVELS[None, :, None, None] * 100.0  # hPa -> Pa
    e_pa = p_pa * r / (eps + r)
    # Bolton 1980 inverse for Td (degC) from e (hPa)
    e_hpa = np.maximum(e_pa / 100.0, 1e-6)
    ln_e = np.log(e_hpa / 6.112)
    Td_C = (243.5 * ln_e) / (17.67 - ln_e)
    return Td_C.astype(np.float32)


def compute_k_tt(T_C, Td_C):
    """Vectorised K-index and TT (degC) on (T, L, lat, lon)."""
    T850 = T_C[:, IDX_850]
    T700 = T_C[:, IDX_700]
    T500 = T_C[:, IDX_500]
    Td850 = Td_C[:, IDX_850]
    Td700 = Td_C[:, IDX_700]
    K = (T850 - T500) + Td850 - (T700 - Td700)
    TT = (T850 + Td850) - 2.0 * T500
    return K.astype(np.float32), TT.astype(np.float32)


def compute_lcl_grid(T_C, Td_C):
    """LCL pressure (hPa) using 925 hPa parcel, vectorised via metpy."""
    import metpy.calc as mc
    from metpy.units import units
    p0 = np.full(T_C.shape[2:], 925.0)
    out = np.full((T_C.shape[0],) + T_C.shape[2:], np.nan, dtype=np.float32)
    for t in range(T_C.shape[0]):
        Tg = T_C[t, IDX_925]
        Tdg = Td_C[t, IDX_925]
        finite = np.isfinite(Tg) & np.isfinite(Tdg)
        if not finite.any():
            continue
        Tdg = np.minimum(Tdg, Tg)
        try:
            lcl_p, _ = mc.lcl(p0 * units.hPa, Tg * units.degC, Tdg * units.degC)
            arr = lcl_p.to("hPa").magnitude.astype(np.float32)
            arr[~finite] = np.nan
            out[t] = arr
        except Exception:
            pass
    return out


# ---------- IO ----------

def save_field(data, times, lat, lon, name, units_str, long_name, path):
    da = xr.DataArray(
        data,
        dims=("valid_time", "latitude", "longitude"),
        coords={"valid_time": times, "latitude": lat, "longitude": lon},
        name=name,
        attrs={"units": units_str, "long_name": long_name},
    )
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"}}
    da.to_netcdf(path, encoding=encoding)


# ---------- Per-month driver ----------

def process_month(year: int, month: int, logger: logging.Logger, n_workers: int):
    tag = f"{year}{month:02d}"
    logger.info(f"=== Month {tag} ===")
    t0 = time.time()
    times, lat, lon, T_K, q = load_month(year, month)
    T_C = (T_K - 273.15).astype(np.float32)
    Td_C = compute_td_kelvin(T_K, q)
    nT, nL, H, W = T_C.shape
    logger.info(f"  loaded T,q shape=(T={nT},L={nL},H={H},W={W}) in {time.time()-t0:.1f}s")

    # Vectorised cheap indices
    t1 = time.time()
    K, TT = compute_k_tt(T_C, Td_C)
    logger.info(f"  K, TT (vectorised) in {time.time()-t1:.2f}s")

    t1 = time.time()
    LCL = compute_lcl_grid(T_C, Td_C)
    logger.info(f"  LCL (metpy vectorised per-step) in {time.time()-t1:.1f}s")

    # Heavy: CAPE/CIN, SI per profile via process pool
    t1 = time.time()
    CAPE = np.full((nT, H, W), np.nan, dtype=np.float32)
    CIN = np.full((nT, H, W), np.nan, dtype=np.float32)
    SI = np.full((nT, H, W), np.nan, dtype=np.float32)
    tasks = [(t, T_C[t], Td_C[t]) for t in range(nT)]
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init) as ex:
        for t_idx, cape_g, cin_g, si_g in ex.map(_process_timestep, tasks, chunksize=4):
            CAPE[t_idx] = cape_g
            CIN[t_idx] = cin_g
            SI[t_idx] = si_g
            done += 1
            if done % 60 == 0:
                el = time.time() - t1
                logger.info(
                    f"    CAPE/SI progress {done}/{nT} ({100*done/nT:.0f}%) "
                    f"elapsed={el:.0f}s rate={done/max(el,1e-3):.2f} steps/s"
                )
    logger.info(f"  CAPE/CIN/SI (per-profile, {n_workers} workers) in {time.time()-t1:.0f}s")

    # Save
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_field(CAPE, times, lat, lon, "cape", "J/kg",
               "Most-unstable CAPE (metpy)", OUT_ROOT / f"era5_cape_{tag}.nc")
    save_field(CIN, times, lat, lon, "cin", "J/kg",
               "Most-unstable CIN (metpy, negative)", OUT_ROOT / f"era5_cin_{tag}.nc")
    save_field(K, times, lat, lon, "kindex", "degC",
               "K-index from 850/700/500 hPa", OUT_ROOT / f"era5_kindex_{tag}.nc")
    save_field(LCL, times, lat, lon, "lcl", "hPa",
               "LCL pressure of 925 hPa parcel", OUT_ROOT / f"era5_lcl_{tag}.nc")
    save_field(SI, times, lat, lon, "si", "degC",
               "Showalter index (850 -> 500 hPa)", OUT_ROOT / f"era5_si_{tag}.nc")
    save_field(TT, times, lat, lon, "tt", "degC",
               "Total Totals (850/500 hPa)", OUT_ROOT / f"era5_tt_{tag}.nc")

    # Diagnostics: global mins/maxes/NaN-fraction
    def diag(name, arr):
        nan_frac = np.mean(np.isnan(arr))
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            logger.info(f"    {name}: ALL NaN")
            return
        logger.info(
            f"    {name}: min={finite.min():.2f} max={finite.max():.2f} "
            f"mean={finite.mean():.2f} nan_frac={nan_frac:.3%}"
        )
    diag("CAPE", CAPE)
    diag("CIN", CIN)
    diag("K", K)
    diag("LCL", LCL)
    diag("SI", SI)
    diag("TT", TT)

    # 23.7 event probe: 7-28 18:00 UTC, BJ core (115-118E, 39-41N)
    if month == 7:
        target = np.datetime64("2023-07-28T18:00")
        idx = int(np.argmin(np.abs(times.astype("datetime64[ns]") - target)))
        lat_msk = (lat >= 39.0) & (lat <= 41.0)
        lon_msk = (lon >= 115.0) & (lon <= 118.0)
        sub = lambda a: a[idx][np.ix_(lat_msk, lon_msk)]
        logger.info("  [23.7 PROBE 2023-07-28T18:00 UTC, 115-118E,39-41N]")
        for nm, arr in [("CAPE", CAPE), ("CIN", CIN), ("K", K),
                         ("LCL", LCL), ("SI", SI), ("TT", TT)]:
            s = sub(arr)
            if np.all(np.isnan(s)):
                logger.info(f"    {nm}: ALL NaN")
                continue
            logger.info(
                f"    {nm}: min={np.nanmin(s):.2f} max={np.nanmax(s):.2f} mean={np.nanmean(s):.2f}"
            )

    logger.info(f"  month {tag} total {time.time()-t0:.0f}s")


def setup_logger() -> logging.Logger:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("compute_instability")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, mode="a")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) - 1))
    args = ap.parse_args()
    logger = setup_logger()
    logger.info(f"start months={args.months} workers={args.workers}")
    for m in args.months:
        process_month(args.year, m, logger, args.workers)
    logger.info("DONE")


if __name__ == "__main__":
    main()
