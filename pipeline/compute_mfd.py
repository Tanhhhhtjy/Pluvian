"""Compute moisture flux divergence MFD = -div(q*V) from ERA5 925/850/700 hPa.

Output: 5 months x 3 levels = 15 NetCDF files in /Data/tanh/npj/derived/mfd/.
Grid is kept at ERA5 native 0.25 deg; downstream loader handles 0.01 deg interp.

Units check:
    q  [kg/kg], u,v [m/s]  ->  q*V [m/s]  ->  d(qV)/dx [1/s] = [kg kg^-1 s^-1]
Positive MFD = moisture convergence (sink of divergence).

Reuses pipeline.era5_io._load_monthly so reads are identical to the training
loader (subset box, lat ascending, valid_time int64 ns).
"""
from __future__ import annotations
import sys
import time
import logging
import traceback
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.era5_io import _load_monthly  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "derived" / "mfd"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "compute.log"

YEAR = 2023
LEVELS = (925, 850, 700)
MONTHS = (5, 6, 7, 8, 9)
R_EARTH = 6_371_000.0  # m
SIGMA = 2.0  # Gaussian smoothing in grid points
N_WORKERS = 6


def _spherical_divergence(qu: np.ndarray, qv: np.ndarray,
                          lat_asc: np.ndarray, lon_asc: np.ndarray) -> np.ndarray:
    """Centred FD divergence on a regular lon/lat grid (lat ascending).

    qu, qv shape (T, H, W). Returns d(qu)/dx + d(qv)/dy with shape (T, H, W).
    dx varies with latitude (R*cos(lat)*dlon), dy is constant (R*dlat).
    """
    dlon = np.deg2rad(float(np.mean(np.diff(lon_asc))))
    dlat = np.deg2rad(float(np.mean(np.diff(lat_asc))))
    coslat = np.cos(np.deg2rad(lat_asc)).astype(np.float32)  # (H,)
    dx = (R_EARTH * coslat * dlon).astype(np.float32)        # (H,)
    dy = np.float32(R_EARTH * dlat)

    dqu_dx = np.empty_like(qu)
    dqu_dx[:, :, 1:-1] = (qu[:, :, 2:] - qu[:, :, :-2]) / (2.0 * dx[None, :, None])
    dqu_dx[:, :,  0]   = (qu[:, :,  1] - qu[:, :,  0]) /        dx[None, :]
    dqu_dx[:, :, -1]   = (qu[:, :, -1] - qu[:, :, -2]) /        dx[None, :]

    dqv_dy = np.empty_like(qv)
    dqv_dy[:, 1:-1, :] = (qv[:, 2:, :] - qv[:, :-2, :]) / (2.0 * dy)
    dqv_dy[:,  0,  :]  = (qv[:,  1, :] - qv[:,  0, :]) /        dy
    dqv_dy[:, -1,  :]  = (qv[:, -1, :] - qv[:, -2, :]) /        dy

    return dqu_dx + dqv_dy


def compute_mfd_field(u, v, q, lat_asc, lon_asc):
    """Return MFD = -div(q*V), Gaussian-smoothed, shape (T, H, W), float32, units 1/s."""
    qu = (q * u).astype(np.float32)
    qv = (q * v).astype(np.float32)
    div = _spherical_divergence(qu, qv, lat_asc, lon_asc)
    mfd = (-div).astype(np.float32)
    for t in range(mfd.shape[0]):
        mfd[t] = gaussian_filter(mfd[t], sigma=SIGMA, mode="nearest")
    return mfd


def process_one(args):
    level, month = args
    t0 = time.time()
    try:
        tu, lat, lon, u = _load_monthly("u", level, YEAR, month)
        tv, _, _,   v   = _load_monthly("v", level, YEAR, month)
        tq, _, _,   q   = _load_monthly("q", level, YEAR, month)
        assert lat[0] < lat[-1], "expected ascending latitude from _load_monthly"
        if not (np.array_equal(tu, tv) and np.array_equal(tu, tq)):
            return dict(level=level, month=month, status="TIME_MISMATCH",
                        dt=time.time() - t0)

        mfd = compute_mfd_field(u, v, q, lat, lon)

        out_path = OUT_DIR / f"era5_mfd_{level}_{YEAR}{month:02d}.nc"
        ds = xr.Dataset(
            data_vars={
                "mfd": (("valid_time", "latitude", "longitude"), mfd, {
                    "long_name": "Moisture flux divergence (negative of divergence of qV)",
                    "units": "kg kg^-1 s^-1",
                    "formula": "MFD = -div(q*V) = -(d(qu)/dx + d(qv)/dy)",
                    "sign_convention": "positive = moisture convergence",
                    "smoothing": "Gaussian sigma=2 grid points",
                    "scheme": "spherical centred finite difference, R=6371 km",
                }),
            },
            coords={
                "valid_time": tu.astype("datetime64[ns]"),
                "latitude": lat,
                "longitude": lon,
                "pressure_level": level,
            },
            attrs={
                "source": "ERA5 reanalysis-era5-pressure-levels",
                "level_hPa": int(level),
                "year_month": f"{YEAR}-{month:02d}",
                "generator": "pipeline/compute_mfd.py",
            },
        )
        enc = {"mfd": {"zlib": True, "complevel": 4, "dtype": "float32"}}
        ds.to_netcdf(out_path, encoding=enc, engine="netcdf4")
        ds.close()

        info = dict(min=float(mfd.min()), max=float(mfd.max()),
                    mean=float(mfd.mean()), std=float(mfd.std()))

        probe = None
        if level == 850 and month == 7:
            target = np.datetime64("2023-07-29T02:00:00", "ns").astype("int64")
            ti = int(np.argmin(np.abs(tu - target)))
            la_m = (lat >= 37.5) & (lat <= 39.5)
            lo_m = (lon >= 113.0) & (lon <= 115.0)
            box = mfd[ti][np.ix_(la_m, lo_m)]
            jj, ii = np.unravel_index(int(np.argmax(box)), box.shape)
            probe = dict(
                t_idx=ti,
                t=str(np.datetime64(int(tu[ti]), "ns")),
                box_min=float(box.min()),
                box_max=float(box.max()),
                argmax_lonlat=(float(lon[lo_m][ii]), float(lat[la_m][jj])),
            )
        return dict(level=level, month=month, status="OK", dt=time.time() - t0,
                    info=info, probe=probe, path=str(out_path),
                    shape=tuple(mfd.shape))
    except Exception as e:  # pragma: no cover
        return dict(level=level, month=month, status="ERROR",
                    dt=time.time() - t0,
                    error=f"{e}\n{traceback.format_exc()}")


def main():
    logging.basicConfig(
        filename=str(LOG_PATH), filemode="w", level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    root = logging.getLogger()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root.addHandler(sh)

    logging.info("Start MFD computation: levels=%s months=%s sigma=%g",
                 LEVELS, MONTHS, SIGMA)

    jobs = [(lvl, m) for lvl in LEVELS for m in MONTHS]
    t_start = time.time()
    with Pool(min(N_WORKERS, len(jobs))) as pool:
        results = pool.map(process_one, jobs)

    n_ok = 0
    g_min = +np.inf; g_max = -np.inf
    for r in results:
        lvl, mo, st, dt = r["level"], r["month"], r["status"], r["dt"]
        if st == "OK":
            n_ok += 1
            info = r["info"]
            g_min = min(g_min, info["min"]); g_max = max(g_max, info["max"])
            logging.info(
                "[OK]  L%d  %d-%02d  %.1fs  shape=%s  min=%.3e max=%.3e "
                "mean=%.3e std=%.3e  -> %s",
                lvl, YEAR, mo, dt, r["shape"], info["min"], info["max"],
                info["mean"], info["std"], r["path"])
            if r["probe"] is not None:
                p = r["probe"]
                judge = ("STRONG CONVERGENCE >5e-8 [OK]"
                         if p["box_max"] > 5e-8
                         else "WEAK / CHECK UNITS")
                logging.info(
                    "  23.7 probe  t=%s  850hPa  box(113-115E, 37.5-39.5N)  "
                    "min=%.3e max=%.3e argmax@(%.2fE, %.2fN)  [%s]",
                    p["t"], p["box_min"], p["box_max"],
                    p["argmax_lonlat"][0], p["argmax_lonlat"][1], judge)
        else:
            logging.error("[FAIL] L%d  %d-%02d  %.1fs  %s  %s",
                          lvl, YEAR, mo, dt, st, r.get("error", ""))

    logging.info("Done: %d/%d files OK  global_min=%.3e  global_max=%.3e  "
                 "wall=%.1fs", n_ok, len(results), g_min, g_max,
                 time.time() - t_start)


if __name__ == "__main__":
    main()
