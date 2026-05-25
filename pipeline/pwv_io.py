"""PWV IO: 30-min, 16-col text per station.

* Drops the 7 "almost-empty" stations listed in DATA_HANDLING.md §3.3.
* For shared codes (numeric prefix also present in stations/), replaces the
  coarse 0.1° PWV coordinate with the surface-station precise (lon, lat).
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd

# === PLUVIAN PATH RESOLVER ===
import os as _os
_REPO_ROOT = _os.environ.get("PLUVIAN_REPO_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
def _data_path(*parts):
    return _os.path.join(_REPO_ROOT, *parts)


PWV_ROOT = Path(_data_path("pwv"))
STATIONS_ROOT = Path(_data_path("stations"))

DROP_STATIONS = {
    # 🔴 <1000 行（DATA_HANDLING.md §3.3）— 必丢
    "54530_TJA1", "M0168_SXDT", "A2961_KC01", "53782_SX92",
    "53788_HESH", "53594_LQIU", "A1623_XZZZ",
    # 其他 5-8 月行数 <1000 的边界 case（实测后认定）
    "53875_QINY", "54914_SDJY",
}

# 🟠 显著缺失 8 站（保留，每个 case study 之前需复核覆盖度）
PARTIAL_STATIONS = {
    "M0181_TJWQ", "A2567_SW01", "M0003_BJSH", "53888_PINS",
    "A1006_XCGY", "53889_BULZ", "53681_WUTA", "53783_XIYA",
}

PWV_COLS = ["YYYY", "MM", "DD", "hh", "mm", "ss", "SITE",
            "LON", "LAT", "HGT", "PRES", "TEMP",
            "ZTD", "ZHD", "ZWD", "PWV"]

# Radar bbox
LON_LO, LON_HI = 113.0, 120.0
LAT_LO, LAT_HI = 36.0, 42.6


def _station_code(path: Path) -> str:
    """File stem like '53392_SZKB_gps_...'; we keep '53392_SZKB'."""
    parts = path.name.split("_")
    return f"{parts[0]}_{parts[1]}"


def _numeric_id(code: str) -> str:
    return code.split("_", 1)[0]


@lru_cache(maxsize=1)
def _station_coord_index() -> dict:
    """Map numeric station id -> (lon, lat) from surface station csv (first row)."""
    out = {}
    for p in STATIONS_ROOT.glob("station_*.csv"):
        sid = p.stem.replace("station_", "")
        try:
            df = pd.read_csv(p, sep=r"\s+", nrows=1)
            out[sid] = (float(df["Lon"].iloc[0]), float(df["Lat"].iloc[0]))
        except Exception:
            continue
    return out


@lru_cache(maxsize=512)
def _read_one_station(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str, sep=r"\s+", header=0, names=PWV_COLS,
                     skiprows=1, low_memory=False)
    df["time"] = pd.to_datetime(
        dict(year=df.YYYY, month=df.MM, day=df.DD,
             hour=df.hh, minute=df.mm, second=df.ss),
        errors="coerce")
    return df.dropna(subset=["time"])


@lru_cache(maxsize=1)
def list_pwv_stations() -> list:
    """Return list of dicts: code, path, lon, lat (corrected if shared)."""
    sta_idx = _station_coord_index()
    stations = []
    for p in sorted(PWV_ROOT.glob("*.txt")):
        code = _station_code(p)
        if code in DROP_STATIONS:
            continue
        # Read first valid row for coordinates
        try:
            first = pd.read_csv(p, sep=r"\s+", header=0, names=PWV_COLS,
                                skiprows=1, nrows=1)
            lon = float(first["LON"].iloc[0])
            lat = float(first["LAT"].iloc[0])
        except Exception:
            continue
        if not (LON_LO <= lon <= LON_HI and LAT_LO <= lat <= LAT_HI):
            continue
        sid = _numeric_id(code)
        if sid in sta_idx:
            lon, lat = sta_idx[sid]
        stations.append({"code": code, "path": str(p),
                         "lon": float(lon), "lat": float(lat)})
    return stations


def load_pwv_window(start_time, n_steps: int, step_minutes: int = 30):
    """Sample PWV at ``n_steps`` half-hour times starting at start_time.

    Returns ``(values, mask, coords)`` where values/mask have shape
    ``(n_steps, n_stations)`` and coords is ``(n_stations, 2)`` lon/lat.
    """
    start = pd.Timestamp(start_time)
    times = pd.date_range(start, periods=n_steps, freq=f"{step_minutes}min")
    stations = list_pwv_stations()
    values = np.full((n_steps, len(stations)), np.nan, dtype=np.float32)
    coords = np.array([[s["lon"], s["lat"]] for s in stations], dtype=np.float32)

    t_lo, t_hi = times.min(), times.max()
    for j, s in enumerate(stations):
        try:
            df = _read_one_station(s["path"])
        except Exception:
            continue
        m = (df["time"] >= t_lo) & (df["time"] <= t_hi)
        sub = df.loc[m, ["time", "PWV"]].set_index("time")
        if sub.empty:
            continue
        sub = sub[~sub.index.duplicated()]
        # last-fill to requested timestamps
        re = sub.reindex(times, method="ffill")
        values[:, j] = re["PWV"].astype(np.float32).to_numpy()

    mask = np.isfinite(values).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0)
    return values, mask, coords
