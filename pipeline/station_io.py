"""Surface station IO: hourly 14-col CSV per station.

Per DATA_HANDLING.md §4, all 289 in-bbox stations are kept.
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


STATIONS_ROOT = Path(_data_path("stations"))

VAR_COLS = ["PRS", "WIN_D_INST", "WIN_S_INST", "TEM", "RHU", "PRE_1h"]
LON_LO, LON_HI = 113.0, 120.0
LAT_LO, LAT_HI = 36.0, 42.6


@lru_cache(maxsize=1)
def list_stations() -> list:
    stations = []
    for p in sorted(STATIONS_ROOT.glob("station_*.csv")):
        sid = p.stem.replace("station_", "")
        try:
            head = pd.read_csv(p, sep=r"\s+", nrows=1)
            lon = float(head["Lon"].iloc[0])
            lat = float(head["Lat"].iloc[0])
            alti = float(head["Alti"].iloc[0])
        except Exception:
            continue
        if not (LON_LO <= lon <= LON_HI and LAT_LO <= lat <= LAT_HI):
            continue
        stations.append({"code": sid, "path": str(p),
                         "lon": lon, "lat": lat, "alti": alti})
    return stations


@lru_cache(maxsize=512)
def _read_one_station(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str, sep=r"\s+", low_memory=False)
    df["time"] = pd.to_datetime(
        dict(year=df.Year, month=df.Mon, day=df.Day, hour=df.Hour),
        errors="coerce")
    df = df.dropna(subset=["time"])
    for c in VAR_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            # 9999-style sentinels
            df.loc[df[c].abs() > 1e4, c] = np.nan
    return df


def load_station_window(start_time, n_steps: int, step_minutes: int = 60):
    """Hourly hold-fill window. Returns values (n_steps, n_sta, 6),
    mask (n_steps, n_sta), coords (n_sta, 3) lon/lat/alti."""
    start = pd.Timestamp(start_time)
    times = pd.date_range(start, periods=n_steps, freq=f"{step_minutes}min")
    stations = list_stations()
    values = np.full((n_steps, len(stations), len(VAR_COLS)), np.nan, dtype=np.float32)
    coords = np.array([[s["lon"], s["lat"], s["alti"]] for s in stations], dtype=np.float32)

    t_lo, t_hi = times.min(), times.max()
    for j, s in enumerate(stations):
        try:
            df = _read_one_station(s["path"])
        except Exception:
            continue
        m = (df["time"] >= t_lo) & (df["time"] <= t_hi)
        sub = df.loc[m, ["time"] + VAR_COLS].set_index("time")
        if sub.empty:
            continue
        sub = sub[~sub.index.duplicated()]
        re = sub.reindex(times, method="ffill")
        values[:, j, :] = re[VAR_COLS].to_numpy(dtype=np.float32)

    mask = np.isfinite(values).any(axis=-1).astype(np.float32)
    values = np.nan_to_num(values, nan=0.0)
    return values, mask, coords


def daily_precip_diagnosis(date) -> tuple:
    """Compute (total_pre_mm, n_stations_with_>0.1mm) on the given calendar day."""
    d = pd.Timestamp(date).normalize()
    total = 0.0
    n_wet = 0
    for s in list_stations():
        try:
            df = _read_one_station(s["path"])
        except Exception:
            continue
        m = (df["time"] >= d) & (df["time"] < d + pd.Timedelta(days=1))
        v = df.loc[m, "PRE_1h"].dropna()
        if v.empty:
            continue
        ssum = float(v.clip(lower=0).sum())
        total += ssum
        if ssum > 0.1:
            n_wet += 1
    return total, n_wet
