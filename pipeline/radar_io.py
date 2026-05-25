"""Radar IO: read 6-min radar frames as (T, H, W) numpy.

`data0` is stored as int32 with `scale_factor=0.01`; xarray's default decoding
applies that factor and yields physical dBZ (0 – ~70). DATA_HANDLING.md §2.1
calls this "integer dBZ" — meaning values are quantized at 1 dBZ resolution,
*not* that scale_factor should be skipped.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

from .utils import RADAR_H, RADAR_W

# === PLUVIAN PATH RESOLVER ===
import os as _os
_REPO_ROOT = _os.environ.get("PLUVIAN_REPO_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
def _data_path(*parts):
    return _os.path.join(_REPO_ROOT, *parts)


RADAR_ROOT = Path(_data_path("radar_nc"))


def date_dir(date) -> Path:
    d = pd.Timestamp(date)
    return RADAR_ROOT / f"{d.year}{d.month:02d}" / d.strftime("%Y%m%d")


def has_radar_day(date) -> bool:
    return date_dir(date).is_dir()


def _read_frame(path: Path) -> np.ndarray:
    # xarray default decoding applies scale_factor=0.01 so values come out in
    # the physical 0-~70 dBZ range — what downstream code treats as "integer
    # dBZ" per DATA_HANDLING.md §2.1.
    with xr.open_dataset(path) as ds:
        arr = ds["data0"].values
    arr = np.squeeze(arr)  # -> (H, W)
    arr = np.asarray(arr, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def load_radar_sequence(start_time, n_frames: int):
    """Load contiguous 6-min frames starting at ``start_time``.

    Returns ``(radar, frame_mask, sample_valid, times)``:
      - ``radar``: ``(T, H, W) float32`` physical dBZ, last-frame ffill across
        short gaps (zeros until the first observed frame).
      - ``frame_mask``: ``(T,) float32`` 0/1, 1 = frame was observed.
      - ``sample_valid``: ``bool``; False when any contiguous run of missing
        frames exceeds 3 (DATA_HANDLING.md §2.3).
    """
    start = pd.Timestamp(start_time)
    times = [start + pd.Timedelta(minutes=6 * i) for i in range(n_frames)]
    out = np.zeros((n_frames, RADAR_H, RADAR_W), dtype=np.float32)
    frame_mask = np.zeros(n_frames, dtype=np.float32)
    last = None
    max_run = cur_run = 0
    for i, t in enumerate(times):
        f = date_dir(t) / f"{t.strftime('%Y%m%d_%H%M%S')}.nc"
        present = False
        if f.exists():
            try:
                last = _read_frame(f)
                present = True
            except Exception:
                present = False
        if present:
            frame_mask[i] = 1.0
            cur_run = 0
        else:
            cur_run += 1
            if cur_run > max_run:
                max_run = cur_run
        if last is not None:
            out[i] = last
    sample_valid = bool(max_run <= 3)
    return out, frame_mask, sample_valid, times
