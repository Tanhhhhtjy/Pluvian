"""Top-level Dataset.

Returns dict samples with radar, PWV, station, ERA5 tensors aligned to a
6-min × 0.01° radar grid window. See DATA_HANDLING.md.

ERA5 spatial interp is on-the-fly with ~6s/sample warm; use num_workers>=4 in DataLoader to amortize.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    class Dataset:  # minimal stand-in
        pass

from . import radar_io, pwv_io, station_io, era5_io

MANIFEST_PATH = Path(__file__).with_name("manifest.csv")


class NPJDataset(Dataset):
    """One sample = a `window_minutes` chunk starting at `start_time`.

    Args:
        manifest: path to manifest.csv (used to enumerate windows). If None,
            the on-disk manifest under pipeline/ is used.
        window_minutes: sample length in minutes (default 180 = 3 h, 30 frames).
        starts: optional explicit list of start timestamps to override.
    """

    def __init__(self, manifest=MANIFEST_PATH, window_minutes: int = 180,
                 starts=None, drop_pwv: bool = False, load_mfd: bool = False,
                 load_era5: bool = True):
        self.window_minutes = window_minutes
        self.n_frames = window_minutes // 6
        self.drop_pwv = drop_pwv
        self.load_mfd = bool(load_mfd)
        self.load_era5 = bool(load_era5)
        if starts is not None:
            self._starts = [pd.Timestamp(s) for s in starts]
            self._meta = [{"date": pd.Timestamp(s).strftime("%Y-%m-%d"),
                           "zero_day": False} for s in starts]
        else:
            mf = pd.read_csv(manifest)
            mf = mf[mf["split"].isin(("train", "val", "test_robust",
                                      "event_test"))]
            self._starts = []
            self._meta = []
            for _, row in mf.iterrows():
                day = pd.Timestamp(row["date"])
                # Phase 7a: honor per-row `starts_per_day` if present.
                if "starts_per_day" in row and not pd.isna(row["starts_per_day"]):
                    spd = int(row["starts_per_day"])
                else:
                    spd = 8
                if spd <= 0:
                    continue
                step_hours = 24.0 / spd
                for k in range(spd):
                    self._starts.append(day + pd.Timedelta(hours=step_hours * k))
                    self._meta.append({"date": row["date"],
                                       "zero_day": bool(row["zero_day"]),
                                       "split": row["split"]})

    def __len__(self):
        return len(self._starts)

    def _to_tensor(self, x):
        if _HAS_TORCH:
            return torch.from_numpy(np.ascontiguousarray(x))
        return x

    def __getitem__(self, idx):
        start = self._starts[idx]
        meta = dict(self._meta[idx])

        radar, radar_mask, radar_valid, _ = radar_io.load_radar_sequence(
            start, self.n_frames)
        if meta.get("zero_day"):
            radar[:] = 0.0
            radar_mask[:] = 1.0  # zero-day frames are "known no-rain"
            radar_valid = True
        meta["radar_sample_valid"] = radar_valid

        # PWV: 30 min steps
        n_pwv = max(1, self.window_minutes // 30)
        pwv_vals, pwv_mask, pwv_coords = pwv_io.load_pwv_window(start, n_pwv)
        # PWV 通道清零仅由显式 drop_pwv 开关控制（用于 +PWV ablation 的对照评估）。
        # 注意：旧逻辑曾对 test_robust 自动清零——那是因为旧 test_robust 在 9 月、
        # 本就无 PWV。新划分(Phase 7c)把 test_robust 移到 5–8 月 PWV 覆盖期，必须
        # 保留其真实 PWV，否则留出测试会被悄悄抹掉 PWV。
        if self.drop_pwv:
            pwv_vals[:] = 0.0
            pwv_mask[:] = 0.0

        # Stations: 1 h steps
        n_sta = max(1, self.window_minutes // 60)
        sta_vals, sta_mask, sta_coords = station_io.load_station_window(start, n_sta)

        # ERA5 (optional — skip when model doesn't use it to save DataLoader time)
        if self.load_era5:
            era5 = era5_io.load_era5_window(start, self.n_frames)
            era5_t = {k: self._to_tensor(v) for k, v in era5.items()}
        else:
            # Provide tiny zero tensors with right keys/shape signature so the
            # collate/split logic doesn't need to special-case
            import numpy as np
            zero = np.zeros((self.n_frames, 8, 4, 4), dtype=np.float32)
            era5_t = {k: self._to_tensor(zero) for k in ("u", "v", "q", "t")}

        sample = {
            "radar": self._to_tensor(radar),
            "radar_mask": self._to_tensor(radar_mask),
            "pwv_grid": self._to_tensor(pwv_vals),
            "pwv_mask": self._to_tensor(pwv_mask),
            "pwv_coords": self._to_tensor(pwv_coords),
            "station_grid": self._to_tensor(sta_vals),
            "station_mask": self._to_tensor(sta_mask),
            "station_coords": self._to_tensor(sta_coords),
            "era5": era5_t,
            "time": str(start),
            "meta": meta,
        }
        if self.load_mfd:
            mfd = era5_io.load_mfd_window(start, self.n_frames)
            sample["era5_mfd"] = self._to_tensor(mfd)
        return sample
