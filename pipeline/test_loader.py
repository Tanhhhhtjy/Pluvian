"""Smoke test for the data loader.

Loads a 3 h sample starting 2023-07-28 12:00 and reports shapes / ranges.
Self-checks (per Audit-T1 复审条件):
  1. ``pwv_io.list_pwv_stations()`` returns 176 stations (M0181_TJWQ kept).
  2. ``NPJDataset`` defaults to ``drop_pwv=False``; True zeroes PWV channel.
  3. Single warm sample wall-time < 5s.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data_loader import NPJDataset
from pipeline.utils import dbz_to_rainrate
from pipeline import pwv_io


def main():
    start = pd.Timestamp("2023-07-28 12:00")

    n_pwv = len(pwv_io.list_pwv_stations())
    print(f"[self-check 1] PWV stations = {n_pwv}")
    assert n_pwv == 176, f"expected 176 PWV stations, got {n_pwv}"

    # First load: cold cache (first-time ERA5 monthly read).
    t0 = time.time()
    ds = NPJDataset(starts=[start], window_minutes=180)
    sample = ds[0]
    cold = time.time() - t0
    # Second load on same start hits the per-start ERA5 cache and PWV/station caches.
    t0 = time.time()
    sample = ds[0]
    warm = time.time() - t0
    print(f"[self-check 3] wall: cold={cold:.2f}s warm={warm:.2f}s")
    assert warm < 5.0, f"warm single-sample wall {warm:.2f}s > 5s budget"

    # drop_pwv default + True behaviour
    assert ds.drop_pwv is False
    ds_drop = NPJDataset(starts=[start], window_minutes=180, drop_pwv=True)
    s_drop = ds_drop[0]
    assert float(np.asarray(s_drop["pwv_grid"]).__abs__().sum()) == 0.0
    assert float(np.asarray(s_drop["pwv_mask"]).sum()) == 0.0
    print("[self-check 2] drop_pwv=True zeros pwv_grid and pwv_mask")

    radar = np.asarray(sample["radar"])
    rmask = np.asarray(sample["radar_mask"])
    print(f"[radar] shape={radar.shape} dtype={radar.dtype} "
          f"min={radar.min():.2f} max={radar.max():.2f} "
          f"mean={radar.mean():.3f} frame_valid={rmask.mean():.3f} "
          f"sample_valid={sample['meta'].get('radar_sample_valid')}")
    rr = dbz_to_rainrate(radar)
    print(f"  -> rainrate mm/h max={rr.max():.2f}")
    assert radar.max() > 30, "Expected strong reflectivity on 07-28 12:00 case"

    pwv = np.asarray(sample["pwv_grid"])
    pwv_m = np.asarray(sample["pwv_mask"])
    print(f"[pwv] shape={pwv.shape} valid_frac={pwv_m.mean():.3f} "
          f"per-station valid_frac>=0.7: {(pwv_m.mean(axis=0)>=0.7).mean():.3f}")

    sta = np.asarray(sample["station_grid"])
    sta_m = np.asarray(sample["station_mask"])
    print(f"[station] shape={sta.shape} valid_frac={sta_m.mean():.3f}")

    e = sample["era5"]
    for k, v in e.items():
        arr = np.asarray(v)
        print(f"[era5/{k}] shape={arr.shape} min={arr.min():.3f} "
              f"max={arr.max():.3f}")
        assert arr.shape[-2:] == (661, 701), "ERA5 not interp'd to radar grid"

    print(f"[time] {sample['time']}  meta={sample['meta']}")
    print("OK")


if __name__ == "__main__":
    main()
