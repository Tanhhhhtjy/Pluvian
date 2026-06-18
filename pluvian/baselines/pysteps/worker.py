"""Worker function for parallel pysteps execution.

This module is imported in worker processes (via multiprocessing.Pool).
Each worker computes pysteps forecast for one window and returns the
hit/fa/miss counts + FSS components + per-window error sums. The main
process accumulates these.

Lives in a separate module to avoid pickling issues with closures.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.data_loader import NPJDataset  # noqa: E402
from pysteps.motion.lucaskanade import dense_lucaskanade  # noqa: E402
from pysteps.nowcasts.steps import forecast as steps_forecast  # noqa: E402
from scripts.train import _csi_counts, _fss_components  # noqa: E402

# Module-level dataset cache (one per worker process). Lazy-initialized
# on first call so the main process doesn't need to pre-spawn.
_DS = None
_DS_KEY = None


def _get_ds(starts_tuple: tuple, n_frames_total: int) -> NPJDataset:
    """Per-worker NPJDataset, cached across calls in the same process."""
    global _DS, _DS_KEY
    key = (starts_tuple, n_frames_total)
    if _DS_KEY != key:
        _DS = NPJDataset(
            starts=list(starts_tuple),
            window_minutes=n_frames_total * 6,
            load_era5=False,
        )
        _DS_KEY = key
    return _DS


def worker(args: dict) -> dict:
    """Compute pysteps forecast for one window. Return metric components.

    args keys:
      idx              -- index into starts list
      starts_tuple     -- tuple of all start_time strings (for dataset init)
      n_input          -- e.g. 12
      n_forecast       -- e.g. 18
      n_ens            -- pysteps ensemble members
      kmperpixel       -- e.g. 1.0
      timestep_min     -- e.g. 6
      thresholds       -- tuple of CSI thresholds
      fss_nbrs         -- tuple of FSS neighbourhood sizes
    Returns dict with: idx, status, and if ok: csi[thr], fss_num[nbr],
                       fss_den[nbr], mae_sum, mse_sum, n_pix
    """
    idx = args["idx"]
    n_input = args["n_input"]
    n_forecast = args["n_forecast"]
    n_total = n_input + n_forecast

    ds = _get_ds(args["starts_tuple"], n_total)
    try:
        sample = ds[idx]
        radar = sample["radar"]
        radar = radar.numpy() if hasattr(radar, "numpy") else np.asarray(radar)
        history = radar[:n_input].astype(np.float64)
        target = radar[n_input:n_input + n_forecast]

        # Clean NaN/inf in input.
        history = np.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
        motion = dense_lucaskanade(history)
        fc = steps_forecast(
            precip=history,
            velocity=motion,
            timesteps=n_forecast,
            n_ens_members=args["n_ens"],
            n_cascade_levels=6,
            precip_thr=0.1,
            kmperpixel=args["kmperpixel"],
            timestep=args["timestep_min"],
            seed=42,
            num_workers=1,
            vel_pert_method=None,
            mask_method="incremental",
            noise_method="nonparametric",
            measure_time=False,
            return_output=True,
        )
        # fc: (n_ens, T_out, H, W) — take ensemble mean
        ens_mean = np.nanmean(fc, axis=0)
        ens_mean = np.nan_to_num(ens_mean, nan=0.0, posinf=0.0, neginf=0.0)
        pred = ens_mean.astype(np.float32)
    except Exception as e:
        return {"idx": idx, "status": "error", "err": repr(e)[:200]}

    pred_t = torch.from_numpy(pred).float().unsqueeze(0)
    tgt_t = torch.from_numpy(target).float().unsqueeze(0)

    out: dict = {"idx": idx, "status": "ok"}
    out["mae_sum"] = float((pred_t - tgt_t).abs().sum().item())
    out["mse_sum"] = float(((pred_t - tgt_t) ** 2).sum().item())
    out["n_pix"] = int(pred_t.numel())

    out["csi"] = {}
    for thr in args["thresholds"]:
        h, fa, m = _csi_counts(pred_t, tgt_t, float(thr))
        out["csi"][int(thr)] = (h, fa, m)
    out["fss"] = {}
    for nbr in args["fss_nbrs"]:
        num, den = _fss_components(pred_t, tgt_t, 1.0, nbr)
        out["fss"][int(nbr)] = (float(num), float(den))

    return out
