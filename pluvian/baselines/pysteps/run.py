"""pysteps STEPS nowcasting baseline runner.

Runs pysteps STEPS (Bowler et al. 2006 / Pulkkinen et al. 2019, GMD)
on our manifest splits and writes a metric.json compatible with our
existing ckpt/eval/*/metric.json schema.

This is a non-DL baseline; no training. Pure inference: for each
window we feed pysteps the 12 history frames, get 18 forecast frames
(ensemble mean), and compute the same full-set CSI/FSS/MAE as
scripts/eval.py.

Usage:
    python -m pluvian.baselines.pysteps.run \\
        --split event_test \\
        --out_dir ckpt/eval/baseline_pysteps_steps \\
        [--n_ens 12] [--kmperpixel 1.0] [--timestep 6] [--n_workers 8]

Output:
    <out_dir>/<split>/metric.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow `python -m pluvian.baselines.pysteps.run` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.data_loader import NPJDataset  # noqa: E402

# pysteps imports
from pysteps.motion.lucaskanade import dense_lucaskanade  # noqa: E402
from pysteps.nowcasts.steps import forecast as steps_forecast  # noqa: E402

# Reuse train.py CSI/FSS helpers (full-set accumulator-compatible).
from scripts.train import (  # noqa: E402
    _csi_counts, _fss_components, _fss_binary,
)

THRESHOLDS = (1.0, 5.0, 10.0, 30.0)
FSS_NBRS = (3, 11)


def _starts_for_split(manifest_path: Path, splits: list[str]) -> list:
    """Reproduce scripts/train.py::_starts_for_splits behaviour."""
    import pandas as pd
    mf = pd.read_csv(manifest_path)
    mf = mf[mf["split"].isin(splits)]
    out = []
    for _, row in mf.iterrows():
        day = pd.Timestamp(row["date"])
        if "starts_per_day" in row and not pd.isna(row["starts_per_day"]):
            spd = int(row["starts_per_day"])
        else:
            spd = 8
        if spd <= 0:
            continue
        step_h = 24.0 / spd
        for k in range(spd):
            out.append(day + pd.Timedelta(hours=step_h * k))
    return [t.strftime("%Y-%m-%dT%H:%M:%S") for t in out]


def _pysteps_forecast_one(
    history: np.ndarray,   # (T_in, H, W) mm/h
    n_forecast: int,
    n_ens: int,
    kmperpixel: float,
    timestep_min: int,
) -> np.ndarray:
    """Run pysteps STEPS for a single sample. Returns (T_out, H, W) ensemble mean."""
    # pysteps wants float64 typically; cast.
    history = history.astype(np.float64)
    # Replace NaN / inf with 0 (no-rain).
    history = np.nan_to_num(history, nan=0.0, posinf=0.0, neginf=0.0)
    # Compute motion field from last few history frames.
    motion = dense_lucaskanade(history)
    # Forecast.
    forecast = steps_forecast(
        precip=history,
        velocity=motion,
        timesteps=n_forecast,
        n_ens_members=n_ens,
        n_cascade_levels=6,
        precip_thr=0.1,        # mm/h rain detection threshold
        kmperpixel=kmperpixel,
        timestep=timestep_min,
        seed=42,
        num_workers=1,
        vel_pert_method=None,  # disable velocity perturbations for speed
        mask_method="incremental",
        noise_method="nonparametric",
        measure_time=False,
        return_output=True,
    )
    # forecast: (n_ens, T_out, H, W)
    ens_mean = np.mean(forecast, axis=0)
    # NaN-safe: any unresolved pixel becomes 0.
    ens_mean = np.nan_to_num(ens_mean, nan=0.0, posinf=0.0, neginf=0.0)
    return ens_mean.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, action="append",
                    help="manifest split label, can be repeated")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "pipeline" / "manifest.csv")
    ap.add_argument("--n_ens", type=int, default=12,
                    help="number of pysteps ensemble members (we take the mean)")
    ap.add_argument("--kmperpixel", type=float, default=1.0,
                    help="approx grid spacing in km (0.01 deg ≈ 1.1 km)")
    ap.add_argument("--timestep_min", type=int, default=6,
                    help="minutes per radar frame")
    ap.add_argument("--n_input", type=int, default=12)
    ap.add_argument("--n_forecast", type=int, default=18)
    ap.add_argument("--limit", type=int, default=0,
                    help="only run this many windows (0 = all). For dev.")
    ap.add_argument("--n_workers", type=int, default=1,
                    help="parallel pysteps workers (multiprocessing). default 1.")
    args = ap.parse_args()

    splits = args.split
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n=== pysteps STEPS on split={split} ===")
        starts = _starts_for_split(args.manifest, [split])
        if args.limit:
            starts = starts[: args.limit]
        print(f"  windows: {len(starts)}  n_workers: {args.n_workers}")

        # Accumulators.
        csi_totals = {int(t): {"hits": 0, "fa": 0, "miss": 0} for t in THRESHOLDS}
        fss_totals = {nbr: {"num": 0.0, "den": 0.0} for nbr in FSS_NBRS}
        mae_acc = 0.0
        mse_acc = 0.0
        n_pix = 0
        n_done = 0
        n_err = 0
        t0 = time.time()

        worker_args = [
            {
                "idx": idx,
                "starts_tuple": tuple(starts),
                "n_input": args.n_input,
                "n_forecast": args.n_forecast,
                "n_ens": args.n_ens,
                "kmperpixel": args.kmperpixel,
                "timestep_min": args.timestep_min,
                "thresholds": tuple(THRESHOLDS),
                "fss_nbrs": tuple(FSS_NBRS),
            }
            for idx in range(len(starts))
        ]

        # Pool size: cap at n_workers but no larger than len(starts).
        nw = max(1, min(args.n_workers, len(starts)))
        if nw == 1:
            # Avoid multiprocessing entirely when single worker.
            from pluvian.baselines.pysteps.worker import worker as _w
            results_iter = (_w(wa) for wa in worker_args)
        else:
            from pluvian.baselines.pysteps.worker import worker as _w
            # Use spawn to avoid fork-related FFT/state issues.
            import multiprocessing as mp
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(nw)
            results_iter = pool.imap_unordered(_w, worker_args, chunksize=1)

        try:
            for res in results_iter:
                if res.get("status") != "ok":
                    n_err += 1
                    if n_err <= 5:
                        print(f"  [skip] idx={res.get('idx')} err={res.get('err')}")
                    continue
                mae_acc += res["mae_sum"]
                mse_acc += res["mse_sum"]
                n_pix += res["n_pix"]
                for thr, (h, fa, m) in res["csi"].items():
                    ct = csi_totals[int(thr)]
                    ct["hits"] += h
                    ct["fa"] += fa
                    ct["miss"] += m
                for nbr, (num, den) in res["fss"].items():
                    fss_totals[int(nbr)]["num"] += num
                    fss_totals[int(nbr)]["den"] += den
                n_done += 1
                if n_done % 10 == 0 or n_done == len(starts):
                    el = time.time() - t0
                    eta = el / n_done * (len(starts) - n_done)
                    print(f"  [{n_done}/{len(starts)}] elapsed={el:.1f}s  "
                          f"sec/window={el / n_done:.2f}  ETA={eta:.0f}s  err={n_err}")
        finally:
            if nw > 1:
                pool.close()
                pool.join()

        # Aggregate.
        mae = mae_acc / n_pix
        weighted_mse = mse_acc / n_pix
        results: dict = {
            "mae": mae,
            "weighted_mse": weighted_mse,
            "n_windows": n_done,
            "n_input": args.n_input,
            "n_forecast": args.n_forecast,
            "n_ens": args.n_ens,
            "baseline": "pysteps_steps_v1",
            "split": [split],
        }
        for t_int, ct in csi_totals.items():
            denom = ct["hits"] + ct["fa"] + ct["miss"]
            results[f"csi_{t_int}mm"] = (ct["hits"] / denom) if denom > 0 else float("nan")
        for nbr, ft in fss_totals.items():
            results[f"fss_{nbr}px"] = (1.0 - ft["num"] / ft["den"]) if ft["den"] > 0 else float("nan")

        out_sub = out_root / split
        out_sub.mkdir(parents=True, exist_ok=True)
        with open(out_sub / "metric.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote {out_sub / 'metric.json'}")
        print(f"  summary: csi_1mm={results.get('csi_1mm', float('nan')):.4f} "
              f"csi_10mm={results.get('csi_10mm', float('nan')):.4f} "
              f"csi_30mm={results.get('csi_30mm', float('nan')):.4f} "
              f"mae={results['mae']:.4f}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
