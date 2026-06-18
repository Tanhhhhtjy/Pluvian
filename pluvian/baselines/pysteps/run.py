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
        [--n_ens 12] [--kmperpixel 1.0] [--timestep 6]

Output:
    <out_dir>/<split>/metric.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    args = ap.parse_args()

    splits = args.split
    out_root = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n=== pysteps STEPS on split={split} ===")
        starts = _starts_for_split(args.manifest, [split])
        if args.limit:
            starts = starts[: args.limit]
        print(f"  windows: {len(starts)}")
        ds = NPJDataset(
            starts=starts,
            window_minutes=(args.n_input + args.n_forecast) * args.timestep_min,
            load_era5=False,
        )

        # Accumulators.
        csi_totals = {int(t): {"hits": 0, "fa": 0, "miss": 0} for t in THRESHOLDS}
        # FSS using full-set numerator/denominator accumulator (mirror train.py).
        fss_totals = {nbr: {"num": 0.0, "den": 0.0} for nbr in FSS_NBRS}
        mae_acc = 0.0
        mse_acc = 0.0
        n_pix = 0
        n_done = 0
        t0 = time.time()

        for idx in range(len(ds)):
            sample = ds[idx]
            radar = sample["radar"]  # (T, H, W) torch.float32 mm/h
            radar = radar.numpy() if hasattr(radar, "numpy") else np.asarray(radar)
            history = radar[: args.n_input]                       # (12, H, W)
            target = radar[args.n_input : args.n_input + args.n_forecast]  # (18, H, W)

            try:
                pred = _pysteps_forecast_one(
                    history,
                    n_forecast=args.n_forecast,
                    n_ens=args.n_ens,
                    kmperpixel=args.kmperpixel,
                    timestep_min=args.timestep_min,
                )
            except Exception as e:
                print(f"  [skip] window {idx} pysteps failed: {e!r}")
                continue

            pred_t = torch.from_numpy(pred).float().unsqueeze(0)       # (1, T, H, W)
            tgt_t = torch.from_numpy(target).float().unsqueeze(0)      # (1, T, H, W)

            mae_acc += float((pred_t - tgt_t).abs().sum().item())
            mse_acc += float(((pred_t - tgt_t) ** 2).sum().item())
            n_pix += int(pred_t.numel())

            for thr in THRESHOLDS:
                h, fa, m = _csi_counts(pred_t, tgt_t, float(thr))
                ct = csi_totals[int(thr)]
                ct["hits"] += h
                ct["fa"] += fa
                ct["miss"] += m

            # FSS at csi_1mm threshold (mirrors paper figure default).
            for nbr in FSS_NBRS:
                num, den = _fss_components(pred_t, tgt_t, 1.0, nbr)
                fss_totals[nbr]["num"] += float(num)
                fss_totals[nbr]["den"] += float(den)

            n_done += 1
            if n_done % 10 == 0:
                el = time.time() - t0
                print(f"  [{n_done}/{len(ds)}] elapsed={el:.1f}s  "
                      f"sec/window={el / n_done:.2f}")

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
