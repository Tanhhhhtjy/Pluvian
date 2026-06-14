"""Find the event_test start_time where ab3 most outperforms ab3b on >30mm/h.

For each window in the event_test split we run both ab3 and ab3b checkpoints,
accumulate hit/false-alarm/miss counts at 30 mm/h across all forecast leads,
compute per-window CSI@30 for each model, and report the windows where
(ab3_csi30 - ab3b_csi30) is largest — these are the cases that visually
showcase "ab3b smears the core, ab3 keeps it".

Usage:
    CUDA_VISIBLE_DEVICES=2 \
    /home/WuMingrui/miniconda3/envs/npj/bin/python \
        scripts/find_case_study_window.py \
        --ab3 ckpt/ablation_3_era5_p7c/best.pt \
        --ab3b ckpt/ablation_3b_era5_budget_p7c/best.pt \
        --split event_test \
        --out_csv eval/case_study_search.csv

Read-only: never touches the checkpoints or evaluation outputs of the
canonical pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.data_loader import NPJDataset  # noqa: E402

from scripts.train import (  # noqa: E402
    _starts_for_splits,
    _collate,
    _to_device,
    _csi_counts,
    split_batch,
)
from scripts.eval import _load_ckpt  # noqa: E402


@torch.no_grad()
def _forward_one(model, batch, T_in, T_out, mfd_ch):
    model_in, rain_tgt, _ = split_batch(batch, T_in, T_out, mfd_ch)
    out = model(model_in)
    rain_pred = out["rain_pred"].float()
    if rain_pred.shape[-2:] != rain_tgt.shape[-2:]:
        Ht, Wt = rain_tgt.shape[-2:]
        rain_pred = rain_pred[..., :Ht, :Wt].contiguous()
    return rain_pred, rain_tgt.float()


def _csi(hits: int, fa: int, miss: int) -> float:
    denom = hits + fa + miss
    return hits / denom if denom > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab3", type=Path, required=True)
    ap.add_argument("--ab3b", type=Path, required=True)
    ap.add_argument("--split", type=str, default="event_test")
    ap.add_argument("--out_csv", type=Path,
                    default=REPO_ROOT / "eval/case_study_search.csv")
    ap.add_argument("--thr", type=float, default=30.0,
                    help="rain-rate threshold (mm/h) defining the strong core")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[load] ab3  <- {args.ab3}")
    model_a, cfg_a = _load_ckpt(args.ab3, device)
    print(f"[load] ab3b <- {args.ab3b}")
    model_b, cfg_b = _load_ckpt(args.ab3b, device)

    # Use ab3's data config; both should share manifest/window.
    cfg = cfg_a
    manifest = REPO_ROOT / cfg["data"]["manifest"]
    starts = _starts_for_splits(manifest, [args.split])

    # Both ab3 and ab3b need ERA5; load it once and feed both.
    load_mfd = bool(cfg["model"].get("mfd_channel_enabled", False)) or \
        bool(cfg_b["model"].get("mfd_channel_enabled", False))
    load_era5 = True

    ds = NPJDataset(
        manifest=manifest, window_minutes=cfg["data"]["window_minutes"],
        starts=starts, drop_pwv=cfg["data"]["drop_pwv_val"],
        load_mfd=load_mfd, load_era5=load_era5,
    )
    print(f"[ds] split={args.split} windows={len(ds)}")

    loader = DataLoader(
        ds, batch_size=1, shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=_collate,
    )

    T_in_a = cfg_a["model"]["input_frames"]
    T_out_a = cfg_a["model"]["forecast_frames"]
    mfd_ch_a = cfg_a["model"]["mfd_channels"] if cfg_a["model"]["mfd_channel_enabled"] else 0
    T_in_b = cfg_b["model"]["input_frames"]
    T_out_b = cfg_b["model"]["forecast_frames"]
    mfd_ch_b = cfg_b["model"]["mfd_channels"] if cfg_b["model"]["mfd_channel_enabled"] else 0

    rows = []
    for i, batch in enumerate(loader):
        batch = _to_device(batch, device)
        t_str = batch["time"][0] if isinstance(batch["time"], list) else str(batch["time"])

        pred_a, tgt = _forward_one(model_a, batch, T_in_a, T_out_a, mfd_ch_a)
        pred_b, tgt_b = _forward_one(model_b, batch, T_in_b, T_out_b, mfd_ch_b)
        # Sanity: targets should match (same input window).
        assert tgt.shape == tgt_b.shape, f"target shape mismatch at {t_str}"

        ha, faa, ma = _csi_counts(pred_a, tgt, args.thr)
        hb, fab, mb = _csi_counts(pred_b, tgt, args.thr)
        csi_a = _csi(ha, faa, ma)
        csi_b = _csi(hb, fab, mb)

        # GT strong-core mass: per-lead count of >= thr pixels, take max over leads.
        gt_bin = (tgt >= args.thr)
        # tgt shape (B=1, T_out, H, W); count over (H,W) for each lead, then max over leads.
        per_lead = gt_bin.reshape(gt_bin.shape[1], -1).sum(dim=1)
        gt_max_core = int(per_lead.max().item())
        gt_total_core = int(gt_bin.sum().item())

        score = (csi_a if csi_a == csi_a else 0.0) - (csi_b if csi_b == csi_b else 0.0)

        rows.append(dict(
            idx=i,
            start_time=t_str,
            csi30_ab3=csi_a,
            csi30_ab3b=csi_b,
            score_window=score,
            gt_max_core_px=gt_max_core,
            gt_total_core_px=gt_total_core,
            ab3_hits=ha, ab3_fa=faa, ab3_miss=ma,
            ab3b_hits=hb, ab3b_fa=fab, ab3b_miss=mb,
        ))
        if (i + 1) % 8 == 0:
            print(f"  [{i+1}/{len(ds)}] {t_str}  score={score:+.4f}  "
                  f"ab3={csi_a:.3f} ab3b={csi_b:.3f}  gt_core={gt_max_core}")

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[write] {args.out_csv}")

    # Top by score, but only over windows with non-trivial GT core.
    min_core = 50
    cand = df[df["gt_max_core_px"] >= min_core].copy()
    print(f"\n[filter] windows with gt_max_core_px >= {min_core}: {len(cand)} / {len(df)}")
    top = cand.sort_values("score_window", ascending=False).head(10)
    print("\nTop 10 windows by (ab3_csi30 - ab3b_csi30):")
    print(top[["start_time", "score_window", "csi30_ab3", "csi30_ab3b",
               "gt_max_core_px", "gt_total_core_px"]].to_string(index=False))


if __name__ == "__main__":
    main()
