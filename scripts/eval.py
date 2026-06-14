"""Evaluation entry-point for a trained Pluvian checkpoint.

Usage:
    python scripts/eval.py --ckpt ckpt/ablation_5_budget_loss/best.pt \
        --out_dir eval/ablation_5_budget_loss [--split val] [--n_samples 4]

Loads checkpoint (which contains its config), runs the chosen split,
writes:
    <out_dir>/metric.json   — averaged metrics over the split
    <out_dir>/sample_*.png  — 4 sample predictions (or n_samples)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import Pluvian
from model.losses import WeightedMSE
from pipeline.data_loader import NPJDataset
from pipeline.utils import RADAR_LAT, RADAR_LON

# Re-use helpers from train.py
from scripts.train import (  # type: ignore  # noqa: E402
    _starts_for_splits, _collate, split_batch, _to_device,
    _csi, _csi_counts, _fss_binary, _fss_components, _crps_marginal,
)


def _load_ckpt(path: Path, device: torch.device):
    state = torch.load(path, map_location=device, weights_only=False)
    cfg = state["config"]
    m = cfg["model"]
    model = Pluvian(
        radar_channels=m["radar_channels"],
        pwv_enabled=m["pwv_enabled"],
        pwv_concat_only=m["pwv_concat_only"],
        era5_enabled=m["era5_enabled"],
        mfd_channel_enabled=m["mfd_channel_enabled"],
        forecast_frames=m["forecast_frames"],
        input_frames=m["input_frames"],
        hidden_dim=m["hidden_dim"],
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        n_era5_vars=m["n_era5_vars"],
        n_era5_levels=m["n_era5_levels"],
        mfd_channels=m["mfd_channels"],
        intensity_stratified=bool(m.get("intensity_stratified", False)),
        band_centers=tuple(m.get("band_centers", (0.0, 0.5, 4.5, 19.0, 50.0))),
        gated_fusion=bool(m.get("gated_fusion", False)),
        cdu_decoder=bool(m.get("cdu_decoder", False)),
    ).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, cfg


def _save_sample_png(pred: np.ndarray, target: np.ndarray, path: Path, title: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[plot] matplotlib unavailable, skipping {path.name}")
        return
    T = pred.shape[0]
    # show three forecast horizons: t+6, t+30, t+108 (idx 0, 4, T-1)
    idxs = [0, min(4, T - 1), T - 1]
    fig, axes = plt.subplots(2, len(idxs), figsize=(4 * len(idxs), 7))
    vmax = max(float(pred.max()), float(target.max()), 1.0)
    for col, ti in enumerate(idxs):
        axes[0, col].imshow(target[ti], origin="lower", vmin=0, vmax=vmax, cmap="turbo")
        axes[0, col].set_title(f"target t+{(ti+1)*6}min")
        axes[0, col].axis("off")
        axes[1, col].imshow(pred[ti], origin="lower", vmin=0, vmax=vmax, cmap="turbo")
        axes[1, col].set_title(f"pred t+{(ti+1)*6}min")
        axes[1, col].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--split", type=str, default=None,
                    help="manifest split label (default: cfg.data.splits_val)")
    ap.add_argument("--drop_pwv", action="store_true",
                    help="override drop_pwv to True for robustness eval")
    ap.add_argument("--n_samples", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_ckpt(args.ckpt, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = [args.split] if args.split else cfg["data"]["splits_val"]
    manifest = REPO_ROOT / cfg["data"]["manifest"]
    starts = _starts_for_splits(manifest, splits)
    drop_pwv = args.drop_pwv or cfg["data"]["drop_pwv_val"]
    load_mfd = bool(cfg["model"].get("mfd_channel_enabled", False))
    load_era5 = bool(cfg["model"].get("era5_enabled", True)) or load_mfd
    ds = NPJDataset(
        manifest=manifest, window_minutes=cfg["data"]["window_minutes"],
        starts=starts, drop_pwv=drop_pwv,
        load_mfd=load_mfd, load_era5=load_era5,
    )
    print(f"[eval] split={splits} windows={len(ds)} drop_pwv={drop_pwv}")

    bs = args.batch_size or cfg["data"]["batch_size"]
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=cfg["data"]["num_workers"],
                        collate_fn=_collate)

    T_in = cfg["model"]["input_frames"]
    T_out = cfg["model"]["forecast_frames"]
    mfd_ch = cfg["model"]["mfd_channels"] if cfg["model"]["mfd_channel_enabled"] else 0
    thresholds = (1.0, 5.0, 10.0, 30.0)
    fss_nbrs = (3, 11)
    # Phase 7a: full-set CSI/FSS via running totals (mirrors train.validate).
    csi_totals = {int(t): {"hits": 0, "fa": 0, "miss": 0} for t in thresholds}
    fss_totals = {nbr: {"num": 0.0, "den": 0.0} for nbr in fss_nbrs}
    agg: dict[str, list[float]] = {}
    agg["mae"] = []
    agg["crps"] = []
    data_loss = WeightedMSE().to(device)
    agg["weighted_mse"] = []

    sample_count = 0
    for batch in loader:
        batch = _to_device(batch, device)
        model_in, rain_tgt, _ = split_batch(batch, T_in, T_out, mfd_ch)
        out = model(model_in)
        rain_pred = out["rain_pred"].float()
        if rain_pred.shape[-2:] != rain_tgt.shape[-2:]:
            Ht, Wt = rain_tgt.shape[-2:]
            rain_pred = rain_pred[..., :Ht, :Wt].contiguous()
        rain_tgt_f = rain_tgt.float()

        agg["weighted_mse"].append(data_loss(rain_pred, rain_tgt_f).item())
        agg["mae"].append((rain_pred - rain_tgt_f).abs().mean().item())
        agg["crps"].append(_crps_marginal(rain_pred, rain_tgt_f))
        for thr in thresholds:
            h, fa, m = _csi_counts(rain_pred, rain_tgt_f, thr)
            ct = csi_totals[int(thr)]
            ct["hits"] += h; ct["fa"] += fa; ct["miss"] += m
        for nbr in fss_nbrs:
            num, den = _fss_components(rain_pred, rain_tgt_f, 1.0, nbr)
            fss_totals[nbr]["num"] += num
            fss_totals[nbr]["den"] += den

        # save up to n_samples sample images
        if sample_count < args.n_samples:
            B = rain_pred.shape[0]
            for b in range(B):
                if sample_count >= args.n_samples:
                    break
                t_str = batch["time"][b] if isinstance(batch["time"], list) else str(batch["time"])
                safe_t = t_str.replace(":", "").replace(" ", "_")
                _save_sample_png(
                    rain_pred[b].cpu().numpy(),
                    rain_tgt_f[b].cpu().numpy(),
                    args.out_dir / f"sample_{sample_count}_{safe_t}.png",
                    title=f"{cfg['name']} | {t_str}",
                )
                sample_count += 1

    metrics = {}
    for k, vs in agg.items():
        vs = [v for v in vs if v == v]
        metrics[k] = float(np.mean(vs)) if vs else float("nan")
    # Phase 7a: full-set CSI/FSS using accumulated counts.
    for thr_int, ct in csi_totals.items():
        denom = ct["hits"] + ct["fa"] + ct["miss"]
        metrics[f"csi_{thr_int}mm"] = ct["hits"] / denom if denom > 0 else float("nan")
    for nbr, ft in fss_totals.items():
        metrics[f"fss_{nbr}px"] = (
            1.0 - ft["num"] / ft["den"] if ft["den"] > 0 else float("nan"))
    metrics["n_windows"] = len(ds)
    metrics["ckpt"] = str(args.ckpt)
    metrics["split"] = splits

    with open(args.out_dir / "metric.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[metrics]", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
