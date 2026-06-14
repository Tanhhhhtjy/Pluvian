"""Pooled (neighborhood-tolerant) CSI diagnostic, inspired by exPreCast (ICLR 2026).

For each checkpoint and threshold, compute full-set CSI at max-pool sizes
{1, 4, 16}. Pool size 1 is the standard exact-grid CSI. Larger pools apply
max-pooling to both prediction and target before scoring, giving credit when a
strong cell is predicted in the right neighborhood but spatially displaced.

This separates two failure modes behind a csi30 drop:
  - the strong core vanished (pooled CSI stays low at all pool sizes), vs.
  - the strong core is present but displaced (pooled CSI recovers as pool grows).

Read-only: writes a JSON + Markdown report, never touches training artifacts.

Usage:
    python scripts/eval_pooled_csi.py \
        --ckpt ab3=ckpt/ablation_3_era5_p7c/best.pt \
        --ckpt ab3b=ckpt/ablation_3b_era5_budget_p7c/best.pt \
        --split event_test --split test_robust \
        --out_dir ckpt/figures/pooled_csi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.data_loader import NPJDataset
from scripts.eval import _load_ckpt
from scripts.train import _starts_for_splits, _collate, split_batch, _to_device


def _pooled_csi_counts(pred, target, thr, pool):
    """Full-set (hits, fa, miss) at threshold thr after max-pool of size `pool`."""
    p = (pred >= thr).float()
    t = (target >= thr).float()
    if pool > 1:
        p = F.max_pool2d(p, kernel_size=pool, stride=pool, ceil_mode=True)
        t = F.max_pool2d(t, kernel_size=pool, stride=pool, ceil_mode=True)
    pb = p > 0.5
    tb = t > 0.5
    hits = int((pb & tb).sum().item())
    fa = int((pb & ~tb).sum().item())
    miss = int((~pb & tb).sum().item())
    return hits, fa, miss


def _eval_one(ckpt_path, split, thresholds, pools, device):
    model, cfg = _load_ckpt(ckpt_path, device)
    manifest = REPO_ROOT / cfg["data"]["manifest"]
    starts = _starts_for_splits(manifest, [split])
    drop_pwv = cfg["data"]["drop_pwv_val"]
    load_mfd = bool(cfg["model"].get("mfd_channel_enabled", False))
    load_era5 = bool(cfg["model"].get("era5_enabled", True)) or load_mfd
    ds = NPJDataset(
        manifest=manifest, window_minutes=cfg["data"]["window_minutes"],
        starts=starts, drop_pwv=drop_pwv, load_mfd=load_mfd, load_era5=load_era5,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=cfg["data"]["num_workers"], collate_fn=_collate)
    T_in = cfg["model"]["input_frames"]
    T_out = cfg["model"]["forecast_frames"]
    mfd_ch = cfg["model"]["mfd_channels"] if cfg["model"]["mfd_channel_enabled"] else 0

    totals = {(thr, pool): {"hits": 0, "fa": 0, "miss": 0}
              for thr in thresholds for pool in pools}
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            model_in, rain_tgt, _ = split_batch(batch, T_in, T_out, mfd_ch)
            out = model(model_in)
            rain_pred = out["rain_pred"].float()
            if rain_pred.shape[-2:] != rain_tgt.shape[-2:]:
                Ht, Wt = rain_tgt.shape[-2:]
                rain_pred = rain_pred[..., :Ht, :Wt].contiguous()
            rain_tgt_f = rain_tgt.float()
            B, T = rain_pred.shape[0], rain_pred.shape[1]
            p2 = rain_pred.reshape(B * T, 1, *rain_pred.shape[-2:])
            t2 = rain_tgt_f.reshape(B * T, 1, *rain_tgt_f.shape[-2:])
            for thr in thresholds:
                for pool in pools:
                    h, fa, m = _pooled_csi_counts(p2, t2, thr, pool)
                    ct = totals[(thr, pool)]
                    ct["hits"] += h; ct["fa"] += fa; ct["miss"] += m
    result = {}
    for (thr, pool), ct in totals.items():
        denom = ct["hits"] + ct["fa"] + ct["miss"]
        result[f"csi{int(thr)}_pool{pool}"] = ct["hits"] / denom if denom > 0 else float("nan")
    result["n_windows"] = len(ds)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="name=path, repeatable")
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 10.0, 30.0])
    ap.add_argument("--pools", type=int, nargs="+", default=[1, 4, 16])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = dict(kv.split("=", 1) for kv in args.ckpt)

    report = {}
    for split in args.split:
        report[split] = {}
        for name, path in ckpts.items():
            print(f"[pooled-csi] {name} split={split}")
            report[split][name] = _eval_one(path, split, args.thresholds, args.pools, device)

    out_json = args.out_dir / "pooled_csi.json"
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# Pooled CSI (neighborhood-tolerant) diagnostic\n"]
    for split in args.split:
        lines.append(f"\n## {split}\n")
        header = "| model | " + " | ".join(
            f"csi{int(thr)}_p{pool}" for thr in args.thresholds for pool in args.pools) + " |"
        sep = "|" + "---|" * (1 + len(args.thresholds) * len(args.pools))
        lines.append(header)
        lines.append(sep)
        for name in ckpts:
            r = report[split][name]
            cells = " | ".join(
                f"{r[f'csi{int(thr)}_pool{pool}']:.4f}"
                for thr in args.thresholds for pool in args.pools)
            lines.append(f"| {name} | {cells} |")
    out_md = args.out_dir / "pooled_csi.md"
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[pooled-csi] wrote {out_json} and {out_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
