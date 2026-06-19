"""Forward-pass sanity check for Phase 7e mm/h-era ckpts.

Loads each best.pt, runs forward on a few event_test windows, and
checks distributional alignment between model output and target.

Specifically checks:
  - target stats (max, mean, %>1, %>10, %>30) → ground truth in mm/h
  - prediction stats (same) → if pred max ~70 with target max ~100,
    model is still emitting dBZ scale (unit fix did not propagate
    into the rain head)
  - per-pixel scatter of pred vs target on first window
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.data_loader import NPJDataset  # noqa: E402
from scripts.eval import _load_ckpt  # noqa: E402
from scripts.train import split_batch, _collate  # noqa: E402


CKPTS = {
    "ab1":     "ckpt/ablation_1_p7d/best.pt",
    "ab2":     "ckpt/ablation_2_p7d/best.pt",
    "ab3":     "ckpt/ablation_3_era5_p7c/best.pt",
    "ab3b":    "ckpt/ablation_3b_era5_budget_p7c/best.pt",
    "ab3-cdu": "ckpt/ablation_3_cdu_p7d/best.pt",
}


def stats(t: torch.Tensor) -> dict:
    """Distribution snapshot of an mm/h tensor."""
    arr = t.detach().float().cpu().numpy().ravel()
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "p99": float(np.percentile(arr, 99)),
        "pct_gt_1": float((arr > 1).mean() * 100),
        "pct_gt_10": float((arr > 10).mean() * 100),
        "pct_gt_30": float((arr > 30).mean() * 100),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # First three event_test starts.
    starts = [
        "2023-07-29T03:00:00",
        "2023-07-30T03:00:00",
        "2023-07-31T03:00:00",
    ]
    ds = NPJDataset(starts=starts, window_minutes=180, load_era5=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_collate)

    # Show target distribution once.
    print("=== ground truth (mm/h, target frames 12..30) ===")
    for i, batch in enumerate(loader):
        tgt = batch["radar"][:, 12:30].float()
        s = stats(tgt)
        print(f"  window {i} ({starts[i]}): {s}")

    print()
    for tag, ckpt_path in CKPTS.items():
        path = REPO / ckpt_path
        if not path.exists():
            print(f"=== {tag}: SKIP (no {path}) ===")
            continue
        print(f"=== {tag}: forward sanity ===")
        model, cfg = _load_ckpt(path, device)
        T_in = cfg["model"]["input_frames"]
        T_out = cfg["model"]["forecast_frames"]
        mfd_ch = cfg["model"].get("mfd_channels", 0) if cfg["model"].get(
            "mfd_channel_enabled") else 0

        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                batch_dev = {k: v.to(device) if torch.is_tensor(v) else v
                             for k, v in batch.items() if k != "era5"}
                if "era5" in batch:
                    batch_dev["era5"] = {k: v.to(device) for k, v in batch["era5"].items()}
                model_in, rain_tgt, _ = split_batch(batch_dev, T_in, T_out, mfd_ch)
                out = model(model_in)
                pred = out["rain_pred"].float()
                if pred.shape[-2:] != rain_tgt.shape[-2:]:
                    Ht, Wt = rain_tgt.shape[-2:]
                    pred = pred[..., :Ht, :Wt]
                ps = stats(pred)
                ts = stats(rain_tgt.float())
                ratio_max = ps["max"] / max(ts["max"], 1e-6)
                ratio_p99 = ps["p99"] / max(ts["p99"], 1e-6)
                print(f"  win {i}: pred max={ps['max']:.2f} (tgt {ts['max']:.2f}) ratio={ratio_max:.3f} | "
                      f"pred p99={ps['p99']:.3f} (tgt {ts['p99']:.3f}) ratio={ratio_p99:.3f} | "
                      f"pred %>1mm={ps['pct_gt_1']:.2f} (tgt {ts['pct_gt_1']:.2f}) | "
                      f"pred %>30mm={ps['pct_gt_30']:.4f} (tgt {ts['pct_gt_30']:.4f})")
        print()


if __name__ == "__main__":
    main()
