"""Canonical 4-row GT / ab1 / ab2 / diff comparison figure.

Replicates the figures/event_23_7_ab1_vs_ab2.png paradigm against the
Phase-7c checkpoints (ablation_{1,2}_new_7b_p7c). For one start time, four
lead columns (+18/+36/+72/+108 min):

  row 1  GT            radar ground truth (dBZ, NWS colormap)
  row 2  ab1-new       radar-only forecast
  row 3  ab2-new       radar+PWV (gated fusion) forecast
  row 4  diff          ab2 - ab1 (dBZ, RdBu_r, +-30)

Usage:
    python scripts/plot_ab1_vs_ab2.py CKPT_AB1 CKPT_AB2 OUT_PNG START
"""
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from pipeline.data_loader import NPJDataset
from model.pluvian import Pluvian
from scripts.plot_prediction import DBZ_CMAP, DBZ_NORM

T_IN, T_OUT = 12, 18
LAT0, LAT1, LON0, LON1 = 36.0, 42.6, 113.0, 120.0
LEAD_IDX = [2, 5, 11, 17]  # ~ +18, +36, +72, +108 min


def build_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or ck.get("cfg")
    m = cfg["model"]
    model = Pluvian(
        radar_channels=m["radar_channels"], pwv_enabled=m["pwv_enabled"],
        pwv_concat_only=m["pwv_concat_only"], era5_enabled=m["era5_enabled"],
        mfd_channel_enabled=m["mfd_channel_enabled"],
        forecast_frames=m["forecast_frames"], input_frames=m["input_frames"],
        hidden_dim=m["hidden_dim"], n_layers=m["n_layers"], n_heads=m["n_heads"],
        n_era5_vars=m["n_era5_vars"], n_era5_levels=m["n_era5_levels"],
        mfd_channels=m["mfd_channels"], gated_fusion=m.get("gated_fusion", True),
        intensity_stratified=m.get("intensity_stratified", False),
        band_centers=tuple(m.get("band_centers", (0.0, 0.5, 4.5, 19.0, 50.0))),
    )
    model.load_state_dict(ck["model"], strict=False)
    return model.to(device).eval()


def make_batch(sample, device):
    batch = {
        "radar": sample["radar"][:T_IN].unsqueeze(0).float(),
        "pwv_grid": sample["pwv_grid"][: T_IN // 5].unsqueeze(0).float(),
        "pwv_mask": sample["pwv_mask"][: T_IN // 5].unsqueeze(0).float(),
        "pwv_coords": sample["pwv_coords"].unsqueeze(0).float(),
        "station_grid": sample["station_grid"][: T_IN // 10].unsqueeze(0).float(),
        "station_mask": sample["station_mask"][: T_IN // 10].unsqueeze(0).float(),
        "station_coords": sample["station_coords"].unsqueeze(0).float(),
        "era5": {k: v[:T_IN].unsqueeze(0).float() for k, v in sample["era5"].items()},
    }

    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return x

    return to_dev(batch)


def main(ckpt_ab1, ckpt_ab2, out_png, start="2023-08-08T06:00:00"):
    print(f"=== ab1 vs ab2 4-row figure  start={start} ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NPJDataset(manifest=Path(__file__).resolve().parent.parent / "pipeline" / "manifest.csv",
                    starts=[start], load_mfd=False, window_minutes=180)
    sample = ds[0]
    radar_gt = sample["radar"][T_IN:T_IN + T_OUT].numpy()  # (T_OUT,H,W) dBZ
    Ht, Wt = radar_gt.shape[-2:]

    ab1 = build_model(ckpt_ab1, device)
    ab2 = build_model(ckpt_ab2, device)
    t0 = time.time()
    with torch.no_grad():
        pred_ab1 = ab1(make_batch(sample, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]
        pred_ab2 = ab2(make_batch(sample, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]
    print(f"  forward: {time.time()-t0:.1f}s  ab1 max={pred_ab1.max():.1f}  ab2 max={pred_ab2.max():.1f}")

    extent = [LON0, LON1, LAT0, LAT1]
    n_cols = len(LEAD_IDX)
    fig, axes = plt.subplots(4, n_cols, figsize=(3.5 * n_cols, 12.5))
    mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11})

    row_lbl = ["Ground truth", "ab1-new (radar only)", "ab2-new (radar+PWV)", "ab2 - ab1"]
    for j, idx in enumerate(LEAD_IDX):
        lab = f"+{(idx + 1) * 6} min"
        gt = radar_gt[idx]
        p1 = pred_ab1[idx]
        p2 = pred_ab2[idx]
        diff = p2 - p1

        for r, (data, title) in enumerate([
            (gt, f"GT  {lab}"),
            (p1, f"ab1  {lab}"),
            (p2, f"ab2  {lab}"),
        ]):
            ax = axes[r, j]
            ax.imshow(data, origin="lower", extent=extent, cmap=DBZ_CMAP, norm=DBZ_NORM, aspect="auto")
            ax.set_title(title, fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row_lbl[r], fontsize=11)

        ax = axes[3, j]
        im2 = ax.imshow(diff, origin="lower", extent=extent, cmap="RdBu_r",
                        vmin=-30, vmax=30, aspect="auto")
        ax.set_title("diff", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(row_lbl[3] + " (dBZ)", fontsize=11)

    cbar_ax = fig.add_axes([0.92, 0.40, 0.012, 0.40])
    fig.colorbar(plt.cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_CMAP), cax=cbar_ax, label="dBZ")
    cbar_ax2 = fig.add_axes([0.92, 0.10, 0.012, 0.15])
    fig.colorbar(im2, cax=cbar_ax2, label="diff (dBZ)")

    fig.suptitle(f"Pluvian  GT vs ab1-new vs ab2-new  start={start}", y=0.995, fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.91, 0.985])
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print("  saved:", out_png)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], a[4] if len(a) > 4 else "2023-08-08T06:00:00")
