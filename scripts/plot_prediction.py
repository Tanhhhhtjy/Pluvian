"""
Plot prediction visualization for ab1 (and later ab2) on the 23·7 case.

For one selected sample, show:
  - Input radar frames (last 3 of input window)
  - Ground truth future frames at several lead times
  - Pluvian predicted future frames at same lead times
"""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from pipeline.data_loader import NPJDataset
from model.pluvian import Pluvian


# Standard NWS-style dBZ colormap (composite reflectivity colors)
DBZ_COLORS = [
    (0.95, 0.95, 0.95),   # <5  dBZ — very light
    (0.70, 0.95, 0.85),   # 5
    (0.40, 0.85, 0.80),   # 10
    (0.20, 0.65, 0.95),   # 15
    (0.10, 0.40, 0.85),   # 20
    (0.05, 0.95, 0.45),   # 25
    (0.45, 0.95, 0.05),   # 30
    (0.95, 0.95, 0.05),   # 35
    (0.95, 0.65, 0.05),   # 40
    (0.95, 0.30, 0.05),   # 45
    (0.95, 0.05, 0.05),   # 50
    (0.65, 0.05, 0.55),   # 55+
]
DBZ_BOUNDS = [-5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 70]
DBZ_CMAP = mcolors.LinearSegmentedColormap.from_list("nws_dbz", DBZ_COLORS)
DBZ_NORM = mcolors.BoundaryNorm(DBZ_BOUNDS, DBZ_CMAP.N, extend="both")


def rain_to_dbz(rain_mmh):
    """Convert rain rate (mm/h) back to dBZ using inverse Z=300*R^1.4."""
    # Z = 300 * R^1.4 → dBZ = 10 log10(Z)
    rain = np.clip(rain_mmh, 1e-3, None)
    Z = 300.0 * rain ** 1.4
    return 10.0 * np.log10(Z)


def main(ckpt_path: str, out_path: str, start_time: str = "2023-07-29T02:00:00"):
    print(f"=== Plot prediction for {start_time} ===")
    print(f"  ckpt: {ckpt_path}")

    # Load dataset (1 sample at 23·7 event)
    ds = NPJDataset(
        manifest=Path(__file__).resolve().parent.parent / "pipeline" / "manifest.csv",
        starts=[start_time],
        load_mfd=False,
        window_minutes=180,
    )
    sample = ds[0]

    # Inputs: radar T=30 frames; T_in=12 used by model, T_out=18 are GT future
    radar_full = sample["radar"]  # (T=30, H, W) dBZ
    T_in = 12
    T_out = 18
    radar_in = radar_full[:T_in]
    radar_gt = radar_full[T_in:T_in + T_out]
    print(f"  radar full shape: {radar_full.shape}")
    print(f"  radar_in: {radar_in.shape}  max={radar_in.max():.1f} dBZ")
    print(f"  radar_gt: {radar_gt.shape}  max={radar_gt.max():.1f} dBZ")

    # Load model from checkpoint
    print(f"  loading checkpoint...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Try to read config out of ckpt (Phase 7a saves under 'config', older 'cfg')
    cfg = ckpt.get("config", None) or ckpt.get("cfg", None)
    if cfg is None:
        # Use ab1 base config defaults
        model = Pluvian(
            pwv_enabled=False,
            era5_enabled=False,
            mfd_channel_enabled=False,
            forecast_frames=18,
            input_frames=12,
            hidden_dim=192,
            n_layers=4,
            n_heads=8,
        )
    else:
        m = cfg.get("model", {})
        model = Pluvian(
            pwv_enabled=m.get("pwv_enabled", False),
            pwv_concat_only=m.get("pwv_concat_only", False),
            era5_enabled=m.get("era5_enabled", False),
            mfd_channel_enabled=m.get("mfd_channel_enabled", False),
            forecast_frames=m.get("forecast_frames", 18),
            input_frames=m.get("input_frames", 12),
            hidden_dim=m.get("hidden_dim", 192),
            n_layers=m.get("n_layers", 4),
            n_heads=m.get("n_heads", 8),
            gated_fusion=m.get("gated_fusion", False),
            intensity_stratified=m.get("intensity_stratified", False),
            band_centers=tuple(m.get("band_centers", (0.0, 0.5, 4.5, 19.0, 50.0))),
        )
    model.load_state_dict(ckpt["model"], strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    # Build batch
    def to_batch(x):
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0)
        if isinstance(x, dict):
            return {k: to_batch(v) for k, v in x.items()}
        return x

    batch = {
        "radar": radar_in.unsqueeze(0).float(),
        "pwv_grid": sample["pwv_grid"][:T_in // 5].unsqueeze(0).float(),
        "pwv_mask": sample["pwv_mask"][:T_in // 5].unsqueeze(0).float(),
        "pwv_coords": sample["pwv_coords"].unsqueeze(0).float(),
        "station_grid": sample["station_grid"][:T_in // 10].unsqueeze(0).float(),
        "station_mask": sample["station_mask"][:T_in // 10].unsqueeze(0).float(),
        "station_coords": sample["station_coords"].unsqueeze(0).float(),
        "era5": {k: v[:T_in].unsqueeze(0).float() for k, v in sample["era5"].items()},
    }

    # Forward
    print(f"  running forward (cpu, ~30s for full res)...")
    import time
    t0 = time.time()
    def _to_dev(x):
        if isinstance(x, torch.Tensor): return x.to(device)
        if isinstance(x, dict): return {k: _to_dev(v) for k, v in x.items()}
        return x
    batch = _to_dev(batch)
    with torch.no_grad():
        out = model(batch)
    print(f"  forward time: {time.time()-t0:.1f}s")

    rain_pred = out["rain_pred"][0].cpu()  # (T_out, H, W) — model output is in dBZ (target = radar dBZ)
    print(f"  rain_pred shape: {rain_pred.shape}  range [{rain_pred.min():.2f}, {rain_pred.max():.2f}]")

    # Model output is already in dBZ. Crop to GT shape.
    pred_dbz = rain_pred.numpy()
    Ht, Wt = radar_gt.shape[-2:]
    pred_dbz = pred_dbz[:, :Ht, :Wt]

    # Pick lead time indices to show: every 6 min, model predicts T_out frames
    # If T_out=18, frames are t+6,12,18,...,108 min. Pick a few.
    lead_idx = [2, 5, 11, 17]   # ≈ +18, +36, +72, +108 min
    lead_labels = [f"+{(i+1)*6} min" for i in lead_idx]

    # 4 columns × 3 rows: GT / Pred / Diff
    n_cols = len(lead_idx)
    fig, axes = plt.subplots(3, n_cols, figsize=(3.5 * n_cols, 9.5))

    # Style
    mpl.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11})

    # lat/lon for axis
    lat = np.linspace(36.0, 42.6, Ht)
    lon = np.linspace(113.0, 120.0, Wt)
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    for j, (idx, lab) in enumerate(zip(lead_idx, lead_labels)):
        gt = radar_gt[idx].numpy()
        pr = pred_dbz[idx]
        diff = pr - gt

        ax = axes[0, j]
        ax.imshow(gt, origin="lower", extent=extent, cmap=DBZ_CMAP, norm=DBZ_NORM, aspect="auto")
        ax.set_title(f"GT  {lab}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Ground truth", fontsize=11)

        ax = axes[1, j]
        ax.imshow(pr, origin="lower", extent=extent, cmap=DBZ_CMAP, norm=DBZ_NORM, aspect="auto")
        ax.set_title(f"Pluvian  {lab}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Model pred", fontsize=11)

        ax = axes[2, j]
        im2 = ax.imshow(diff, origin="lower", extent=extent, cmap="RdBu_r", vmin=-30, vmax=30, aspect="auto")
        ax.set_title(f"diff", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("Pred − GT (dBZ)", fontsize=11)

    # Add colorbars
    cbar_ax = fig.add_axes([0.91, 0.66, 0.012, 0.20])
    fig.colorbar(plt.cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_CMAP),
                 cax=cbar_ax, label="dBZ")
    cbar_ax2 = fig.add_axes([0.91, 0.10, 0.012, 0.20])
    fig.colorbar(im2, cax=cbar_ax2, label="diff (dBZ)")

    fig.suptitle(
        f"Pluvian ab1 (radar only) — 23·7 case start={start_time}",
        y=1.0, fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 0.90, 0.97])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main(
        ckpt_path="/root/autodl-tmp/pluvian/ckpt/ablation_1_radar_only/best.pt",
        out_path="/root/autodl-tmp/pluvian/figures/ab1_prediction_23p7.png",
        start_time="2023-07-29T02:00:00",
    )
