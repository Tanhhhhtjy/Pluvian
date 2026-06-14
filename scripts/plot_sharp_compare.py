"""Sharpness comparison: p7c ab2 baseline vs sharpA vs sharpB.

Two outputs per start time:
  1. cmp_<start>.png  — 4 rows (GT / baseline / sharpA / sharpB) x 4 lead cols,
     DBZ NWS colormap. Visual de-blur check.
  2. rapsd_<start>.png — radially-averaged power spectral density (RAPSD) of
     GT vs each model, averaged over all 18 lead frames, log-log. The honest
     blur diagnostic: a blurry model loses energy at high wavenumber k; a
     successful sharpen pushes the predicted spectrum back onto GT out to
     higher k. (pysteps / Harris et al. JAMES 2022.)

Usage:
    python scripts/plot_sharp_compare.py BASE_CKPT SHARPA_CKPT SHARPB_CKPT OUTDIR START
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
from scripts.plot_prediction import DBZ_CMAP, DBZ_NORM
from scripts.plot_ab1_vs_ab2 import build_model as _build_model_embedded, make_batch, T_IN, T_OUT, LEAD_IDX
from scripts.train import load_config
from model.pluvian import Pluvian

LAT0, LAT1, LON0, LON1 = 36.0, 42.6, 113.0, 120.0
# Fallback config for ckpts that don't embed their own (gan / latent fine-tunes)
FALLBACK_CFG = Path(__file__).resolve().parent.parent / "configs" / "ablation_2_new_7b_p7c.yaml"


def build_model(ckpt_path, device):
    """Build Pluvian from a ckpt; fall back to FALLBACK_CFG when the ckpt does
    not embed its own model config (gan/latent fine-tune checkpoints)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ck.get("config") or ck.get("cfg"):
        return _build_model_embedded(ckpt_path, device)
    m = load_config(FALLBACK_CFG)["model"]
    model = Pluvian(
        radar_channels=m["radar_channels"], pwv_enabled=m["pwv_enabled"],
        pwv_concat_only=m["pwv_concat_only"], era5_enabled=m["era5_enabled"],
        mfd_channel_enabled=m["mfd_channel_enabled"],
        forecast_frames=m["forecast_frames"], input_frames=m["input_frames"],
        hidden_dim=m["hidden_dim"], n_layers=m["n_layers"], n_heads=m["n_heads"],
        n_era5_vars=m["n_era5_vars"], n_era5_levels=m["n_era5_levels"],
        mfd_channels=m["mfd_channels"],
        intensity_stratified=bool(m.get("intensity_stratified", False)),
        band_centers=tuple(m.get("band_centers", (0.0, 0.5, 4.5, 19.0, 50.0))),
        gated_fusion=bool(m.get("gated_fusion", False)),
    )
    model.load_state_dict(ck["model"], strict=False)
    return model.to(device).eval()


def rapsd(field):
    """Radially-averaged power spectral density of a 2D field (H,W)."""
    f = np.fft.fftshift(np.fft.fft2(field))
    power = np.abs(f) ** 2
    H, W = field.shape
    cy, cx = H // 2, W // 2
    y, x = np.indices((H, W))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    rmax = min(cy, cx)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    radial = tbin / np.maximum(nr, 1)
    return radial[:rmax]  # index = wavenumber k (pixels^-1)


def predict(model, sample, device, Ht, Wt):
    with torch.no_grad():
        return model(make_batch(sample, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]


def main(base_ck, a_ck, b_ck, outdir, start):
    print(f"=== sharp compare  start={start} ===")
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NPJDataset(manifest=Path(__file__).resolve().parent.parent / "pipeline" / "manifest.csv",
                    starts=[start], load_mfd=False, window_minutes=180)
    sample = ds[0]
    gt = sample["radar"][T_IN:T_IN + T_OUT].numpy()
    Ht, Wt = gt.shape[-2:]

    models = {
        "baseline (p7c)": build_model(base_ck, device),
        "sharpA (mod)": build_model(a_ck, device),
        "sharpB (aggr)": build_model(b_ck, device),
    }
    t0 = time.time()
    preds = {name: predict(m, sample, device, Ht, Wt) for name, m in models.items()}
    print(f"  forward x3: {time.time()-t0:.1f}s  " +
          "  ".join(f"{n} max={p.max():.1f}" for n, p in preds.items()))

    # ---- figure 1: visual comparison ----
    extent = [LON0, LON1, LAT0, LAT1]
    rows = ["Ground truth"] + list(preds.keys())
    data_rows = [gt] + list(preds.values())
    n_cols = len(LEAD_IDX)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.5 * n_cols, 3.1 * len(rows)))
    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    for j, idx in enumerate(LEAD_IDX):
        lab = f"+{(idx + 1) * 6} min"
        for r, (rname, arr) in enumerate(zip(rows, data_rows)):
            ax = axes[r, j]
            ax.imshow(arr[idx], origin="lower", extent=extent, cmap=DBZ_CMAP, norm=DBZ_NORM, aspect="auto")
            ax.set_title(f"{rname.split()[0]}  {lab}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rname, fontsize=11)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
    fig.colorbar(plt.cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_CMAP), cax=cbar_ax, label="dBZ")
    fig.suptitle(f"Sharpen check  GT vs baseline vs sharpA vs sharpB  start={start}", y=0.997, fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.91, 0.985])
    out1 = outdir / f"cmp_{start[:13]}.png"
    fig.savefig(out1, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  saved:", out1)

    # ---- figure 2: RAPSD (averaged over all lead frames) ----
    def mean_rapsd(arr):
        curves = np.stack([rapsd(arr[t]) for t in range(arr.shape[0])])
        return curves.mean(0)
    fig2, ax = plt.subplots(figsize=(7, 5.5))
    k = None
    series = [("GT", gt, "k", 2.2)] + [
        (n, p, c, 1.6) for (n, p), c in zip(preds.items(), ["C0", "C1", "C2"])
    ]
    for name, arr, color, lw in series:
        curve = mean_rapsd(arr)
        if k is None:
            k = np.arange(1, len(curve))
        ax.loglog(k, curve[1:], color=color, lw=lw, label=name)
    ax.set_xlabel("wavenumber k (pixels$^{-1}$)  — high k = fine detail")
    ax.set_ylabel("radially-averaged power")
    ax.set_title(f"RAPSD  start={start}\n(closer to GT at high k = sharper; blur = energy deficit)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    out2 = outdir / f"rapsd_{start[:13]}.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight"); plt.close(fig2)
    print("  saved:", out2)

    # ---- numeric: high-k energy ratio vs GT (last 40% of spectrum) ----
    gt_c = mean_rapsd(gt)
    hi = slice(int(0.6 * len(gt_c)), len(gt_c))
    gt_hi = gt_c[hi].sum()
    print("  high-k energy ratio (pred/GT, 1.0=match):")
    for name, arr in preds.items():
        ratio = mean_rapsd(arr)[hi].sum() / gt_hi
        print(f"    {name:18s} {ratio:.3f}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], a[2], a[3], a[4] if len(a) > 4 else "2023-05-19T10:00:00")
