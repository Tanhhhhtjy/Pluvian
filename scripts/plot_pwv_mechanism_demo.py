"""DEMO: PWV mechanism figure for one event_test start.

Four panels testing the co-location hypothesis — does PWV help where the
moisture is and where the model opens its PWV gate?

  1. PWV input field (interpolated from GNSS stations, mean over input window)
  2. g_pwv gate map  — per-pixel sigmoid gate inside GatedAsymmFusion
                       (where the model admits the PWV branch)
  3. counterfactual  ab2(PWV on) - ab2(PWV zeroed)  forecast delta (dBZ)
                       — what the PWV signal actually changed in the forecast
  4. error reduction |ab1 - GT| - |ab2 - GT|  (positive = ab2 closer to truth)

Usage:
    python scripts/plot_pwv_mechanism_demo.py CKPT_AB1 CKPT_AB2 OUT_PNG [START]
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data_loader import NPJDataset
from model.pluvian import Pluvian
from scripts.plot_prediction import DBZ_CMAP, DBZ_NORM

T_IN, T_OUT = 12, 18
LAT0, LAT1, LON0, LON1 = 36.0, 42.6, 113.0, 120.0


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


def make_batch(sample, device, zero_pwv=False):
    pg = sample["pwv_grid"][: T_IN // 5].clone()
    pm = sample["pwv_mask"][: T_IN // 5].clone()
    if zero_pwv:
        pg.zero_(); pm.zero_()
    batch = {
        "radar": sample["radar"][:T_IN].unsqueeze(0).float(),
        "pwv_grid": pg.unsqueeze(0).float(),
        "pwv_mask": pm.unsqueeze(0).float(),
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


def main(ckpt_ab1, ckpt_ab2, out_png, start="2023-07-31T06:00:00"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NPJDataset(manifest=Path(__file__).resolve().parent.parent / "pipeline" / "manifest.csv",
                    starts=[start], load_mfd=False, window_minutes=180)
    sample = ds[0]
    radar_gt = sample["radar"][T_IN:T_IN + T_OUT].numpy()  # (T_OUT,H,W) dBZ
    Ht, Wt = radar_gt.shape[-2:]

    ab1 = build_model(ckpt_ab1, device)
    ab2 = build_model(ckpt_ab2, device)

    # hook the per-pixel PWV gate (pre-sigmoid Conv2d output)
    gate_store = {}

    def hook(_m, _inp, out):
        gate_store["g"] = torch.sigmoid(out).detach().float().cpu()

    h = ab2.fusion.gate_pwv.register_forward_hook(hook)

    with torch.no_grad():
        pred_ab1 = ab1(make_batch(sample, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]
        pred_on = ab2(make_batch(sample, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]
        g_pwv = gate_store["g"]  # (B*T,1,Hf,Wf)
        pred_off = ab2(make_batch(sample, device, zero_pwv=True))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]
    h.remove()

    # gate: mean over input frames, upsample to radar grid
    g = g_pwv.mean(dim=0)[0].numpy()  # (Hf,Wf)
    g_up = np.array(torch.nn.functional.interpolate(
        torch.from_numpy(g)[None, None], size=(Ht, Wt), mode="bilinear",
        align_corners=False)[0, 0])

    # PWV interpolated field (mean over input frames)
    coords = sample["pwv_coords"].numpy()           # (N,2) lon,lat
    pvals = sample["pwv_grid"][: T_IN // 5].numpy()  # (t,N)
    pmask = sample["pwv_mask"][: T_IN // 5].numpy()
    valid = pmask.mean(0) > 0.5
    pmean = (pvals * pmask).sum(0) / np.clip(pmask.sum(0), 1, None)
    gx, gy = np.meshgrid(np.linspace(LON0, LON1, Wt), np.linspace(LAT0, LAT1, Ht))
    pwv_field = griddata(coords[valid], pmean[valid], (gx, gy), method="linear")

    # forecast-space maps (mean over lead frames)
    counterfactual = (pred_on - pred_off).mean(0)                      # dBZ
    err_red = (np.abs(pred_ab1 - radar_gt) - np.abs(pred_on - radar_gt)).mean(0)  # dBZ

    extent = [LON0, LON1, LAT0, LAT1]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    mpl.rcParams.update({"font.family": "DejaVu Sans"})

    ax = axes[0]
    im = ax.imshow(pwv_field, origin="lower", extent=extent, cmap="YlGnBu", aspect="auto")
    ax.scatter(coords[valid, 0], coords[valid, 1], s=4, c="k", alpha=0.35)
    ax.set_title("(a) PWV input field (mm)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="PWV (mm)")

    ax = axes[1]
    im = ax.imshow(g_up, origin="lower", extent=extent, cmap="magma", vmin=0, aspect="auto")
    ax.set_title("(b) PWV gate $g_{pwv}$ (open = admit PWV)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="gate")

    ax = axes[2]
    vmax = np.percentile(np.abs(counterfactual), 99) or 1.0
    im = ax.imshow(counterfactual, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("(c) Counterfactual  ab2$_{PWV}$ - ab2$_{noPWV}$ (dBZ)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="dBZ")

    ax = axes[3]
    vmax = np.percentile(np.abs(err_red), 99) or 1.0
    im = ax.imshow(err_red, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_title("(d) Error reduction |ab1-GT|-|ab2-GT| (dBZ, +=ab2 better)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="dBZ")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"PWV mechanism demo  —  start={start}", y=1.02, fontsize=13)
    plt.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    print("saved:", out_png)
    # quick co-location stats
    def corr(a, b):
        a = a.ravel(); b = b.ravel(); ok = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[ok], b[ok])[0, 1])
    print("corr(gate, |counterfactual|):", corr(g_up, np.abs(counterfactual)))
    print("corr(PWV field, err_reduction):", corr(pwv_field, err_red))
    print("corr(gate, err_reduction):", corr(g_up, err_red))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], a[4] if len(a) > 4 else "2023-07-31T06:00:00")
