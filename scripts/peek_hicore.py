"""Quick single-event peek at a hi-core GAN ckpt (6-band head) while it trains.

build_model's fallback config is 5-band, so a 6-band hi-core ckpt must be built
with its own config. Renders GT / baseline / coreA(5-band) / hicore(6-band) at
the peak frame and prints peak dBZ + high-k ratio — does the new top band break
the hard 50 dBZ ceiling?

    python scripts/peek_hicore.py BACKBONE_CK COREA_CK HICORE_CK HICORE_CFG OUT START
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data_loader import NPJDataset
from scripts.plot_prediction import DBZ_CMAP, DBZ_NORM
from scripts.plot_ab1_vs_ab2 import make_batch, T_IN, T_OUT
from scripts.plot_sharp_compare import build_model, rapsd
from scripts.train import load_config
from model.pluvian import Pluvian

MANIFEST = Path(__file__).resolve().parent.parent / "pipeline" / "manifest.csv"
LAT0, LAT1, LON0, LON1 = 36.0, 42.6, 113.0, 120.0


def build_with_cfg(cfg_path, ckpt, device):
    m = load_config(cfg_path)["model"]
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
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"], strict=False)
    return model.to(device).eval()


def main(backbone_ck, corea_ck, hicore_ck, hicore_cfg, out, start):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NPJDataset(manifest=MANIFEST, starts=[start], load_mfd=False, window_minutes=180)
    s = ds[0]
    gt = s["radar"][T_IN:T_IN + T_OUT].numpy()
    Ht, Wt = gt.shape[-2:]
    base = build_model(backbone_ck, device)
    corea = build_model(corea_ck, device)
    hicore = build_with_cfg(hicore_cfg, hicore_ck, device)

    def pred(m):
        with torch.no_grad():
            return m(make_batch(s, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]
    fields = {"Ground truth": gt, "baseline": pred(base),
              "GAN coreA (5-band)": pred(corea), "GAN hi-core (6-band)": pred(hicore)}

    gt_c = np.stack([rapsd(gt[t]) for t in range(gt.shape[0])]).mean(0)
    hi = slice(int(0.6 * len(gt_c)), len(gt_c)); gt_hi = gt_c[hi].sum()
    pf = int(np.argmax(gt.reshape(gt.shape[0], -1).max(1)))
    print(f"{start}  GTpk={gt.max():.0f}  peak-frame=+{(pf+1)*6}min")
    for n, a in fields.items():
        hk = np.stack([rapsd(a[t]) for t in range(a.shape[0])]).mean(0)[hi].sum() / gt_hi
        print(f"  {n:22s} peak={a.max():5.1f}  high-k={hk:.2f}")

    extent = [LON0, LON1, LAT0, LAT1]
    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, (n, a) in zip(axes, fields.items()):
        ax.imshow(a[pf], origin="lower", extent=extent, cmap=DBZ_CMAP, norm=DBZ_NORM, aspect="auto")
        ax.set_title(f"{n}\nmax {a[pf].max():.0f} dBZ", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Hi-core peek  {start[:13]}  +{(pf+1)*6}min", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main(*sys.argv[1:7])
