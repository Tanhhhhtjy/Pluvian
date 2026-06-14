"""Phase 7d — tame the residual-diffusion sample so it SHARPENS existing cores
instead of repainting the whole storm red.

The raw sampler gates the residual wherever mu>12 dBZ and adds it at full
amplitude with no spatial smoothing -> 60+ dBZ speckle spread across the moderate
echo, high-k ratio 1.6 (> GT), contour destroyed. Here we sweep three knobs,
all pure post-processing on the SAME sampled residual (no retrain):

  * gate_thresh / gate_width : sigmoid gate on mu. High thresh => residual only
    acts inside already-strong echo, so light rain stays == baseline (contour
    preserved).
  * resid_scale : shrink the residual amplitude.
  * smooth_sigma : Gaussian-blur the residual to kill pixel speckle.

Renders GT / baseline / GAN-core / raw-diff / tuned variants at each event's
peak frame, and prints high-k ratio + peak dBZ per variant so we can pick the
setting that restores cores WITHOUT wrecking the outline.

    python scripts/tune_diffusion.py BACKBONE_CK GAN_CK DIFF_CK OUTDIR \
        --starts 2023-08-08T07:00:00,2023-07-31T07:00:00 --diff-steps 30
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.data_loader import NPJDataset
from scripts.plot_prediction import DBZ_CMAP, DBZ_NORM
from scripts.plot_ab1_vs_ab2 import make_batch, T_IN, T_OUT
from scripts.plot_sharp_compare import build_model, rapsd
from model.diffusion_head import CondEpsUNet, GaussianDiffusion, pad_to_multiple

MANIFEST = Path(__file__).resolve().parent.parent / "pipeline" / "manifest.csv"
LAT0, LAT1, LON0, LON1 = 36.0, 42.6, 113.0, 120.0

# (label, gate_thresh, gate_width, resid_scale, smooth_sigma)
VARIANTS = [
    ("raw",  8.0, 3.0, 1.0, 0.0),
    ("A",   25.0, 4.0, 0.6, 1.0),
    ("B",   30.0, 4.0, 0.8, 1.5),
    ("C",   20.0, 3.0, 0.5, 1.0),
]


def mean_rapsd(arr):
    return np.stack([rapsd(arr[t]) for t in range(arr.shape[0])]).mean(0)


def highk(arr, gt_hi, hi):
    return float(mean_rapsd(arr)[hi].sum() / gt_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone_ck", type=Path)
    ap.add_argument("gan_ck", type=Path)
    ap.add_argument("diff_ck", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--starts", required=True)
    ap.add_argument("--diff-steps", type=int, default=30)
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    starts = [s for s in args.starts.split(",") if s.strip()]

    backbone = build_model(args.backbone_ck, device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    gan = build_model(args.gan_ck, device)
    dk = torch.load(args.diff_ck, map_location=device, weights_only=False)
    net = CondEpsUNet(base=dk["base"]).to(device)
    diff = GaussianDiffusion(net, timesteps=dk["timesteps"]).to(device)
    net.load_state_dict(dk["net"]); net.eval()
    RES, MU = dk["res_scale"], dk["mu_scale"]
    print(f"[diff] ep={dk['epoch']} base={dk['base']} steps={args.diff_steps}")

    cols = ["GT", "baseline", "GAN-core"] + [f"diff-{v[0]}" for v in VARIANTS]
    fig, axes = plt.subplots(len(starts), len(cols), figsize=(len(cols) * 3.0, len(starts) * 3.0))
    if len(starts) == 1:
        axes = axes[None, :]
    extent = [LON0, LON1, LAT0, LAT1]
    mpl.rcParams.update({"font.family": "DejaVu Sans"})

    for ri, st in enumerate(starts):
        ds = NPJDataset(manifest=MANIFEST, starts=[st], load_mfd=False, window_minutes=180)
        sample = ds[0]
        gt = sample["radar"][T_IN:T_IN + T_OUT].numpy()
        Ht, Wt = gt.shape[-2:]
        gt_c = mean_rapsd(gt); hi = slice(int(0.6 * len(gt_c)), len(gt_c)); gt_hi = gt_c[hi].sum()

        with torch.no_grad():
            fused, skips = backbone._encode_and_fuse(make_batch(sample, device), return_skips=True)
            dec = backbone.decoder(fused, encoder_skips=skips)
            mu = (dec[0] if isinstance(dec, (tuple, list)) else dec)[0, :, :Ht, :Wt]
            gan_np = gan(make_batch(sample, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]
        mu_np = mu.cpu().numpy()

        # one shared sampled residual; all variants are post-processing of it
        cond_p, _ = pad_to_multiple((mu / MU).unsqueeze(1), 8)
        with torch.no_grad():
            r = diff.ddim_sample(cond_p.float(), n_steps=args.diff_steps)
        r = r[..., :Ht, :Wt].squeeze(1).cpu().numpy() * RES  # (T,H,W) dBZ residual

        variant_fields = {}
        for (lab, thr, w, scale, sig) in VARIANTS:
            gate = 1.0 / (1.0 + np.exp(-(mu_np - thr) / w))
            rr = r * scale
            if sig > 0:
                rr = np.stack([gaussian_filter(rr[t], sigma=sig) for t in range(rr.shape[0])])
            variant_fields[lab] = np.clip(mu_np + gate * rr, 0, 65)

        pf = int(np.argmax(gt.reshape(gt.shape[0], -1).max(1)))
        print(f"\n{st}  GTpk={gt.max():.0f}  peak-frame=+{(pf+1)*6}min")
        print(f"  {'method':10s} {'high-k':>7s} {'peak':>6s}")
        report = [("baseline", mu_np), ("GAN-core", gan_np)] + [(f"diff-{l}", variant_fields[l]) for l, *_ in VARIANTS]
        for name, arr in report:
            print(f"  {name:10s} {highk(arr, gt_hi, hi):7.2f} {arr.max():6.0f}")

        fields = [gt, mu_np, gan_np] + [variant_fields[v[0]] for v in VARIANTS]
        for ci, (cname, arr) in enumerate(zip(cols, fields)):
            ax = axes[ri, ci]
            ax.imshow(arr[pf], origin="lower", extent=extent, cmap=DBZ_CMAP, norm=DBZ_NORM, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(cname, fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"{st[:13]}\npk{gt.max():.0f}", fontsize=9)
            ax.text(.02, .96, f"{arr[pf].max():.0f}", transform=ax.transAxes, fontsize=8,
                    va="top", color="w", bbox=dict(boxstyle="round", fc="k", alpha=.45, lw=0))

    cax = fig.add_axes([0.93, 0.25, 0.011, 0.5])
    fig.colorbar(plt.cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_CMAP), cax=cax, label="dBZ")
    fig.suptitle("Diffusion taming: strict gate + smoothing restores contour while lifting cores", fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.92, 0.97])
    out = outdir / "tune_diffusion.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
