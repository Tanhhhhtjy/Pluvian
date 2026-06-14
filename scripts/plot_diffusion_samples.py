"""Render DiffCast-style residual-diffusion samples vs the backbone mean.

    python scripts/plot_diffusion_samples.py BACKBONE_CKPT DIFF_CKPT OUTDIR START [STEPS]

Rows: GT / backbone-mean / diff sample#1 / diff sample#2 (+ DDIM steps),
4 lead cols, DBZ NWS colormap. Plus RAPSD(GT, mean, samples) + high-k energy
ratio: does drawing a fresh residual grow the intense echo cores the mean blurs?
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from pipeline.data_loader import NPJDataset
from scripts.plot_prediction import DBZ_CMAP, DBZ_NORM
from scripts.plot_ab1_vs_ab2 import make_batch, T_IN, T_OUT, LEAD_IDX
from scripts.plot_sharp_compare import build_model, rapsd
from model.diffusion_head import CondEpsUNet, GaussianDiffusion, pad_to_multiple

LAT0, LAT1, LON0, LON1 = 36.0, 42.6, 113.0, 120.0


def main(backbone_ck, diff_ck, outdir, start, steps=30):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = NPJDataset(manifest=Path(__file__).resolve().parent.parent / "pipeline" / "manifest.csv",
                    starts=[start], load_mfd=False, window_minutes=180)
    sample = ds[0]
    gt = sample["radar"][T_IN:T_IN + T_OUT].numpy()
    Ht, Wt = gt.shape[-2:]

    backbone = build_model(backbone_ck, device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    dk = torch.load(diff_ck, map_location=device, weights_only=False)
    net = CondEpsUNet(base=dk["base"]).to(device)
    diff = GaussianDiffusion(net, timesteps=dk["timesteps"]).to(device)
    net.load_state_dict(dk["net"]); net.eval()
    RES, MU = dk["res_scale"], dk["mu_scale"]
    print(f"  diff ckpt ep={dk['epoch']} base={dk['base']} T={dk['timesteps']} steps={steps}")

    batch = make_batch(sample, device)
    with torch.no_grad():
        fused, skips = backbone._encode_and_fuse(batch, return_skips=True)
        dec = backbone.decoder(fused, encoder_skips=skips)
        mu = (dec[0] if isinstance(dec, (tuple, list)) else dec)[0, :, :Ht, :Wt]  # (T,H,W) dBZ
    mu_np = mu.cpu().numpy()

    cond = (mu / MU).unsqueeze(1)               # (T,1,H,W)
    cond_p, hw = pad_to_multiple(cond, 8)
    # soft gate: the diffusion prior injects noise over the WHOLE field, but the
    # true residual is ~0 in calm (echo-free) regions -> speckle. Suppress the
    # sampled residual where the backbone mean has no echo, and clamp the final
    # field to a physical dBZ range. Gate is smooth so storm edges stay alive.
    gate = 1.0 / (1.0 + np.exp(-(mu_np - 8.0) / 3.0))   # ~0 where mu<5, ~1 where mu>12

    def draw_sample():
        with torch.no_grad():
            r = diff.ddim_sample(cond_p.float(), n_steps=steps)  # (T,1,Hp,Wp) unit resid
        r = r[..., :Ht, :Wt].squeeze(1).cpu().numpy() * RES
        return np.clip(mu_np + gate * r, 0, 65)

    s1, s2 = draw_sample(), draw_sample()
    print(f"  mean max={mu_np.max():.1f}  sample maxes={s1.max():.1f}, {s2.max():.1f}")

    extent = [LON0, LON1, LAT0, LAT1]
    rows = [("Ground truth", gt), ("backbone mean", mu_np),
            ("diff sample #1", s1), ("diff sample #2", s2)]
    fig, axes = plt.subplots(len(rows), len(LEAD_IDX), figsize=(3.5 * len(LEAD_IDX), 3.0 * len(rows)))
    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    for j, idx in enumerate(LEAD_IDX):
        lab = f"+{(idx + 1) * 6} min"
        for r, (rname, arr) in enumerate(rows):
            ax = axes[r, j]
            ax.imshow(arr[idx], origin="lower", extent=extent, cmap=DBZ_CMAP, norm=DBZ_NORM, aspect="auto")
            ax.set_title(f"{rname.split()[0]}  {lab}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(rname, fontsize=10)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
    fig.colorbar(plt.cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_CMAP), cax=cax, label="dBZ")
    fig.suptitle(f"Residual-diffusion samples  start={start}", y=0.997, fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.91, 0.985])
    out1 = outdir / f"diff_{start[:13]}.png"
    fig.savefig(out1, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  saved:", out1)

    def mean_rapsd(arr):
        return np.stack([rapsd(arr[t]) for t in range(arr.shape[0])]).mean(0)
    fig2, ax = plt.subplots(figsize=(7, 5.5))
    for name, arr, c, lw in [("GT", gt, "k", 2.2), ("backbone mean", mu_np, "C0", 1.6),
                             ("diff #1", s1, "C1", 1.6), ("diff #2", s2, "C2", 1.6)]:
        cur = mean_rapsd(arr)
        ax.loglog(np.arange(1, len(cur)), cur[1:], color=c, lw=lw, label=name)
    ax.set_xlabel("wavenumber k (pixels$^{-1}$)"); ax.set_ylabel("radially-averaged power")
    ax.set_title(f"RAPSD  start={start}  (sample closer to GT at high k = cores recovered)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    out2 = outdir / f"diff_rapsd_{start[:13]}.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight"); plt.close(fig2)
    print("  saved:", out2)

    gt_c = mean_rapsd(gt); hi = slice(int(0.6 * len(gt_c)), len(gt_c)); gt_hi = gt_c[hi].sum()
    print("  high-k energy ratio (pred/GT):")
    print(f"    backbone mean   {mean_rapsd(mu_np)[hi].sum()/gt_hi:.3f}")
    print(f"    diff sample #1  {mean_rapsd(s1)[hi].sum()/gt_hi:.3f}")
    print(f"    diff sample #2  {mean_rapsd(s2)[hi].sum()/gt_hi:.3f}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], a[2], a[3] if len(a) > 3 else "2023-05-19T10:00:00",
         int(a[4]) if len(a) > 4 else 30)
