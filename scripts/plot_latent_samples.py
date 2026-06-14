"""Render latent-head (CVAE) samples vs the deterministic backbone mean.

    python scripts/plot_latent_samples.py CONFIG BACKBONE_CKPT HEAD_CKPT OUTDIR START [K]

Rows: GT / backbone-mean (z=0) / ensemble-mean(K) / sample#1 / sample#2,
4 lead cols. Plus RAPSD(GT, mean, sample) + high-k energy ratio so we can
see whether sampling actually grows the intense echo cores the mean blurs out.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from pipeline.data_loader import NPJDataset
from model.latent_head import LatentSharpHead
from scripts.plot_prediction import DBZ_CMAP, DBZ_NORM
from scripts.plot_ab1_vs_ab2 import build_model, make_batch, T_IN, T_OUT, LEAD_IDX
from scripts.plot_sharp_compare import rapsd

LAT0, LAT1, LON0, LON1 = 36.0, 42.6, 113.0, 120.0


def main(config, backbone_ck, head_ck, outdir, start, K=4):
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
    hs = torch.load(head_ck, map_location=device, weights_only=False)
    head = LatentSharpHead(n_ch=backbone.hidden_dim, z_dim=hs["z_dim"], t_out=T_OUT).to(device)
    head.load_state_dict(hs["head"]); head.eval()

    batch = make_batch(sample, device)
    with torch.no_grad():
        fused, skips = backbone._encode_and_fuse(batch, return_skips=True)

        def decode(fused_in):
            dec = backbone.decoder(fused_in, encoder_skips=skips)
            return dec[0][0].cpu().numpy()[:, :Ht, :Wt]

        mean = decode(fused)  # z=0 -> FiLM identity-ish (film init ~0) = backbone mean
        samples = []
        for _ in range(K):
            z = head.sample_prior(1, device)
            samples.append(decode(head.apply_film(fused, z)))
    ens_mean = np.mean(samples, axis=0)
    print(f"  backbone max={mean.max():.1f}  sample maxes={[f'{s.max():.1f}' for s in samples]}")

    extent = [LON0, LON1, LAT0, LAT1]
    rows = [("Ground truth", gt), ("backbone mean", mean),
            (f"ens-mean(K={K})", ens_mean), ("sample #1", samples[0]),
            ("sample #2", samples[1] if K > 1 else samples[0])]
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
    fig.suptitle(f"Latent-head samples  start={start}", y=0.997, fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.91, 0.985])
    out1 = outdir / f"latent_{start[:13]}.png"
    fig.savefig(out1, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  saved:", out1)

    # RAPSD
    def mean_rapsd(arr):
        return np.stack([rapsd(arr[t]) for t in range(arr.shape[0])]).mean(0)
    fig2, ax = plt.subplots(figsize=(7, 5.5))
    for name, arr, c, lw in [("GT", gt, "k", 2.2), ("backbone mean", mean, "C0", 1.6),
                             ("sample #1", samples[0], "C1", 1.6), ("sample #2", samples[1] if K > 1 else samples[0], "C2", 1.6)]:
        cur = mean_rapsd(arr)
        ax.loglog(np.arange(1, len(cur)), cur[1:], color=c, lw=lw, label=name)
    ax.set_xlabel("wavenumber k (pixels$^{-1}$)"); ax.set_ylabel("radially-averaged power")
    ax.set_title(f"RAPSD  start={start}  (sample closer to GT at high k = cores recovered)")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    out2 = outdir / f"latent_rapsd_{start[:13]}.png"
    fig2.savefig(out2, dpi=200, bbox_inches="tight"); plt.close(fig2)
    print("  saved:", out2)

    gt_c = mean_rapsd(gt); hi = slice(int(0.6 * len(gt_c)), len(gt_c)); gt_hi = gt_c[hi].sum()
    print("  high-k energy ratio (pred/GT):")
    print(f"    backbone mean   {mean_rapsd(mean)[hi].sum()/gt_hi:.3f}")
    for i, s in enumerate(samples):
        print(f"    sample #{i+1}       {mean_rapsd(s)[hi].sum()/gt_hi:.3f}")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], a[2], a[3], a[4] if len(a) > 4 else "2023-05-19T10:00:00",
         int(a[5]) if len(a) > 5 else 4)
