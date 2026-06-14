"""Phase 7d — multi-case generative evaluation.

Turns the single-case generative demo into a defensible, aggregated comparison
over a set of held-out (test/event) strong-convection windows. For each method
we report the two honest blur diagnostics:

  * high-k energy ratio (pred/GT) — radially-averaged power spectrum, last 40%
    of wavenumbers, summed. 1.0 == GT fine-structure recovered; the MSE mean
    sits near ~0.03 (pure blur).
  * peak-dBZ recovery — max intensity over the 18 forecast frames, pred vs GT.
    The conditional mean caps ~47-50 dBZ; the question is which method reaches
    the real 60+ dBZ cores.

Methods: baseline (frozen backbone conditional mean) / GAN-core (adversarial
decoder fine-tune) / diffusion (DiffCast-style residual DDPM, stochastic).

The integrity line: these are PERCEPTUAL / spectral-fidelity diagnostics, not
CSI. We never claim a skill-score gain from sampling — only that sampling
restores the high-wavenumber energy and extreme cores the mean averages away.

    python scripts/eval_generative.py BACKBONE_CK GAN_CK DIFF_CK OUTDIR \
        --days 2023-07-29,2023-07-30,2023-07-31,2023-08-01,2023-08-08,2023-08-20 \
        --n-select 8 --n-show 6 --diff-steps 30
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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


def load_sample(start):
    """Return (sample, gt[T,H,W]) or None if the window is unavailable."""
    try:
        ds = NPJDataset(manifest=MANIFEST, starts=[start], load_mfd=False, window_minutes=180)
        s = ds[0]
        gt = s["radar"][T_IN:T_IN + T_OUT].numpy()
        if gt.shape[0] < T_OUT or not np.isfinite(gt).any():
            return None
        return s, gt
    except Exception as e:  # invalid window / missing radar
        print(f"  [skip] {start}: {e}")
        return None


def mean_rapsd(arr):
    return np.stack([rapsd(arr[t]) for t in range(arr.shape[0])]).mean(0)


def highk_ratio(arr, gt_hi, hi):
    return float(mean_rapsd(arr)[hi].sum() / gt_hi)


def backbone_mu(backbone, sample, device, Ht, Wt):
    batch = make_batch(sample, device)
    with torch.no_grad():
        fused, skips = backbone._encode_and_fuse(batch, return_skips=True)
        dec = backbone.decoder(fused, encoder_skips=skips)
        mu = (dec[0] if isinstance(dec, (tuple, list)) else dec)[0, :, :Ht, :Wt]
    return mu  # (T,H,W) tensor, dBZ


def diff_draw(diff, mu, RES, MU, steps, device):
    """One stochastic residual sample, mu-gated + clamped (same as the demo)."""
    mu_np = mu.cpu().numpy()
    cond = (mu / MU).unsqueeze(1)
    cond_p, _ = pad_to_multiple(cond, 8)
    gate = 1.0 / (1.0 + np.exp(-(mu_np - 8.0) / 3.0))
    Ht, Wt = mu_np.shape[-2:]
    with torch.no_grad():
        r = diff.ddim_sample(cond_p.float().to(device), n_steps=steps)
    r = r[..., :Ht, :Wt].squeeze(1).cpu().numpy() * RES
    return np.clip(mu_np + gate * r, 0, 65)


def gan_predict(gan_model, sample, device, Ht, Wt):
    with torch.no_grad():
        out = gan_model(make_batch(sample, device))["rain_pred"][0].cpu().numpy()
    return out[:, :Ht, :Wt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone_ck", type=Path)
    ap.add_argument("gan_ck", type=Path)
    ap.add_argument("diff_ck", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--days", required=True, help="comma-separated YYYY-MM-DD test days")
    ap.add_argument("--hours", default="1,4,7,10,13,16,19", help="candidate start hours per day")
    ap.add_argument("--n-select", type=int, default=8, help="strongest GT windows to keep")
    ap.add_argument("--n-show", type=int, default=6, help="events in the qualitative grid")
    ap.add_argument("--n-diff", type=int, default=2, help="diffusion samples per event")
    ap.add_argument("--diff-steps", type=int, default=30)
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    days = [d for d in args.days.split(",") if d.strip()]
    hours = [int(h) for h in args.hours.split(",") if h.strip()]
    cand = [f"{d}T{h:02d}:00:00" for d in days for h in hours]

    # ---- pass 1: rank candidate windows by GT peak intensity (pick strong echo) ----
    print(f"[select] scanning {len(cand)} candidate windows for strongest GT echo")
    scored = []
    for st in cand:
        r = load_sample(st)
        if r is None:
            continue
        _, gt = r
        scored.append((float(gt.max()), st))
    scored.sort(reverse=True)
    chosen = [st for _, st in scored[:args.n_select]]
    print("[select] kept (GT peak dBZ, start):")
    for pk, st in scored[:args.n_select]:
        print(f"    {pk:5.1f}  {st}")

    # ---- models ----
    backbone = build_model(args.backbone_ck, device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    gan = build_model(args.gan_ck, device)
    dk = torch.load(args.diff_ck, map_location=device, weights_only=False)
    net = CondEpsUNet(base=dk["base"]).to(device)
    diff = GaussianDiffusion(net, timesteps=dk["timesteps"]).to(device)
    net.load_state_dict(dk["net"]); net.eval()
    RES, MU = dk["res_scale"], dk["mu_scale"]
    print(f"[diff] ep={dk['epoch']} base={dk['base']} T={dk['timesteps']} steps={args.diff_steps}")

    # ---- pass 2: per-event metrics + cache fields for the grid ----
    rows = []          # metric dicts
    grid_cache = []    # (start, gt, mu, gan, diff_sample, peak_frame)
    for st in chosen:
        r = load_sample(st)
        if r is None:
            continue
        sample, gt = r
        Ht, Wt = gt.shape[-2:]
        gt_c = mean_rapsd(gt)
        hi = slice(int(0.6 * len(gt_c)), len(gt_c))
        gt_hi = gt_c[hi].sum()

        mu = backbone_mu(backbone, sample, device, Ht, Wt)
        mu_np = mu.cpu().numpy()
        gan_np = gan_predict(gan, sample, device, Ht, Wt)
        diff_samps = [diff_draw(diff, mu, RES, MU, args.diff_steps, device) for _ in range(args.n_diff)]
        diff_np = diff_samps[0]

        row = {
            "start": st,
            "gt_peak": float(gt.max()),
            "base_peak": float(mu_np.max()),
            "gan_peak": float(gan_np.max()),
            "diff_peak": float(np.mean([s.max() for s in diff_samps])),
            "base_hk": highk_ratio(mu_np, gt_hi, hi),
            "gan_hk": highk_ratio(gan_np, gt_hi, hi),
            "diff_hk": float(np.mean([highk_ratio(s, gt_hi, hi) for s in diff_samps])),
        }
        rows.append(row)
        pf = int(np.argmax(gt.reshape(gt.shape[0], -1).max(1)))  # frame of peak GT echo
        grid_cache.append((st, gt, mu_np, gan_np, diff_np, pf))
        print(f"  {st}  GTpk={row['gt_peak']:.0f}  hk base/gan/diff="
              f"{row['base_hk']:.2f}/{row['gan_hk']:.2f}/{row['diff_hk']:.2f}  "
              f"pk base/gan/diff={row['base_peak']:.0f}/{row['gan_peak']:.0f}/{row['diff_peak']:.0f}")

    # ---- aggregate ----
    def col(k):
        return np.array([r[k] for r in rows], float)
    agg = {m: {"hk_mean": float(col(f"{m}_hk").mean()), "hk_std": float(col(f"{m}_hk").std()),
               "peak_mean": float(col(f"{m}_peak").mean()), "peak_std": float(col(f"{m}_peak").std())}
           for m in ("base", "gan", "diff")}
    agg["gt_peak_mean"] = float(col("gt_peak").mean())
    agg["n_events"] = len(rows)
    (outdir / "eval_generative.json").write_text(json.dumps({"agg": agg, "events": rows}, indent=2))
    print("\n[aggregate] over", len(rows), "events  (mean +/- std)")
    print(f"  high-k ratio   base {agg['base']['hk_mean']:.3f}  gan {agg['gan']['hk_mean']:.3f}  diff {agg['diff']['hk_mean']:.3f}")
    print(f"  peak dBZ       GT {agg['gt_peak_mean']:.1f} | base {agg['base']['peak_mean']:.1f}  "
          f"gan {agg['gan']['peak_mean']:.1f}  diff {agg['diff']['peak_mean']:.1f}")

    # ---- figure 1: summary (high-k ratio + peak recovery) ----
    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    methods = ["base", "gan", "diff"]
    labels = ["baseline\n(MSE mean)", "GAN-core", "diffusion"]
    colors = ["C0", "C1", "C2"]
    for i, (m, c) in enumerate(zip(methods, colors)):
        v = col(f"{m}_hk")
        axA.scatter(np.full_like(v, i) + np.linspace(-.12, .12, len(v)), v, color=c, s=40, alpha=.8, zorder=3)
        axA.bar(i, v.mean(), width=.55, color=c, alpha=.25, zorder=1)
    axA.axhline(1.0, color="k", ls="--", lw=1.2, label="GT (=1.0)")
    axA.set_xticks(range(3)); axA.set_xticklabels(labels)
    axA.set_ylabel("high-k energy ratio  (pred / GT)")
    axA.set_title(f"Fine-structure recovery  (n={len(rows)} events)\nhigher = more high-wavenumber energy")
    axA.legend(); axA.grid(True, axis="y", alpha=.3)

    gtp = col("gt_peak")
    for m, c, lab in zip(methods, colors, ["baseline", "GAN-core", "diffusion"]):
        axB.scatter(gtp, col(f"{m}_peak"), color=c, s=45, alpha=.85, label=lab, zorder=3)
    lim = [min(gtp.min(), 40) - 2, gtp.max() + 3]
    axB.plot(lim, lim, "k--", lw=1.2, label="perfect (y=x)")
    axB.set_xlim(lim); axB.set_ylim(lim)
    axB.set_xlabel("GT peak dBZ"); axB.set_ylabel("predicted peak dBZ")
    axB.set_title("Extreme-core recovery\n(on the y=x line = reaches true peak)")
    axB.legend(); axB.grid(True, alpha=.3)
    fig.suptitle("Phase 7d generative evaluation — baseline vs GAN-core vs diffusion", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    f1 = outdir / "summary_generative.png"
    fig.savefig(f1, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  saved:", f1)

    # ---- figure 2: qualitative grid (rows=events, cols=methods at peak frame) ----
    grid_cache.sort(key=lambda x: x[1].max(), reverse=True)
    show = grid_cache[:args.n_show]
    cols = ["Ground truth", "baseline (mean)", "GAN-core", "diffusion"]
    extent = [LON0, LON1, LAT0, LAT1]
    fig2, axes = plt.subplots(len(show), 4, figsize=(4 * 3.4, len(show) * 3.0))
    if len(show) == 1:
        axes = axes[None, :]
    for ri, (st, gt, mu_np, gan_np, diff_np, pf) in enumerate(show):
        fields = [gt[pf], mu_np[pf], gan_np[pf], diff_np[pf]]
        for ci, (cname, arr) in enumerate(zip(cols, fields)):
            ax = axes[ri, ci]
            ax.imshow(arr, origin="lower", extent=extent, cmap=DBZ_CMAP, norm=DBZ_NORM, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(cname, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{st[:13]}\n+{(pf + 1) * 6}min  pk{gt.max():.0f}", fontsize=9)
            ax.text(.02, .96, f"max {arr.max():.0f}", transform=ax.transAxes, fontsize=8,
                    va="top", ha="left", color="w",
                    bbox=dict(boxstyle="round", fc="k", alpha=.45, lw=0))
    cax = fig2.add_axes([0.93, 0.25, 0.011, 0.5])
    fig2.colorbar(plt.cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_CMAP), cax=cax, label="dBZ")
    fig2.suptitle("Per-event peak-frame comparison  (baseline blurs cores; diffusion restores 60+ dBZ)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.92, 0.975])
    f2 = outdir / "grid_generative.png"
    fig2.savefig(f2, dpi=200, bbox_inches="tight"); plt.close(fig2)
    print("  saved:", f2)


if __name__ == "__main__":
    main()
