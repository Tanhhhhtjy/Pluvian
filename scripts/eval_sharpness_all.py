"""Phase 7d — ONE honest cross-method sharpness comparison.

Every prior eval scored each de-blur method in isolation and leaned on a single
gameable number: max-dBZ vs a clip ceiling. That made the diffusion look like
the winner ("restores 65 dBZ") when in fact its residual *overshoots* and is
hard-clipped to 65 on every event (np.clip(.,0,65) in eval_generative.diff_draw)
and its high-k energy ratio is ~1.6 — i.e. it injects MORE fine-structure energy
than ground truth: hallucinated speckle, not faithful restoration.

This script puts all candidates on ONE axis with only honest, clip-resistant
diagnostics, over the same strongest-GT-echo windows:

  methods: baseline(MSE mean) / sharpB(spectral+FSS loss FT) / GAN-coreA(cap50)
           / hi-core(6-band cap65) / diffusion(residual DDIM, UNCLIPPED)

  metrics vs GT (per event, then averaged):
    * RAPSD high-k energy ratio (pred/GT, last 40% of k). 1.0 = match,
      <1 = blur (energy deficit), >1 = OVER-sharpen / hallucinate. Not a
      "bigger is better" score — the target is 1.0.
    * 99th / 99.9th percentile dBZ. Distribution-level intensity recovery that
      a single hot pixel + a clip ceiling cannot fake (unlike max()).
    * for diffusion: fraction of forecast pixels that would exceed the 65-dBZ
      operational clip — a direct measure of how much its "peak" is the ceiling.

Outputs sharpness_all.json + sharpness_all.png (RAPSD overlay + two honest bars).

    python scripts/eval_sharpness_all.py BACKBONE_CK SHARPB_CK COREA_CK \
        HICORE_CK HICORE_CFG DIFF_CK OUTDIR \
        --days 2023-07-29,2023-07-31,2023-08-01,2023-08-08,2023-08-20 \
        --hours 1,7,13,19 --n-select 8 --diff-steps 30
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
from scripts.plot_ab1_vs_ab2 import make_batch, T_IN, T_OUT
from scripts.plot_sharp_compare import build_model, rapsd
from scripts.peek_hicore import build_with_cfg
from model.diffusion_head import CondEpsUNet, GaussianDiffusion, pad_to_multiple

MANIFEST = Path(__file__).resolve().parent.parent / "pipeline" / "manifest.csv"

# Display order + colors. base/sharpB/coreA/hicore are Pluvian forwards;
# diff is the residual DDIM head, reported UNCLIPPED to expose overshoot.
METHODS = ["baseline", "sharpB", "gan_coreA", "hicore", "diff"]
LABELS = {"baseline": "baseline\n(MSE mean)", "sharpB": "sharpB\n(spec+FSS FT)",
          "gan_coreA": "GAN-coreA\n(cap 50)", "hicore": "hi-core\n(6-band)",
          "diff": "diffusion\n(unclipped)"}
COLORS = {"baseline": "C0", "sharpB": "C1", "gan_coreA": "C2",
          "hicore": "C3", "diff": "C4"}


def load(start):
    try:
        ds = NPJDataset(manifest=MANIFEST, starts=[start], load_mfd=False, window_minutes=180)
        s = ds[0]
        gt = s["radar"][T_IN:T_IN + T_OUT].numpy()
        if gt.shape[0] < T_OUT or not np.isfinite(gt).any():
            return None
        return s, gt
    except Exception as e:
        print(f"  [skip] {start}: {e}")
        return None


def mean_rapsd(a):
    return np.stack([rapsd(a[t]) for t in range(a.shape[0])]).mean(0)


def predict(model, sample, device, Ht, Wt):
    with torch.no_grad():
        return model(make_batch(sample, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]


def backbone_mu(backbone, sample, device, Ht, Wt):
    batch = make_batch(sample, device)
    with torch.no_grad():
        fused, skips = backbone._encode_and_fuse(batch, return_skips=True)
        dec = backbone.decoder(fused, encoder_skips=skips)
        mu = (dec[0] if isinstance(dec, (tuple, list)) else dec)[0, :, :Ht, :Wt]
    return mu


def diff_draw_unclipped(diff, mu, RES, MU, steps, device, hi_cap):
    """Residual DDIM sample, mu-gated. Clipped only at hi_cap (>=999 == none)
    so the true overshoot is visible instead of pinned to the 65 ceiling."""
    mu_np = mu.cpu().numpy()
    cond = (mu / MU).unsqueeze(1)
    cond_p, _ = pad_to_multiple(cond, 8)
    gate = 1.0 / (1.0 + np.exp(-(mu_np - 8.0) / 3.0))
    Ht, Wt = mu_np.shape[-2:]
    with torch.no_grad():
        r = diff.ddim_sample(cond_p.float().to(device), n_steps=steps)
    r = r[..., :Ht, :Wt].squeeze(1).cpu().numpy() * RES
    return np.clip(mu_np + gate * r, 0, hi_cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone_ck", type=Path)
    ap.add_argument("sharpb_ck", type=Path)
    ap.add_argument("corea_ck", type=Path)
    ap.add_argument("hicore_ck", type=Path)
    ap.add_argument("hicore_cfg", type=Path)
    ap.add_argument("diff_ck", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--days", required=True)
    ap.add_argument("--hours", default="1,7,13,19")
    ap.add_argument("--n-select", type=int, default=8)
    ap.add_argument("--diff-steps", type=int, default=30)
    ap.add_argument("--n-diff", type=int, default=2)
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    days = [d for d in args.days.split(",") if d.strip()]
    hours = [int(h) for h in args.hours.split(",") if h.strip()]
    cand = [f"{d}T{h:02d}:00:00" for d in days for h in hours]

    # pass 1: rank by GT peak, keep strongest n_select
    print(f"[select] scanning {len(cand)} windows for strongest GT echo")
    scored = []
    for st in cand:
        r = load(st)
        if r is not None:
            scored.append((float(r[1].max()), st))
    scored.sort(reverse=True)
    chosen = [st for _, st in scored[:args.n_select]]
    for pk, st in scored[:args.n_select]:
        print(f"    {pk:5.1f}  {st}")

    # models
    backbone = build_model(args.backbone_ck, device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    sharpb = build_model(args.sharpb_ck, device)
    corea = build_model(args.corea_ck, device)
    hicore = build_with_cfg(args.hicore_cfg, args.hicore_ck, device)
    dk = torch.load(args.diff_ck, map_location=device, weights_only=False)
    net = CondEpsUNet(base=dk["base"]).to(device)
    diff = GaussianDiffusion(net, timesteps=dk["timesteps"]).to(device)
    net.load_state_dict(dk["net"]); net.eval()
    RES, MU = dk["res_scale"], dk["mu_scale"]
    print(f"[diff] ep={dk['epoch']} base={dk['base']} T={dk['timesteps']} steps={args.diff_steps}")

    rapsd_sum = {m: None for m in METHODS}
    gt_rapsd_sum = None
    rows = []
    for st in chosen:
        r = load(st)
        if r is None:
            continue
        sample, gt = r
        Ht, Wt = gt.shape[-2:]
        gt_c = mean_rapsd(gt)
        hi = slice(int(0.6 * len(gt_c)), len(gt_c))
        gt_hi = gt_c[hi].sum()
        gt_rapsd_sum = gt_c if gt_rapsd_sum is None else gt_rapsd_sum + gt_c

        mu = backbone_mu(backbone, sample, device, Ht, Wt)
        diff_samps = [diff_draw_unclipped(diff, mu, RES, MU, args.diff_steps, device, 999)
                      for _ in range(args.n_diff)]
        fields = {
            "baseline": mu.cpu().numpy(),
            "sharpB": predict(sharpb, sample, device, Ht, Wt),
            "gan_coreA": predict(corea, sample, device, Ht, Wt),
            "hicore": predict(hicore, sample, device, Ht, Wt),
            "diff": diff_samps[0],
        }
        row = {"start": st, "gt_peak": float(gt.max()),
               "gt_p99": float(np.percentile(gt, 99)),
               "gt_p999": float(np.percentile(gt, 99.9))}
        for m, f in fields.items():
            c = mean_rapsd(f)
            rapsd_sum[m] = c if rapsd_sum[m] is None else rapsd_sum[m] + c
            row[f"{m}_hk"] = float(c[hi].sum() / gt_hi)
            row[f"{m}_p99"] = float(np.percentile(f, 99))
            row[f"{m}_p999"] = float(np.percentile(f, 99.9))
        # how much of diffusion's intensity is past the operational 65 clip
        row["diff_frac_over65"] = float((diff_samps[0] > 65).mean())
        rows.append(row)
        print(f"  {st}  GTpk={row['gt_peak']:.0f}  hk " +
              "/".join(f"{row[f'{m}_hk']:.2f}" for m in METHODS) +
              f"  diff>65={row['diff_frac_over65']*100:.2f}%")

    n = len(rows)
    def col(k): return np.array([r[k] for r in rows], float)
    agg = {"n": n, "gt_peak": float(col("gt_peak").mean()),
           "gt_p99": float(col("gt_p99").mean()), "gt_p999": float(col("gt_p999").mean()),
           "diff_frac_over65": float(col("diff_frac_over65").mean())}
    for m in METHODS:
        agg[m] = {"hk": float(col(f"{m}_hk").mean()), "hk_std": float(col(f"{m}_hk").std()),
                  "p99": float(col(f"{m}_p99").mean()), "p999": float(col(f"{m}_p999").mean())}
    (outdir / "sharpness_all.json").write_text(json.dumps({"agg": agg, "events": rows}, indent=2))

    print(f"\n[aggregate over {n} events]  GT: peak {agg['gt_peak']:.1f}  "
          f"p99 {agg['gt_p99']:.1f}  p99.9 {agg['gt_p999']:.1f}")
    print(f"  {'method':12s} {'hk(1.0)':>9s} {'p99':>7s} {'p99.9':>7s}")
    for m in METHODS:
        print(f"  {m:12s} {agg[m]['hk']:9.2f} {agg[m]['p99']:7.1f} {agg[m]['p999']:7.1f}")
    print(f"  diffusion fraction of pixels past 65-dBZ clip: {agg['diff_frac_over65']*100:.2f}%  "
          "(its 'peak 65' is the ceiling, not the model)")

    # ---- figure: RAPSD overlay + honest hk bar + honest p99.9 bar ----
    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(17, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1, 1])
    axR = fig.add_subplot(gs[0, 0]); axH = fig.add_subplot(gs[0, 1]); axP = fig.add_subplot(gs[0, 2])

    gt_curve = gt_rapsd_sum / n
    k = np.arange(1, len(gt_curve))
    axR.loglog(k, gt_curve[1:], color="k", lw=2.4, label="GT", zorder=5)
    for m in METHODS:
        c = rapsd_sum[m] / n
        axR.loglog(k, c[1:], color=COLORS[m], lw=1.7, label=LABELS[m].replace("\n", " "))
    axR.axvspan(int(0.6 * len(gt_curve)), len(gt_curve), color="grey", alpha=.10)
    axR.set_xlabel("wavenumber k (pixels$^{-1}$)  — high k = fine detail")
    axR.set_ylabel("radially-averaged power")
    axR.set_title("RAPSD vs GT (mean over events × leads)\n"
                  "on GT = faithful · below = blur · above = hallucinated")
    axR.legend(fontsize=8, loc="lower left"); axR.grid(True, which="both", alpha=.3)

    xs = np.arange(len(METHODS)); cs = [COLORS[m] for m in METHODS]
    hk = [agg[m]["hk"] for m in METHODS]
    axH.bar(xs, hk, color=cs, alpha=.8)
    axH.axhline(1.0, color="k", ls="--", lw=1.3, label="GT target = 1.0")
    axH.set_xticks(xs); axH.set_xticklabels([LABELS[m] for m in METHODS], fontsize=8)
    axH.set_ylabel("high-k energy ratio (pred / GT)")
    axH.set_title("Fine-structure recovery\n<1 blur · =1 faithful · >1 hallucinate")
    for x, v in zip(xs, hk):
        axH.text(x, v + .03, f"{v:.2f}", ha="center", fontsize=8)
    axH.legend(fontsize=8); axH.grid(True, axis="y", alpha=.3)

    p999 = [agg[m]["p999"] for m in METHODS]
    axP.bar(xs, p999, color=cs, alpha=.8)
    axP.axhline(agg["gt_p999"], color="k", ls="--", lw=1.3, label=f"GT p99.9 = {agg['gt_p999']:.0f}")
    axP.set_xticks(xs); axP.set_xticklabels([LABELS[m] for m in METHODS], fontsize=8)
    axP.set_ylabel("99.9th-percentile dBZ")
    axP.set_title("Intensity-distribution recovery\n(clip/​hot-pixel-resistant, unlike max)")
    for x, v in zip(xs, p999):
        axP.text(x, v + .4, f"{v:.0f}", ha="center", fontsize=8)
    axP.legend(fontsize=8); axP.grid(True, axis="y", alpha=.3)

    fig.suptitle(f"Phase 7d — honest sharpness comparison across all de-blur methods "
                 f"(n={n} strong-echo events)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = outdir / "sharpness_all.png"
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
    print("  saved:", out)


if __name__ == "__main__":
    main()
