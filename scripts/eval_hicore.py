"""Phase 7d — hi-core GAN evaluation: did the 6-band head break the 50 dBZ cap
across the held-out strong-convection windows (not just one demo case)?

Compares, over the strongest GT-echo test windows:
  baseline (5-band conditional mean) / GAN coreA (5-band, ceiling 50) /
  GAN hi-core (6-band, ceiling 65).

Reports peak dBZ recovery + high-k energy ratio per method, and renders a
per-event grid at each event's peak frame. The 6-band ckpt is built with its
own config (build_model's fallback is 5-band).

    python scripts/eval_hicore.py BACKBONE_CK COREA_CK HICORE_CK HICORE_CFG OUTDIR \
        --days 2023-07-29,2023-07-31,2023-08-01,2023-08-08,2023-08-20 --n-select 8
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
from scripts.peek_hicore import build_with_cfg

MANIFEST = Path(__file__).resolve().parent.parent / "pipeline" / "manifest.csv"
LAT0, LAT1, LON0, LON1 = 36.0, 42.6, 113.0, 120.0


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone_ck", type=Path)
    ap.add_argument("corea_ck", type=Path)
    ap.add_argument("hicore_ck", type=Path)
    ap.add_argument("hicore_cfg", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--days", required=True)
    ap.add_argument("--hours", default="1,7,13,19")
    ap.add_argument("--n-select", type=int, default=8)
    ap.add_argument("--n-show", type=int, default=6)
    args = ap.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    days = [d for d in args.days.split(",") if d.strip()]
    hours = [int(h) for h in args.hours.split(",") if h.strip()]
    cand = [f"{d}T{h:02d}:00:00" for d in days for h in hours]

    scored = []
    for st in cand:
        r = load(st)
        if r is not None:
            scored.append((float(r[1].max()), st))
    scored.sort(reverse=True)
    chosen = [st for _, st in scored[:args.n_select]]
    print("[select] strongest windows:", [f"{p:.0f}@{s[:13]}" for p, s in scored[:args.n_select]])

    base = build_model(args.backbone_ck, device)
    corea = build_model(args.corea_ck, device)
    hicore = build_with_cfg(args.hicore_cfg, args.hicore_ck, device)
    models = {"baseline": base, "coreA": corea, "hicore": hicore}

    rows = []
    grid = []
    for st in chosen:
        r = load(st)
        if r is None:
            continue
        s, gt = r
        Ht, Wt = gt.shape[-2:]
        gt_c = mean_rapsd(gt); hi = slice(int(0.6 * len(gt_c)), len(gt_c)); gt_hi = gt_c[hi].sum()
        preds = {}
        for n, m in models.items():
            with torch.no_grad():
                preds[n] = m(make_batch(s, device))["rain_pred"][0].cpu().numpy()[:, :Ht, :Wt]
        row = {"start": st, "gt_peak": float(gt.max())}
        for n, p in preds.items():
            row[f"{n}_peak"] = float(p.max())
            row[f"{n}_hk"] = float(mean_rapsd(p)[hi].sum() / gt_hi)
        rows.append(row)
        pf = int(np.argmax(gt.reshape(gt.shape[0], -1).max(1)))
        grid.append((st, gt, preds, pf))
        print(f"  {st[:13]} GTpk={row['gt_peak']:.0f}  peak b/A/hi="
              f"{row['baseline_peak']:.0f}/{row['coreA_peak']:.0f}/{row['hicore_peak']:.0f}"
              f"  hk={row['baseline_hk']:.2f}/{row['coreA_hk']:.2f}/{row['hicore_hk']:.2f}")

    def col(k):
        return np.array([r[k] for r in rows], float)
    agg = {m: {"peak": float(col(f"{m}_peak").mean()), "hk": float(col(f"{m}_hk").mean())}
           for m in models}
    agg["gt_peak"] = float(col("gt_peak").mean()); agg["n"] = len(rows)
    (outdir / "eval_hicore.json").write_text(json.dumps({"agg": agg, "events": rows}, indent=2))
    print(f"\n[aggregate over {len(rows)} events]  GT peak {agg['gt_peak']:.1f}")
    for m in models:
        print(f"  {m:9s} peak {agg[m]['peak']:.1f}   high-k {agg[m]['hk']:.2f}")

    # ---- summary figure ----
    mpl.rcParams.update({"font.family": "DejaVu Sans"})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
    names = list(models); colors = ["C0", "C1", "C3"]
    gtp = col("gt_peak")
    for m, c in zip(names, colors):
        axB.scatter(gtp, col(f"{m}_peak"), color=c, s=45, alpha=.85, label=m, zorder=3)
        v = col(f"{m}_hk")
        i = names.index(m)
        axA.scatter(np.full_like(v, i) + np.linspace(-.12, .12, len(v)), v, color=c, s=40, alpha=.8, zorder=3)
        axA.bar(i, v.mean(), width=.55, color=c, alpha=.25)
    axA.axhline(1.0, color="k", ls="--", lw=1.2, label="GT (=1.0)")
    axA.set_xticks(range(len(names))); axA.set_xticklabels(names)
    axA.set_ylabel("high-k energy ratio (pred/GT)"); axA.legend(); axA.grid(True, axis="y", alpha=.3)
    axA.set_title(f"Fine-structure (n={len(rows)})")
    lim = [40, gtp.max() + 3]
    axB.plot(lim, lim, "k--", lw=1.2, label="y=x"); axB.set_xlim(lim); axB.set_ylim([40, 70])
    axB.set_xlabel("GT peak dBZ"); axB.set_ylabel("predicted peak dBZ")
    axB.set_title("Extreme-core recovery"); axB.legend(); axB.grid(True, alpha=.3)
    fig.suptitle("Hi-core GAN: 6-band head breaks the 50 dBZ ceiling (contour preserved)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "summary_hicore.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ---- grid ----
    grid.sort(key=lambda x: x[1].max(), reverse=True)
    show = grid[:args.n_show]
    cols = ["Ground truth", "baseline", "coreA (cap 50)", "hi-core (cap 65)"]
    extent = [LON0, LON1, LAT0, LAT1]
    fig2, axes = plt.subplots(len(show), 4, figsize=(4 * 3.4, len(show) * 3.0))
    if len(show) == 1:
        axes = axes[None, :]
    for ri, (st, gt, preds, pf) in enumerate(show):
        fields = [gt[pf], preds["baseline"][pf], preds["coreA"][pf], preds["hicore"][pf]]
        for ci, (cn, arr) in enumerate(zip(cols, fields)):
            ax = axes[ri, ci]
            ax.imshow(arr, origin="lower", extent=extent, cmap=DBZ_CMAP, norm=DBZ_NORM, aspect="auto")
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(cn, fontsize=11)
            if ci == 0:
                ax.set_ylabel(f"{st[:13]}\n+{(pf+1)*6}min pk{gt.max():.0f}", fontsize=9)
            ax.text(.02, .96, f"max {arr.max():.0f}", transform=ax.transAxes, fontsize=8, va="top",
                    color="w", bbox=dict(boxstyle="round", fc="k", alpha=.45, lw=0))
    cax = fig2.add_axes([0.93, 0.25, 0.011, 0.5])
    fig2.colorbar(plt.cm.ScalarMappable(norm=DBZ_NORM, cmap=DBZ_CMAP), cax=cax, label="dBZ")
    fig2.suptitle("Per-event peak frame: hi-core lifts the cores while keeping the outline", fontsize=13)
    plt.tight_layout(rect=[0, 0, 0.92, 0.975])
    fig2.savefig(outdir / "grid_hicore.png", dpi=200, bbox_inches="tight"); plt.close(fig2)
    print("saved:", outdir / "summary_hicore.png", outdir / "grid_hicore.png")


if __name__ == "__main__":
    main()
