"""Pluvian ablation comparison curves — publication grade.

ab1 (radar only, 60 epochs) vs ab2 (radar + PWV-concat + ERA5, in progress).

Output: figures/ablation_comparison.png and ablation_comparison.pdf
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def load(path):
    return json.load(open(path))


def col(data, key):
    return np.array([float(d[key]) for d in data])


def main():
    # House style — readable in print
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    ab1 = load("/Data/tanh/npj/ckpt/ablation_1_radar_only/metric_log.json")
    ab2 = load("/tmp/ab2_metric.json")

    ep1 = col(ab1, "epoch")
    ep2 = col(ab2, "epoch")

    # Plot four panels
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    (axA, axB), (axC, axD) = axes

    C1 = "#1f77b4"   # ab1 (radar only) — blue
    C2 = "#d62728"   # ab2 (+PWV concat + ERA5) — red

    # ===== csi @ 10 mm/h =====
    axA.plot(ep1, col(ab1, "val/csi_10mm"), "-o", color=C1, ms=3, lw=1.5,
             label="ab1: radar only")
    axA.plot(ep2, col(ab2, "val/csi_10mm"), "-s", color=C2, ms=3, lw=1.5,
             label="ab2: + PWV-concat + ERA5")
    axA.set_ylabel("CSI @ 10 mm/h")
    axA.set_title("(a) Moderate-to-heavy rain detection", loc="left")
    axA.legend(loc="lower right", frameon=False)
    axA.set_ylim(0, 0.7)

    # ===== csi @ 30 mm/h =====
    axB.plot(ep1, col(ab1, "val/csi_30mm"), "-o", color=C1, ms=3, lw=1.5,
             label="ab1")
    axB.plot(ep2, col(ab2, "val/csi_30mm"), "-s", color=C2, ms=3, lw=1.5,
             label="ab2")
    axB.set_ylabel("CSI @ 30 mm/h")
    axB.set_title("(b) Extreme-rain detection", loc="left")
    axB.legend(loc="lower right", frameon=False)
    axB.set_ylim(0, 0.30)

    # ===== val data loss =====
    axC.plot(ep1, col(ab1, "val/data"), "-o", color=C1, ms=3, lw=1.5)
    axC.plot(ep2, col(ab2, "val/data"), "-s", color=C2, ms=3, lw=1.5)
    axC.set_ylabel("val data loss (weighted MSE)")
    axC.set_xlabel("epoch")
    axC.set_title("(c) Validation loss", loc="left")
    axC.set_yscale("log")

    # ===== CRPS =====
    axD.plot(ep1, col(ab1, "val/crps"), "-o", color=C1, ms=3, lw=1.5)
    axD.plot(ep2, col(ab2, "val/crps"), "-s", color=C2, ms=3, lw=1.5)
    axD.set_ylabel("CRPS (lower = better)")
    axD.set_xlabel("epoch")
    axD.set_title("(d) Probabilistic skill (CRPS)", loc="left")

    # Mark ab2 progress with vertical line
    ab2_last_ep = int(ep2[-1])
    for ax in axes.flat:
        ax.axvline(ab2_last_ep, color="grey", lw=0.6, ls=":", alpha=0.7)
    fig.suptitle(
        f"Pluvian ablation 1 vs 2  —  ab2 in progress through epoch {ab2_last_ep}/59  ",
        y=0.995, fontsize=12,
    )

    plt.tight_layout()
    out_png = Path("/Data/tanh/npj/figures/ablation_comparison.png")
    out_pdf = Path("/Data/tanh/npj/figures/ablation_comparison.pdf")
    out_png.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"saved: {out_png}")
    print(f"saved: {out_pdf}")

    # Print summary
    print("\nSummary at ab2's current epoch:", ab2_last_ep)
    print(f"  ab1 ep{ab2_last_ep}: csi@10mm={ab1[ab2_last_ep]['val/csi_10mm']:.4f}, csi@30mm={ab1[ab2_last_ep]['val/csi_30mm']:.4f}, data={ab1[ab2_last_ep]['val/data']:.1f}")
    print(f"  ab2 ep{ab2_last_ep}: csi@10mm={ab2[-1]['val/csi_10mm']:.4f}, csi@30mm={ab2[-1]['val/csi_30mm']:.4f}, data={ab2[-1]['val/data']:.1f}")


if __name__ == "__main__":
    main()
