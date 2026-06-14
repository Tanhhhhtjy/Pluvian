"""Plot the pooled-CSI extreme-rain diagnostic (exPreCast-inspired).

Story: under neighborhood-tolerant max-pooling, ab3's csi30 recovers (the strong
cores are present but spatially displaced), while ab3b's csi30 stays flat even at
16x16 pooling (the budget loss physically smooths the cores away). Light/moderate
csi1/csi10 are essentially tied, so the cost is localized to the extreme band.

Read-only: consumes ckpt/figures/pooled_csi/pooled_csi.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


POOLS = [1, 4, 16]
MODEL_STYLE = {
    "ab3": {"color": "#2b6cb0", "marker": "o", "label": "ab3 (+ERA5)"},
    "ab3b": {"color": "#c53030", "marker": "s", "label": "ab3b (+budget loss)"},
}


def _series(block: dict, thr: int) -> list[float]:
    return [block[f"csi{thr}_pool{p}"] for p in POOLS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path,
                    default=Path("ckpt/figures/pooled_csi/pooled_csi.json"))
    ap.add_argument("--split", default="event_test")
    ap.add_argument("--out", type=Path,
                    default=Path("ckpt/figures/pooled_csi/pooled_csi_diagnostic.png"))
    args = ap.parse_args()

    data = json.loads(args.json.read_text())[args.split]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    thresholds = [(30, "CSI@30mm (extreme)"), (10, "CSI@10mm (moderate)"),
                  (1, "CSI@1mm (light)")]
    for ax, (thr, title) in zip(axes, thresholds):
        for name, block in data.items():
            st = MODEL_STYLE.get(name, {"color": "#444", "marker": "^", "label": name})
            ax.plot(POOLS, _series(block, thr), marker=st["marker"],
                    color=st["color"], linewidth=1.8, markersize=7,
                    label=st["label"])
        ax.set_xscale("log", base=2)
        ax.set_xticks(POOLS)
        ax.set_xticklabels([f"{p}x{p}" for p in POOLS])
        ax.set_xlabel("max-pool window (neighborhood tolerance)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, color="#dddddd", linewidth=0.7)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("pooled CSI")

    ax0 = axes[0]
    ax0.annotate("cores displaced\n(recovers with pooling)",
                 xy=(16, data["ab3"]["csi30_pool16"]),
                 xytext=(2.0, data["ab3"]["csi30_pool16"] + 0.045),
                 fontsize=7.5, color=MODEL_STYLE["ab3"]["color"], ha="left",
                 arrowprops=dict(arrowstyle="->", color=MODEL_STYLE["ab3"]["color"], lw=0.9))
    ax0.annotate("cores smoothed away\n(flat under pooling)",
                 xy=(16, data["ab3b"]["csi30_pool16"]),
                 xytext=(2.0, data["ab3b"]["csi30_pool16"] - 0.055),
                 fontsize=7.5, color=MODEL_STYLE["ab3b"]["color"], ha="left",
                 arrowprops=dict(arrowstyle="->", color=MODEL_STYLE["ab3b"]["color"], lw=0.9))

    fig.suptitle(
        f"Pooled-CSI diagnostic ({args.split}): the budget loss removes extreme "
        "cores, not just displaces them", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[pooled-csi-plot] wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
