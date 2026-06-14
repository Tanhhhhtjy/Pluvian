"""Skill-ladder bar chart: 4 models x 3 CSI thresholds x 2 splits.

Story: csi1/csi10 monotonically improve ab1->ab2->ab3->ab3b, but csi30
*reverses* sharply at ab3b — the budget loss damages the extreme band.
Color encoding makes the reversal visually obvious without needing to read
numbers off the y-axis.

Read-only: pulls numbers from the metric JSON files we already have on disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Model order (left-to-right) and visual encoding.
MODELS_BASE = [
    ("ab1", "radar"),
    ("ab2", "+PWV"),
    ("ab3", "+ERA5"),
    ("ab3b", "+budget loss"),
]
MODEL_CDU = ("ab3-cdu", "+CDU decoder")
# ab1->ab3 use a blue ramp (monotonic-fusion story), ab3b uses red (negative),
# ab3-cdu uses green to flag the recovery path on top of ab3 (no budget loss).
COLORS = {
    "ab1": "#bcd6ec",
    "ab2": "#7faedb",
    "ab3": "#2b6cb0",
    "ab3b": "#c53030",
    "ab3-cdu": "#38a169",
}
SPLITS = ["event_test", "test_robust"]
THRESHOLDS = [1, 10, 30]

SOURCES = {
    "ab1":  ("bootstrap_ci", "ab1_p7d_{split}_ci.json"),
    "ab2":  ("bootstrap_ci", "xcoreA_{split}_ci.json"),
    "ab3":  ("eval", "ablation_3_era5_p7c/{split}/metric.json"),
    "ab3b": ("eval", "ablation_3b_era5_budget_p7c/{split}/metric.json"),
    "ab3-cdu": ("eval", "ablation_3_cdu_p7d/{split}/metric.json"),
}


def _read(model: str, split: str, repo_root: Path) -> dict:
    src, pat = SOURCES[model]
    rel = pat.format(split=split)
    if src == "eval":
        path = repo_root / "ckpt" / "eval" / rel
        return json.loads(path.read_text())
    if src == "bootstrap_ci":
        path = repo_root / "ckpt" / "bootstrap_ci" / "phase7d_skill" / rel
        ci = json.loads(path.read_text())
        # Bootstrap CI files store point estimates in the "estimate" sub-dict.
        return ci["estimate"]
    raise ValueError(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path,
                    default=Path("ckpt/figures/paper_skill_bar/skill_ladder.png"))
    ap.add_argument("--include-cdu", action="store_true",
                    help="append ab3-cdu (CDU dual-branch decoder) as a 5th model")
    args = ap.parse_args()

    models = list(MODELS_BASE)
    if args.include_cdu:
        models.append(MODEL_CDU)
    n_models = len(models)

    # collect: vals[split][thr] = list of (model, value)
    vals: dict[str, dict[int, list[float]]] = {sp: {t: [] for t in THRESHOLDS} for sp in SPLITS}
    for sp in SPLITS:
        for model, _label in models:
            metric = _read(model, sp, args.root)
            for t in THRESHOLDS:
                vals[sp][t].append(metric[f"csi_{t}mm"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0 + (1.5 if args.include_cdu else 0.0), 4.4), sharey=True)
    bar_w = 0.8 / n_models  # span ~0.8 of the unit slot, regardless of model count
    center_off = (n_models - 1) / 2.0
    x = np.arange(len(THRESHOLDS))
    for ax, sp in zip(axes, SPLITS):
        for i, (model, label) in enumerate(models):
            ys = [vals[sp][t][i] for t in THRESHOLDS]
            offsets = x + (i - center_off) * bar_w
            bars = ax.bar(offsets, ys, width=bar_w, color=COLORS[model],
                          edgecolor="#2b2b2b", linewidth=0.6,
                          label=f"{model} ({label})")
            # value labels above each bar
            for b, y in zip(bars, ys):
                ax.text(b.get_x() + b.get_width() / 2, y + 0.008,
                        f"{y:.3f}", ha="center", va="bottom",
                        fontsize=7.5, color="#222")
        # mark the csi30 reversal with a dashed connector ab3 -> ab3b
        i_ab3 = next(i for i, (m, _) in enumerate(models) if m == "ab3")
        i_ab3b = next(i for i, (m, _) in enumerate(models) if m == "ab3b")
        i_thr30 = 2
        x_ab3 = x[i_thr30] + (i_ab3 - center_off) * bar_w
        x_ab3b = x[i_thr30] + (i_ab3b - center_off) * bar_w
        y_ab3 = vals[sp][30][i_ab3]
        y_ab3b = vals[sp][30][i_ab3b]
        ax.annotate("", xy=(x_ab3b, y_ab3b + 0.005),
                    xytext=(x_ab3, y_ab3 + 0.005),
                    arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.4))
        drop_pct = (y_ab3 - y_ab3b) / y_ab3 * 100
        ax.text((x_ab3 + x_ab3b) / 2, max(y_ab3, y_ab3b) + 0.04,
                f"-{drop_pct:.0f}% csi30",
                ha="center", color="#c53030", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"CSI@{t} mm" for t in THRESHOLDS], fontsize=10)
        ax.set_title(sp, fontsize=11)
        ax.set_ylim(0, max(0.75, max(vals[sp][1]) * 1.15))
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("CSI")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.92,
                   ncol=2, columnspacing=0.8)
    fig.suptitle(
        "Light/moderate CSI rises monotonically; the budget loss reverses extreme CSI@30 mm",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[skill-bar] wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
