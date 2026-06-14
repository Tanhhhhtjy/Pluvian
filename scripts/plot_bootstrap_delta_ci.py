"""Plot paired bootstrap delta confidence intervals as compact forest plots."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PREFERRED_COMPARISONS = (
    "ab2_p7d - ab1_p7d",
    "ab3_era5 - ab1_p7d",
    "ab3_era5 - ab2_p7d",
    "xcoreA - ab1_p7d",
    "xcoreB - ab1_p7d",
    "xcoreA - ab2_p7d",
    "xcoreB - ab2_p7d",
)
METRIC_LABELS = {
    "csi_1mm": "Delta CSI1",
    "csi_10mm": "Delta CSI10",
    "csi_30mm": "Delta CSI30",
    "fss_3px": "Delta FSS3",
    "crps": "Delta CRPS",
}
FAVORABLE = {
    "csi_1mm": "right",
    "csi_10mm": "right",
    "csi_30mm": "right",
    "fss_3px": "right",
    "crps": "left",
}


@dataclass(frozen=True)
class PlotRow:
    comparison: str
    estimate: float
    lo: float
    hi: float


def plot_rows_for_split(reports: list[dict[str, Any]], split: str,
                        metric: str) -> list[PlotRow]:
    by_cmp = {r["comparison"]: r for r in reports if r.get("split") == split}
    rows: list[PlotRow] = []
    for comparison in PREFERRED_COMPARISONS:
        report = by_cmp.get(comparison)
        if report is None or metric not in report.get("estimate", {}):
            continue
        lo, hi = report["ci95"][metric]
        rows.append(PlotRow(
            comparison=comparison,
            estimate=float(report["estimate"][metric]),
            lo=float(lo), hi=float(hi),
        ))
    return rows


def _plot_split(reports: list[dict[str, Any]], split: str,
                metrics: list[str], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.8), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    comparisons = list(PREFERRED_COMPARISONS)
    y = list(range(len(comparisons)))
    for ax, metric in zip(axes, metrics):
        rows = plot_rows_for_split(reports, split, metric)
        by_cmp = {r.comparison: r for r in rows}
        xs, xerr_lo, xerr_hi = [], [], []
        for cmp_name in comparisons:
            row = by_cmp.get(cmp_name)
            if row is None:
                xs.append(float("nan")); xerr_lo.append(0.0); xerr_hi.append(0.0)
            else:
                xs.append(row.estimate)
                xerr_lo.append(row.estimate - row.lo)
                xerr_hi.append(row.hi - row.estimate)
        color = "#2b6cb0" if FAVORABLE.get(metric) == "right" else "#8a5a00"
        ax.errorbar(xs, y, xerr=[xerr_lo, xerr_hi], fmt="o", color=color,
                    ecolor="#555555", elinewidth=1.2, capsize=3, markersize=4.5)
        ax.axvline(0.0, color="#222222", linewidth=0.9, linestyle="--")
        ax.grid(axis="x", color="#dddddd", linewidth=0.7)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(comparisons, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(f"Paired day-bootstrap deltas: {split}", fontsize=12)
    fig.text(0.5, 0.015, "CSI/FSS: right is favorable; CRPS: left is favorable.",
             ha="center", va="bottom", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta_json", type=Path, default=Path("docs/experiments/pluvian_bootstrap_delta_ci.json"))
    ap.add_argument("--out_dir", type=Path, default=Path("ckpt/figures/bootstrap_delta_ci"))
    ap.add_argument("--split", action="append", default=None)
    ap.add_argument("--metric", action="append", default=None)
    args = ap.parse_args()
    reports = json.loads(args.delta_json.read_text(encoding="utf-8"))
    splits = args.split or ["event_test", "test_robust"]
    metrics = args.metric or ["csi_1mm", "csi_10mm", "csi_30mm", "fss_3px", "crps"]
    for split in splits:
        out_path = args.out_dir / f"paired_delta_ci_{split}.png"
        _plot_split(reports, split, metrics, out_path)
        print(f"[wrote] {out_path}")
        print(f"[wrote] {out_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
