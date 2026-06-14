"""Summarize Phase 7d holdout metrics and per-lead route diagnostics.

This script is deliberately read-only. It consumes metric/per-lead JSON files
already produced by eval jobs and writes a compact Markdown report for route
decisions after ab3/Track C outputs arrive.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPLITS = ("event_test", "test_robust")
METRIC_MODELS = (
    "ablation_2_p7d_droppwv",
    "ablation_2_p7d_xcoreA",
    "ablation_2_p7d_xcoreB",
    "ablation_3_era5_p7c",
    "ablation_3b_era5_budget_p7c",
)
PER_LEAD_SOURCES = (
    ("ab2_p7d", "ckpt/figures/per_lead/per_lead_{split}.json"),
    ("xcoreA", "ckpt/figures/xcore_per_lead/xcoreA/per_lead_{split}.json"),
    ("xcoreB", "ckpt/figures/xcore_per_lead/xcoreB/per_lead_{split}.json"),
    ("ab3_era5", "ckpt/figures/ab3_per_lead/per_lead_{split}.json"),
    ("ab3b_budget", "ckpt/figures/ab3b_per_lead/per_lead_{split}.json"),
)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any) -> str:
    if value is None:
        return "pending"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def summarize_per_lead(path: Path, model: str,
                       threshold: str = "30",
                       short_min: int = 18,
                       short_max: int = 42) -> dict[str, Any]:
    data = _load_json(path)
    if data is None:
        return {
            "model": model,
            "split": path.stem.replace("per_lead_", ""),
            "status": "pending",
        }

    lead_min = data["lead_min"]
    base = data["ab1"]["csi"][threshold]
    candidate = data["ab2"]["csi"][threshold]
    deltas = [float(b) - float(a) for a, b in zip(base, candidate)]
    short = [d for lead, d in zip(lead_min, deltas)
             if short_min <= int(lead) <= short_max]
    return {
        "model": model,
        "split": data.get("split", path.stem.replace("per_lead_", "")),
        "status": "ok",
        "mean_delta": round(sum(deltas) / len(deltas), 4),
        "short_delta": round(sum(short) / len(short), 4) if short else None,
        "min_delta": round(min(deltas), 4),
        "max_delta": round(max(deltas), 4),
        "positive_leads": sum(1 for d in deltas if d > 0),
        "n_leads": len(deltas),
        "source": str(path),
    }


def _metric_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for model in METRIC_MODELS:
        for split in SPLITS:
            path = root / "ckpt" / "eval" / model / split / "metric.json"
            data = _load_json(path)
            row = {"model": model, "split": split, "source": str(path)}
            if data is None:
                row["status"] = "pending"
            else:
                row.update({
                    "status": "ok",
                    "csi1": data.get("csi_1mm"),
                    "csi5": data.get("csi_5mm"),
                    "csi10": data.get("csi_10mm"),
                    "csi30": data.get("csi_30mm"),
                    "fss3": data.get("fss_3px"),
                    "fss11": data.get("fss_11px"),
                    "crps": data.get("crps"),
                })
            rows.append(row)
    return rows


def _per_lead_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for model, template in PER_LEAD_SOURCES:
        for split in SPLITS:
            rows.append(summarize_per_lead(
                root / template.format(split=split), model=model,
            ))
    return rows


def build_report(root: Path | str = Path(".")) -> str:
    root = Path(root)
    metric_rows = _metric_rows(root)
    per_lead_rows = _per_lead_rows(root)

    lines = [
        "# Pluvian Phase 7d Route Summary",
        "",
        "Read-only summary from existing eval/per-lead JSON files.",
        "",
        "## Aggregate Holdout Metrics",
        "",
        "| model | split | status | csi1 | csi5 | csi10 | csi30 | fss3 | fss11 | crps |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            "| {model} | {split} | {status} | {csi1} | {csi5} | {csi10} | "
            "{csi30} | {fss3} | {fss11} | {crps} |".format(
                model=row["model"], split=row["split"], status=row["status"],
                csi1=_fmt(row.get("csi1")), csi5=_fmt(row.get("csi5")),
                csi10=_fmt(row.get("csi10")), csi30=_fmt(row.get("csi30")),
                fss3=_fmt(row.get("fss3")), fss11=_fmt(row.get("fss11")),
                crps=_fmt(row.get("crps")),
            )
        )

    lines.extend([
        "",
        "## Per-Lead CSI30 Delta vs ab1",
        "",
        "`csi30_short_delta` averages leads +18..+42 min.",
        "",
        "| model | split | status | csi30_mean_delta | csi30_short_delta | min_delta | max_delta | positive_leads |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in per_lead_rows:
        pos = "pending"
        if row.get("status") == "ok":
            pos = f"{row['positive_leads']}/{row['n_leads']}"
        lines.append(
            "| {model} | {split} | {status} | {mean_delta} | {short_delta} | "
            "{min_delta} | {max_delta} | {positive_leads} |".format(
                model=row["model"], split=row["split"], status=row["status"],
                mean_delta=_fmt(row.get("mean_delta")),
                short_delta=_fmt(row.get("short_delta")),
                min_delta=_fmt(row.get("min_delta")),
                max_delta=_fmt(row.get("max_delta")),
                positive_leads=pos,
            )
        )

    ab3_ready = any(
        r["model"] == "ablation_3_era5_p7c" and r["status"] == "ok"
        for r in metric_rows
    )
    ab3b_ready = any(
        r["model"] == "ablation_3b_era5_budget_p7c" and r["status"] == "ok"
        for r in metric_rows
    )
    lines.extend([
        "",
        "## Route Gate",
        "",
        "- ab3 ERA5 holdout status: " + ("ready" if ab3_ready else "pending"),
        "- ab3b budget holdout status: " + ("ready" if ab3b_ready else "pending"),
        "- Decision rule: do not promote an experiment that fixes csi30 by materially damaging csi1/csi10/FSS/CRPS.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    report = build_report(args.root)
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
