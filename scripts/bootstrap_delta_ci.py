"""Paired bootstrap confidence intervals for model metric differences.

Consumes per-window rows emitted by `scripts/eval_bootstrap_ci.py` and reports
candidate-minus-baseline deltas under date-level paired resampling.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_bootstrap_ci import REPORT_KEYS, aggregate_rows


DEFAULT_COMPARISONS = (
    ("ab2_p7d", "ab1_p7d"),
    ("ab3_era5", "ab1_p7d"),
    ("ab3_era5", "ab2_p7d"),
    ("xcoreA", "ab1_p7d"),
    ("xcoreB", "ab1_p7d"),
    ("xcoreA", "ab2_p7d"),
    ("xcoreB", "ab2_p7d"),
)


def _by_date(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row["date"])].append(row)
    return dict(out)


def _delta(candidate_rows: list[dict[str, Any]],
           baseline_rows: list[dict[str, Any]]) -> dict[str, float]:
    cand = aggregate_rows(candidate_rows)
    base = aggregate_rows(baseline_rows)
    return {key: float(cand[key]) - float(base[key]) for key in REPORT_KEYS}


def paired_delta_ci(baseline_rows: list[dict[str, Any]],
                    candidate_rows: list[dict[str, Any]],
                    baseline_name: str,
                    candidate_name: str,
                    split: str,
                    n_boot: int = 1000,
                    seed: int = 0) -> dict[str, Any]:
    base_by_day = _by_date(baseline_rows)
    cand_by_day = _by_date(candidate_rows)
    if set(base_by_day) != set(cand_by_day):
        raise ValueError("paired delta bootstrap requires the same date set")
    days = sorted(base_by_day)
    if not days:
        raise ValueError("no dates to bootstrap")

    estimate = _delta(candidate_rows, baseline_rows)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {key: [] for key in REPORT_KEYS}
    for _ in range(int(n_boot)):
        sample_base: list[dict[str, Any]] = []
        sample_cand: list[dict[str, Any]] = []
        for day in rng.choice(days, size=len(days), replace=True):
            sample_base.extend(base_by_day[str(day)])
            sample_cand.extend(cand_by_day[str(day)])
        d = _delta(sample_cand, sample_base)
        for key in REPORT_KEYS:
            samples[key].append(float(d[key]))

    ci95: dict[str, list[float]] = {}
    for key, vals in samples.items():
        arr = np.asarray([v for v in vals if v == v], dtype=np.float64)
        if arr.size == 0:
            ci95[key] = [float("nan"), float("nan")]
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            ci95[key] = [round(float(lo), 6), round(float(hi), 6)]

    return {
        "comparison": f"{candidate_name} - {baseline_name}",
        "candidate": candidate_name,
        "baseline": baseline_name,
        "split": split,
        "n_days": len(days),
        "n_windows_baseline": len(baseline_rows),
        "n_windows_candidate": len(candidate_rows),
        "estimate": estimate,
        "ci95": ci95,
    }


def _fmt(report: dict[str, Any], key: str) -> str:
    est = float(report["estimate"][key])
    lo, hi = report["ci95"][key]
    return f"{est:.4f} [{float(lo):.4f}, {float(hi):.4f}]"


def format_delta_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Pluvian Paired Bootstrap Delta Confidence Intervals",
        "",
        "Deltas are candidate minus baseline. Bootstrap unit: manifest date, paired between models.",
        "Positive CSI/FSS deltas and negative CRPS deltas are favorable.",
        "",
        "| comparison | split | days | csi1 | csi10 | csi30 | fss3 | fss11 | crps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            "| {comparison} | {split} | {days:d} | {csi1} | {csi10} | {csi30} | "
            "{fss3} | {fss11} | {crps} |".format(
                comparison=report["comparison"], split=report["split"],
                days=int(report["n_days"]),
                csi1=_fmt(report, "csi_1mm"),
                csi10=_fmt(report, "csi_10mm"),
                csi30=_fmt(report, "csi_30mm"),
                fss3=_fmt(report, "fss_3px"),
                fss11=_fmt(report, "fss_11px"),
                crps=_fmt(report, "crps"),
            )
        )
    return "\n".join(lines) + "\n"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_comparison(value: str) -> tuple[str, str]:
    if "-" not in value:
        raise ValueError("comparison must be CANDIDATE-BASELINE")
    candidate, baseline = value.split("-", 1)
    return candidate, baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--comparison", action="append",
                    help="CANDIDATE-BASELINE; default phase7d comparisons")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    comparisons = [_parse_comparison(v) for v in args.comparison] if args.comparison else list(DEFAULT_COMPARISONS)
    reports: list[dict[str, Any]] = []
    for split in args.split:
        for candidate, baseline in comparisons:
            base_path = args.rows_dir / f"{baseline}_{split}_rows.json"
            cand_path = args.rows_dir / f"{candidate}_{split}_rows.json"
            if not base_path.exists() or not cand_path.exists():
                print(f"[skip] missing rows for {candidate}-{baseline} {split}")
                continue
            reports.append(paired_delta_ci(
                _load_rows(base_path), _load_rows(cand_path),
                baseline_name=baseline, candidate_name=candidate, split=split,
                n_boot=args.n_boot, seed=args.seed,
            ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(format_delta_markdown(reports), encoding="utf-8")
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"[wrote] {args.out}")
    print(f"[wrote] {json_path}")


if __name__ == "__main__":
    main()
