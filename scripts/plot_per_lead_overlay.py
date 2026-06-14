"""Overlay per-lead diagnostics from multiple eval_per_lead JSON files."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    path: Path
    key: str = "ab2"


@dataclass(frozen=True)
class LeadSeries:
    label: str
    split: str
    lead_min: list[int]
    values: list[float]
    source: Path


def parse_series_spec(text: str) -> SeriesSpec:
    if "=" not in text:
        raise ValueError("series spec must be label=path[:key]")
    label, rest = text.split("=", 1)
    key = "ab2"
    path_text = rest
    if ":" in rest:
        maybe_path, maybe_key = rest.rsplit(":", 1)
        if maybe_key:
            path_text = maybe_path
            key = maybe_key
    if not label or not path_text:
        raise ValueError("series spec must be label=path[:key]")
    return SeriesSpec(label=label, path=Path(path_text), key=key)


def _metric_values(data: dict[str, Any], key: str, metric: str, threshold: str) -> list[float]:
    if key not in data:
        raise KeyError(f"missing model key {key!r}")
    block = data[key]
    if metric == "csi":
        values = block["csi"][threshold]
    elif metric == "fss":
        values = block["fss"][threshold]
    else:
        raise ValueError("metric must be 'csi' or 'fss'")
    return [float(v) for v in values]


def load_series_specs(specs: list[SeriesSpec], metric: str = "csi",
                      threshold: str = "30", skip_missing: bool = True) -> list[LeadSeries]:
    series: list[LeadSeries] = []
    for spec in specs:
        if not spec.path.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(spec.path)
        data = json.loads(spec.path.read_text(encoding="utf-8"))
        lead_min = [int(v) for v in data["lead_min"]]
        values = _metric_values(data, spec.key, metric, threshold)
        series.append(LeadSeries(
            label=spec.label,
            split=str(data.get("split", spec.path.stem)),
            lead_min=lead_min,
            values=values,
            source=spec.path,
        ))
    return series


def _plot(series: list[LeadSeries], out_path: Path, metric: str,
          threshold: str, baseline_label: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not series:
        raise ValueError("no series loaded")
    baseline = next((row for row in series if row.label == baseline_label), series[0])
    baseline_values = baseline.values
    split = series[0].split
    ylabel = f"{metric.upper()}@{threshold}"
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.4), sharex=True)
    for row in series:
        axes[0].plot(row.lead_min, row.values, marker="o", linewidth=1.5, markersize=3.8, label=row.label)
        if row is baseline:
            continue
        delta = [v - b for v, b in zip(row.values, baseline_values)]
        axes[1].plot(row.lead_min, delta, marker="o", linewidth=1.3, markersize=3.5, label=f"{row.label} - {baseline.label}")
    for ax in axes:
        ax.axvspan(18, 42, color="#f6ad55", alpha=0.18, linewidth=0)
        ax.grid(True, color="#dddddd", linewidth=0.7)
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel(ylabel)
    axes[1].axhline(0.0, color="#222222", linewidth=0.9, linestyle="--")
    axes[1].set_ylabel(f"Delta {ylabel}")
    axes[1].set_xlabel("lead time (min)")
    fig.suptitle(f"Per-lead {ylabel} overlay: {split}", fontsize=12)
    fig.text(0.5, 0.02, "Orange band marks the +18..+42 min short-middle lead window.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True,
                    help="label=path[:key], where key is usually ab1 or ab2")
    ap.add_argument("--metric", choices=("csi", "fss"), default="csi")
    ap.add_argument("--threshold", default="30")
    ap.add_argument("--baseline", default="ab1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--strict", action="store_true", help="fail on missing series files")
    args = ap.parse_args()

    specs = [parse_series_spec(item) for item in args.series]
    series = load_series_specs(
        specs, metric=args.metric, threshold=args.threshold,
        skip_missing=not args.strict,
    )
    _plot(series, args.out, args.metric, args.threshold, args.baseline)
    print(f"[wrote] {args.out}")
    print(f"[wrote] {args.out.with_suffix('.pdf')}")
    print("[series] " + ", ".join(row.label for row in series))


if __name__ == "__main__":
    main()
