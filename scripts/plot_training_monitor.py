"""Plot a lightweight training monitor from Pluvian train.py logs."""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EPOCH_SUMMARY_RE = re.compile(
    r"^\[ep(?P<epoch>\d+)\] avg "
    r"data=(?P<data>\S+) fss=(?P<fss>\S+) budget=(?P<budget>\S+) "
    r"spec=(?P<spec>\S+) total=(?P<total>\S+)\s+"
    r"\((?P<seconds>\S+)s\)"
)
PROGRESS_RE = re.compile(
    r"^\[ep(?P<epoch>\d+)\|it(?P<iteration>\d+)/(?P<total_iterations>\d+)\].*"
    r"\blr=(?P<lr>\S+)\s+step=(?P<step>\d+)"
)
KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>\S+)")


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    data: float
    fss: float
    budget: float
    spec: float
    total: float
    seconds: float


@dataclass(frozen=True)
class ValidationRow:
    epoch: int
    metrics: dict[str, float]


@dataclass(frozen=True)
class ProgressRow:
    epoch: int
    iteration: int
    total_iterations: int
    lr: float
    step: int


@dataclass(frozen=True)
class ParsedTrainingLog:
    epoch_summaries: list[EpochSummary]
    validation_rows: list[ValidationRow]
    latest_progress: ProgressRow | None


def _float(value: str) -> float:
    if value.lower() == "nan":
        return math.nan
    return float(value)


def _metric_pairs(line: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for match in KEY_VALUE_RE.finditer(line):
        key = match.group("key")
        value = match.group("value")
        try:
            metrics[key] = _float(value)
        except ValueError:
            continue
    return metrics


def parse_training_log(text: str) -> ParsedTrainingLog:
    summaries: list[EpochSummary] = []
    validations: list[ValidationRow] = []
    latest_progress: ProgressRow | None = None
    last_summary_epoch: int | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        progress_match = PROGRESS_RE.match(line)
        if progress_match:
            latest_progress = ProgressRow(
                epoch=int(progress_match.group("epoch")),
                iteration=int(progress_match.group("iteration")),
                total_iterations=int(progress_match.group("total_iterations")),
                lr=_float(progress_match.group("lr")),
                step=int(progress_match.group("step")),
            )
            continue

        epoch_match = EPOCH_SUMMARY_RE.match(line)
        if epoch_match:
            epoch = int(epoch_match.group("epoch"))
            summaries.append(EpochSummary(
                epoch=epoch,
                data=_float(epoch_match.group("data")),
                fss=_float(epoch_match.group("fss")),
                budget=_float(epoch_match.group("budget")),
                spec=_float(epoch_match.group("spec")),
                total=_float(epoch_match.group("total")),
                seconds=_float(epoch_match.group("seconds")),
            ))
            last_summary_epoch = epoch
            continue

        if line.startswith("[val]") and last_summary_epoch is not None:
            validations.append(ValidationRow(
                epoch=last_summary_epoch,
                metrics=_metric_pairs(line),
            ))

    return ParsedTrainingLog(
        epoch_summaries=summaries,
        validation_rows=validations,
        latest_progress=latest_progress,
    )


def estimate_remaining_hours(parsed: ParsedTrainingLog, max_epochs: int) -> float | None:
    if not parsed.epoch_summaries:
        return None
    mean_epoch_seconds = sum(r.seconds for r in parsed.epoch_summaries) / len(parsed.epoch_summaries)
    last_completed = max(r.epoch for r in parsed.epoch_summaries)
    if parsed.latest_progress and parsed.latest_progress.epoch > last_completed:
        fraction_done = parsed.latest_progress.iteration / parsed.latest_progress.total_iterations
        remaining_epochs = max(0.0, max_epochs - parsed.latest_progress.epoch)
        seconds = (1.0 - fraction_done) * mean_epoch_seconds + remaining_epochs * mean_epoch_seconds
    else:
        remaining_epochs = max(0, max_epochs - last_completed)
        seconds = remaining_epochs * mean_epoch_seconds
    return seconds / 3600.0


def _latest_validation(parsed: ParsedTrainingLog) -> ValidationRow | None:
    if not parsed.validation_rows:
        return None
    return parsed.validation_rows[-1]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _best_validation(parsed: ParsedTrainingLog, metric: str, higher_is_better: bool) -> ValidationRow | None:
    rows = [r for r in parsed.validation_rows if metric in r.metrics and not math.isnan(r.metrics[metric])]
    if not rows:
        return None
    return max(rows, key=lambda r: r.metrics[metric]) if higher_is_better else min(rows, key=lambda r: r.metrics[metric])


def build_summary(parsed: ParsedTrainingLog, max_epochs: int, source: Path) -> dict[str, Any]:
    latest_val = _latest_validation(parsed)
    best_csi30 = _best_validation(parsed, "csi_30mm", higher_is_better=True)
    best_crps = _best_validation(parsed, "crps", higher_is_better=False)
    eta_hours = estimate_remaining_hours(parsed, max_epochs)
    return _json_safe({
        "source": str(source),
        "completed_epochs": [r.epoch for r in parsed.epoch_summaries],
        "latest_progress": None if parsed.latest_progress is None else parsed.latest_progress.__dict__,
        "latest_validation": None if latest_val is None else {"epoch": latest_val.epoch, **latest_val.metrics},
        "best_csi30": None if best_csi30 is None else {"epoch": best_csi30.epoch, "csi_30mm": best_csi30.metrics["csi_30mm"]},
        "best_crps": None if best_crps is None else {"epoch": best_crps.epoch, "crps": best_crps.metrics["crps"]},
        "eta_hours": eta_hours,
    })


def _fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "pending"
    return f"{value:.4f}"


def write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    latest_progress = summary.get("latest_progress")
    latest_val = summary.get("latest_validation") or {}
    best_csi30 = summary.get("best_csi30") or {}
    best_crps = summary.get("best_crps") or {}
    eta = summary.get("eta_hours")
    lines = [
        "# ab3 ERA5 Training Monitor",
        "",
        f"Source log: `{summary['source']}`",
        "",
        "## Current Progress",
        "",
    ]
    if latest_progress is None:
        lines.append("- Latest iteration: pending")
    else:
        pct = 100.0 * latest_progress["iteration"] / latest_progress["total_iterations"]
        lines.append(
            "- Latest iteration: epoch {epoch}, {iteration}/{total_iterations} "
            "({pct:.1f}%), lr={lr:.2e}, step={step}".format(
                pct=pct, **latest_progress,
            )
        )
    lines.append(f"- Estimated remaining time to 60 epochs: {_fmt(eta)} hours")
    lines.extend([
        "",
        "## Latest Validation",
        "",
        "| epoch | csi1 | csi10 | csi30 | fss3_30 | crps | spread |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| {epoch} | {csi1} | {csi10} | {csi30} | {fss30} | {crps} | {spread} |".format(
            epoch=latest_val.get("epoch", "pending"),
            csi1=_fmt(latest_val.get("csi_1mm")),
            csi10=_fmt(latest_val.get("csi_10mm")),
            csi30=_fmt(latest_val.get("csi_30mm")),
            fss30=_fmt(latest_val.get("fss_3px_30mm")),
            crps=_fmt(latest_val.get("crps")),
            spread=_fmt(latest_val.get("spread")),
        ),
        "",
        "## Best So Far",
        "",
        f"- Best val csi30: epoch {best_csi30.get('epoch', 'pending')} = {_fmt(best_csi30.get('csi_30mm'))}",
        f"- Best val CRPS: epoch {best_crps.get('epoch', 'pending')} = {_fmt(best_crps.get('crps'))}",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_training(parsed: ParsedTrainingLog, out_path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)
    if not parsed.validation_rows:
        axes[0].text(0.5, 0.5, "No validation rows parsed", ha="center", va="center")
        axes[0].set_axis_off()
        axes[1].set_axis_off()
    else:
        epochs = [row.epoch for row in parsed.validation_rows]
        for metric, label, color in [
            ("csi_1mm", "CSI1", "#2b6cb0"),
            ("csi_10mm", "CSI10", "#2f855a"),
            ("csi_30mm", "CSI30", "#c53030"),
            ("fss_3px_30mm", "FSS3 30mm", "#805ad5"),
        ]:
            values = [row.metrics.get(metric, math.nan) for row in parsed.validation_rows]
            axes[0].plot(epochs, values, marker="o", linewidth=1.4, markersize=3.5, label=label, color=color)
        axes[0].set_ylabel("Skill")
        axes[0].grid(True, color="#dddddd", linewidth=0.7)
        axes[0].legend(loc="best", fontsize=8)

        crps = [row.metrics.get("crps", math.nan) for row in parsed.validation_rows]
        spread = [row.metrics.get("spread", math.nan) for row in parsed.validation_rows]
        axes[1].plot(epochs, crps, marker="o", linewidth=1.4, markersize=3.5, label="CRPS", color="#8a5a00")
        axes[1].plot(epochs, spread, marker="o", linewidth=1.4, markersize=3.5, label="spread", color="#4a5568")
        axes[1].set_ylabel("CRPS / spread")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, color="#dddddd", linewidth=0.7)
        axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("ckpt/figures/training_monitor"))
    ap.add_argument("--prefix", default="ab3_era5")
    ap.add_argument("--title", default="ab3 ERA5 training monitor")
    ap.add_argument("--max_epochs", type=int, default=60)
    args = ap.parse_args()

    parsed = parse_training_log(args.log.read_text(encoding="utf-8", errors="replace"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(parsed, args.max_epochs, args.log)
    json_path = args.out_dir / f"{args.prefix}_training_monitor.json"
    md_path = args.out_dir / f"{args.prefix}_training_monitor.md"
    png_path = args.out_dir / f"{args.prefix}_training_monitor.png"
    json_path.write_text(json.dumps(summary, allow_nan=False, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(summary, md_path)
    plot_training(parsed, png_path, args.title)
    print(f"[wrote] {json_path}")
    print(f"[wrote] {md_path}")
    print(f"[wrote] {png_path}")
    print(f"[wrote] {png_path.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
