"""Evaluate water-budget residual diagnostics for trained Pluvian checkpoints.

This is a physical-consistency diagnostic, not a forecast-skill metric. It
loads existing checkpoints, runs them on fixed manifest splits, aligns their
predicted PWV and rain-rate fields to the ERA5/radar overlap grid, and
summarizes the water-budget residual.

Importing this module does not require torch; GPU dependencies are imported only
from the CLI evaluation path.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


BIN_SPECS: tuple[tuple[str, float, float | None], ...] = (
    ("0-1", 0.0, 1.0),
    ("1-5", 1.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10-30", 10.0, 30.0),
    (">=30", 30.0, None),
)
BIN_LABELS = tuple(label for label, _, _ in BIN_SPECS)


def _finite_or_none(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    v = float(value)
    if not np.isfinite(v):
        return None
    return round(v, digits)


def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return None
    return _finite_or_none(float(np.mean(vals)))


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-window budget residual rows by model and split."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["split"]))].append(row)

    summary: list[dict[str, Any]] = []
    for (model, split), group in sorted(grouped.items()):
        bin_counts = {label: 0 for label in BIN_LABELS}
        bin_abs_sums = {label: 0.0 for label in BIN_LABELS}
        for row in group:
            for label in BIN_LABELS:
                bin_counts[label] += int(row.get("bin_counts", {}).get(label, 0))
                bin_abs_sums[label] += float(row.get("bin_abs_sums", {}).get(label, 0.0))

        bin_abs_mean = {}
        for label in BIN_LABELS:
            if bin_counts[label] > 0:
                bin_abs_mean[label] = _finite_or_none(bin_abs_sums[label] / bin_counts[label])
            else:
                bin_abs_mean[label] = None

        days = {str(row.get("time", ""))[:10] for row in group if row.get("time")}
        summary.append({
            "model": model,
            "split": split,
            "n_windows": len(group),
            "n_days": len(days),
            "abs_residual_mean": _mean(group, "abs_residual_mean"),
            "residual_rmse": _mean(group, "residual_rmse"),
            "abs_residual_p95": _mean(group, "abs_residual_p95"),
            "loss_mean": _mean(group, "loss_mean"),
            "rain_rate_mean": _mean(group, "rain_rate_mean"),
            "dpwv_abs_mean": _mean(group, "dpwv_abs_mean"),
            "div_abs_mean": _mean(group, "div_abs_mean"),
            "precip_abs_mean": _mean(group, "precip_abs_mean"),
            "bin_abs_mean": bin_abs_mean,
        })
    return summary


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "--"
    v = float(value)
    if not np.isfinite(v):
        return "--"
    return f"{v:.{digits}f}"


def format_markdown(summary: list[dict[str, Any]]) -> str:
    lines = [
        "# Pluvian Water-Budget Residual Diagnostics",
        "",
        "This is a physical-consistency diagnostic, not a forecast-skill metric. Lower absolute residuals mean the predicted PWV tendency, ERA5 moisture-flux proxy, and predicted precipitation are more mutually consistent under the configured water-budget proxy.",
        "",
        "| model | split | windows | days | mean | rmse | p95 abs | budget loss | rain mm/h | abs dPWV/dt | abs div(qV) | abs P |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {model} | {split} | {windows:d} | {days:d} | {mean} | {rmse} | {p95} | {loss} | {rain} | {dpwv} | {div} | {precip} |".format(
                model=row["model"], split=row["split"],
                windows=int(row["n_windows"]), days=int(row["n_days"]),
                mean=_fmt(row.get("abs_residual_mean")),
                rmse=_fmt(row.get("residual_rmse")),
                p95=_fmt(row.get("abs_residual_p95")),
                loss=_fmt(row.get("loss_mean")),
                rain=_fmt(row.get("rain_rate_mean")),
                dpwv=_fmt(row.get("dpwv_abs_mean")),
                div=_fmt(row.get("div_abs_mean")),
                precip=_fmt(row.get("precip_abs_mean")),
            )
        )

    lines.extend([
        "",
        "## Residual by Predicted Rain-Rate Bin",
        "",
        "Values are mean absolute residual in mm/s, binned by predicted rain rate in mm/h on the ERA5/radar overlap grid.",
        "",
        "| model | split | " + " | ".join(BIN_LABELS) + " |",
        "|---|---|" + "---:|" * len(BIN_LABELS),
    ])
    for row in summary:
        bins = row.get("bin_abs_mean", {})
        values = " | ".join(_fmt(bins.get(label)) for label in BIN_LABELS)
        lines.append(f"| {row['model']} | {row['split']} | {values} |")
    return "\n".join(lines) + "\n"


def make_plot(summary: list[dict[str, Any]], out_path: Path) -> None:
    """Write a compact bar plot of mean absolute residual by model/split."""
    if not summary:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{row['model']}\n{row['split']}" for row in summary]
    values = [float(row["abs_residual_mean"] or 0.0) for row in summary]

    width = max(7.0, 0.7 * len(summary))
    fig, ax = plt.subplots(figsize=(width, 4.2))
    x = np.arange(len(summary))
    ax.bar(x, values, color="#4b8bbe", edgecolor="#1f2d3a", linewidth=0.6)
    ax.set_ylabel("mean |budget residual| (mm/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _parse_ckpt_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    name, path = value.split("=", 1)
    return name, Path(path)


def _bin_stats(abs_residual, rain_rate) -> tuple[dict[str, int], dict[str, float]]:
    counts = {}
    sums = {}
    for label, lo, hi in BIN_SPECS:
        if hi is None:
            mask = rain_rate >= lo
        else:
            mask = (rain_rate >= lo) & (rain_rate < hi)
        count = int(mask.sum().item())
        counts[label] = count
        sums[label] = float(abs_residual[mask].sum().item()) if count else 0.0
    return counts, sums


def _evaluate_checkpoint(
    ckpt: Path,
    model_name: str,
    split: str,
    batch_size: int,
    max_windows: int | None,
    level_idx: int | None,
    delta_p_pa: float,
    smooth_sigma: float,
    huber_delta: float,
) -> list[dict[str, Any]]:
    import sys
    import torch
    from torch.utils.data import DataLoader

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from model.losses import WaterBudgetLoss
    from pipeline.data_loader import NPJDataset
    from scripts.eval import _load_ckpt
    from scripts.train import (
        _budget_loss_inputs,
        _collate,
        _starts_for_splits,
        _to_device,
        split_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_ckpt(ckpt, device)
    manifest = repo_root / cfg["data"]["manifest"]
    starts = _starts_for_splits(manifest, [split])
    if max_windows is not None:
        starts = starts[:max(0, int(max_windows))]
    load_mfd = bool(cfg["model"].get("mfd_channel_enabled", False))
    ds = NPJDataset(
        manifest=manifest,
        window_minutes=cfg["data"]["window_minutes"],
        starts=starts,
        drop_pwv=cfg["data"].get("drop_pwv_val", False),
        load_mfd=load_mfd,
        load_era5=True,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, cfg["data"].get("num_workers", 1) // 2),
        collate_fn=_collate,
    )
    T_in = cfg["model"]["input_frames"]
    T_out = cfg["model"]["forecast_frames"]
    mfd_ch = cfg["model"]["mfd_channels"] if cfg["model"].get("mfd_channel_enabled", False) else 0
    budget_cfg = cfg.get("loss", {}).get("budget", {})
    budget_level = int(level_idx if level_idx is not None else budget_cfg.get("level_idx", 1))
    budget_loss = WaterBudgetLoss(
        delta_p_pa=delta_p_pa,
        smooth_sigma=smooth_sigma,
        huber_delta=huber_delta,
        reduction="none",
    ).to(device)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            model_in, rain_tgt, era5_fut = split_batch(batch, T_in, T_out, mfd_ch)
            out = model(model_in)
            rain_pred = out["rain_pred"].float()
            pwv_pred = out["pwv_pred"].float()
            if rain_pred.shape[-2:] != rain_tgt.shape[-2:]:
                Ht, Wt = rain_tgt.shape[-2:]
                rain_pred = rain_pred[..., :Ht, :Wt].contiguous()
                pwv_pred = pwv_pred[..., :Ht, :Wt].contiguous()

            budget_inputs = _budget_loss_inputs(batch, pwv_pred, rain_pred, era5_fut, budget_level)
            out_budget = budget_loss(
                pwv_pred=budget_inputs["pwv_budget"],
                rain_pred=budget_inputs["rain_budget"],
                u_era5=budget_inputs["u_budget"],
                v_era5=budget_inputs["v_budget"],
                q_era5=budget_inputs["q_budget"],
                lat=budget_inputs["lat_budget"],
                lon=budget_inputs["lon_budget"],
            )
            residual = out_budget["budget_residual"].float()
            loss_elem = out_budget["loss"].float()
            rain_budget = budget_inputs["rain_budget"].float()
            times = batch["time"] if isinstance(batch["time"], list) else [str(batch["time"])]

            for b, t_str in enumerate(times):
                abs_r = residual[b].abs()
                counts, sums = _bin_stats(abs_r, rain_budget[b])
                row = {
                    "model": model_name,
                    "split": split,
                    "time": str(t_str),
                    "date": str(t_str)[:10],
                    "abs_residual_mean": _finite_or_none(abs_r.mean().item()),
                    "residual_rmse": _finite_or_none(torch.sqrt((residual[b] ** 2).mean()).item()),
                    "abs_residual_p95": _finite_or_none(torch.quantile(abs_r.reshape(-1), 0.95).item()),
                    "loss_mean": _finite_or_none(loss_elem[b].mean().item()),
                    "rain_rate_mean": _finite_or_none(rain_budget[b].mean().item()),
                    "dpwv_abs_mean": _finite_or_none(out_budget["dpwv_dt"][b].abs().mean().item()),
                    "div_abs_mean": _finite_or_none(out_budget["div_qV_mm_s"][b].abs().mean().item()),
                    "precip_abs_mean": _finite_or_none(out_budget["P_mm_s"][b].abs().mean().item()),
                    "bin_counts": counts,
                    "bin_abs_sums": {k: _finite_or_none(v) for k, v in sums.items()},
                }
                rows.append(row)
                if max_windows is not None and len(rows) >= max_windows:
                    return rows
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="MODEL=ckpt/path.pt; can be repeated")
    ap.add_argument("--split", action="append", required=True,
                    help="manifest split label; can be repeated")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_windows", type=int, default=None,
                    help="optional quick-look cap per model/split")
    ap.add_argument("--level_idx", type=int, default=None,
                    help="ERA5 pressure-level index; default from checkpoint config or 1")
    ap.add_argument("--delta_p_pa", type=float, default=30000.0)
    ap.add_argument("--smooth_sigma", type=float, default=2.0)
    ap.add_argument("--huber_delta", type=float, default=1e-4)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for ckpt_arg in args.ckpt:
        model_name, ckpt = _parse_ckpt_arg(ckpt_arg)
        for split in args.split:
            print(f"[budget] model={model_name} split={split} ckpt={ckpt}", flush=True)
            rows = _evaluate_checkpoint(
                ckpt=ckpt,
                model_name=model_name,
                split=split,
                batch_size=args.batch_size,
                max_windows=args.max_windows,
                level_idx=args.level_idx,
                delta_p_pa=args.delta_p_pa,
                smooth_sigma=args.smooth_sigma,
                huber_delta=args.huber_delta,
            )
            all_rows.extend(rows)
            stem = f"{model_name}_{split}"
            with open(args.out_dir / f"{stem}_rows.json", "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, allow_nan=False)

    summary = summarize_rows(all_rows)
    with open(args.out_dir / "budget_residual_rows.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, allow_nan=False)
    with open(args.out_dir / "budget_residual_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    md = format_markdown(summary)
    (args.out_dir / "budget_residual_summary.md").write_text(md, encoding="utf-8")
    make_plot(summary, args.out_dir / "budget_residual_summary.png")
    make_plot(summary, args.out_dir / "budget_residual_summary.pdf")
    print(md)


if __name__ == "__main__":
    main()
