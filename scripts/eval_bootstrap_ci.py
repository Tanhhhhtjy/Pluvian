"""Bootstrap confidence intervals for Pluvian holdout metrics.

The script evaluates one or more checkpoints window-by-window, aggregates CSI
from raw contingency counts, and bootstraps by date. Importing this module does
not require torch; GPU dependencies are imported only by the CLI path.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


THRESHOLDS = (1, 5, 10, 30)
FSS_WINDOWS = (3, 11)
SCALAR_KEYS = ("mae", "crps", "weighted_mse")
REPORT_KEYS = (
    "mae", "crps", "weighted_mse",
    "csi_1mm", "csi_5mm", "csi_10mm", "csi_30mm",
    "fss_3px", "fss_11px",
)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate zero rows")
    out: dict[str, float] = {}
    for key in SCALAR_KEYS:
        vals = [float(r[key]) for r in rows if r.get(key) == r.get(key)]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    for thr in THRESHOLDS:
        h = sum(int(r[f"csi_{thr}mm_hits"]) for r in rows)
        fa = sum(int(r[f"csi_{thr}mm_fa"]) for r in rows)
        miss = sum(int(r[f"csi_{thr}mm_miss"]) for r in rows)
        den = h + fa + miss
        out[f"csi_{thr}mm"] = h / den if den > 0 else float("nan")
    for nbr in FSS_WINDOWS:
        num = sum(float(r[f"fss_{nbr}px_num"]) for r in rows)
        den = sum(float(r[f"fss_{nbr}px_den"]) for r in rows)
        out[f"fss_{nbr}px"] = 1.0 - num / den if den > 0 else float("nan")
    out["n_windows"] = float(len(rows))
    out["n_days"] = float(len({str(r["date"]) for r in rows}))
    return out


def bootstrap_ci(rows: list[dict[str, Any]], n_boot: int = 1000,
                 seed: int = 0) -> dict[str, Any]:
    estimate = aggregate_rows(rows)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["date"])].append(row)
    days = sorted(groups)
    if not days:
        raise ValueError("no date groups to bootstrap")

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {k: [] for k in REPORT_KEYS}
    for _ in range(int(n_boot)):
        sampled_rows: list[dict[str, Any]] = []
        for day in rng.choice(days, size=len(days), replace=True):
            sampled_rows.extend(groups[str(day)])
        agg = aggregate_rows(sampled_rows)
        for key in REPORT_KEYS:
            samples[key].append(float(agg[key]))

    ci95: dict[str, list[float]] = {}
    for key, vals in samples.items():
        arr = np.asarray([v for v in vals if v == v], dtype=np.float64)
        if arr.size == 0:
            ci95[key] = [float("nan"), float("nan")]
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            ci95[key] = [round(float(lo), 6), round(float(hi), 6)]
    return {"estimate": estimate, "ci95": ci95}


def _fmt_est_ci(report: dict[str, Any], key: str) -> str:
    est = report["estimate"].get(key, float("nan"))
    lo, hi = report["ci95"].get(key, [float("nan"), float("nan")])
    return f"{float(est):.4f} [{float(lo):.4f}, {float(hi):.4f}]"


def format_markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# Pluvian Bootstrap Confidence Intervals",
        "",
        "Bootstrap unit: manifest date. CSI/FSS are recomputed from full-set raw counts/components inside each resample.",
        "",
        "| model | split | windows | days | csi1 | csi10 | csi30 | fss3 | fss11 | crps |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        est = report["estimate"]
        lines.append(
            "| {model} | {split} | {windows:d} | {days:d} | {csi1} | {csi10} | "
            "{csi30} | {fss3} | {fss11} | {crps} |".format(
                model=report["model"], split=report["split"],
                windows=int(est["n_windows"]), days=int(est["n_days"]),
                csi1=_fmt_est_ci(report, "csi_1mm"),
                csi10=_fmt_est_ci(report, "csi_10mm"),
                csi30=_fmt_est_ci(report, "csi_30mm"),
                fss3=_fmt_est_ci(report, "fss_3px"),
                fss11=_fmt_est_ci(report, "fss_11px"),
                crps=_fmt_est_ci(report, "crps"),
            )
        )
    return "\n".join(lines) + "\n"


def _parse_ckpt_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        p = Path(value)
        return p.parent.name, p
    name, path = value.split("=", 1)
    return name, Path(path)


def _evaluate_checkpoint(ckpt: Path, split: str, batch_size: int) -> list[dict[str, Any]]:
    import sys
    import torch
    from torch.utils.data import DataLoader

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from model.losses import WeightedMSE
    from pipeline.data_loader import NPJDataset
    from scripts.eval import _load_ckpt
    from scripts.train import (
        _starts_for_splits, _collate, split_batch, _to_device,
        _csi_counts, _fss_components, _crps_marginal,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = _load_ckpt(ckpt, device)
    manifest = repo_root / cfg["data"]["manifest"]
    starts = _starts_for_splits(manifest, [split])
    load_mfd = bool(cfg["model"].get("mfd_channel_enabled", False))
    load_era5 = bool(cfg["model"].get("era5_enabled", True)) or load_mfd
    ds = NPJDataset(
        manifest=manifest, window_minutes=cfg["data"]["window_minutes"],
        starts=starts, drop_pwv=cfg["data"].get("drop_pwv_val", False),
        load_mfd=load_mfd, load_era5=load_era5,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=max(1, cfg["data"].get("num_workers", 1) // 2),
                        collate_fn=_collate)
    T_in = cfg["model"]["input_frames"]
    T_out = cfg["model"]["forecast_frames"]
    mfd_ch = cfg["model"]["mfd_channels"] if cfg["model"].get("mfd_channel_enabled", False) else 0
    data_loss = WeightedMSE().to(device)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            model_in, rain_tgt, _ = split_batch(batch, T_in, T_out, mfd_ch)
            out = model(model_in)
            rain_pred = out["rain_pred"].float()
            if rain_pred.shape[-2:] != rain_tgt.shape[-2:]:
                Ht, Wt = rain_tgt.shape[-2:]
                rain_pred = rain_pred[..., :Ht, :Wt].contiguous()
            rain_tgt_f = rain_tgt.float()
            times = batch["time"] if isinstance(batch["time"], list) else [str(batch["time"])]
            for b, t_str in enumerate(times):
                pred_b = rain_pred[b:b + 1]
                tgt_b = rain_tgt_f[b:b + 1]
                row: dict[str, Any] = {
                    "time": str(t_str),
                    "date": str(t_str)[:10],
                    "mae": float((pred_b - tgt_b).abs().mean().item()),
                    "crps": float(_crps_marginal(pred_b, tgt_b)),
                    "weighted_mse": float(data_loss(pred_b, tgt_b).item()),
                }
                for thr in THRESHOLDS:
                    h, fa, miss = _csi_counts(pred_b, tgt_b, float(thr))
                    row[f"csi_{thr}mm_hits"] = h
                    row[f"csi_{thr}mm_fa"] = fa
                    row[f"csi_{thr}mm_miss"] = miss
                for nbr in FSS_WINDOWS:
                    num, den = _fss_components(pred_b, tgt_b, 1.0, int(nbr))
                    row[f"fss_{nbr}px_num"] = num
                    row[f"fss_{nbr}px_den"] = den
                rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="MODEL=ckpt/path.pt; can be repeated")
    ap.add_argument("--split", action="append", required=True,
                    help="manifest split; can be repeated")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for ckpt_arg in args.ckpt:
        model_name, ckpt = _parse_ckpt_arg(ckpt_arg)
        for split in args.split:
            print(f"[eval] model={model_name} split={split} ckpt={ckpt}", flush=True)
            rows = _evaluate_checkpoint(ckpt, split, args.batch_size)
            report = bootstrap_ci(rows, n_boot=args.n_boot, seed=args.seed)
            report.update({"model": model_name, "split": split, "ckpt": str(ckpt)})
            reports.append(report)
            stem = f"{model_name}_{split}"
            with open(args.out_dir / f"{stem}_rows.json", "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
            with open(args.out_dir / f"{stem}_ci.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

    md = format_markdown(reports)
    (args.out_dir / "bootstrap_ci.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
