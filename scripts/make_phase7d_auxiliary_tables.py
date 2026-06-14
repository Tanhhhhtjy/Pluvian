"""Generate auxiliary Phase 7d paper tables for attribution and sharpness lanes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SPLITS = ("event_test", "test_robust")
METRICS = ("csi_1mm", "csi_10mm", "csi_30mm", "fss_3px", "crps")
SHARPNESS_METHODS = (
    ("baseline", "deterministic baseline", "reference only"),
    ("sharpB", "sharpB loss-FT", "texture diagnostic"),
    ("gan_coreA", "GAN-coreA", "realism/sharpness only"),
    ("hicore", "hi-core", "realism/sharpness only"),
    ("diff", "diffusion residual", "invalid without amplitude control"),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{float(value):.3f}"


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{float(value):+.3f}"


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def _tex_row(cells: list[str]) -> str:
    return " & ".join(cells) + r" \\" 


def _load_metric_reports(rows_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(rows_dir.glob("*_ci.json"))]


def _load_drop_metrics(eval_root: Path) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        path = eval_root / "ablation_2_p7d_droppwv" / split / "metric.json"
        if path.exists():
            rows[split] = _load_json(path)
    return rows


def build_drop_pwv_rows(full_reports: list[dict[str, Any]],
                        drop_metrics: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    by_split = {
        report["split"]: report["estimate"]
        for report in full_reports
        if report.get("model") == "ab2_p7d"
    }
    rows = []
    for split in SPLITS:
        full = by_split.get(split)
        drop = drop_metrics.get(split)
        if full is None or drop is None:
            continue
        full_metrics = {metric: round(float(full[metric]), 4) for metric in METRICS}
        drop_values = {metric: round(float(drop[metric]), 4) for metric in METRICS}
        delta = {metric: round(drop_values[metric] - full_metrics[metric], 4) for metric in METRICS}
        rows.append({"split": split, "full": full_metrics, "drop": drop_values, "delta": delta})
    return rows


def format_drop_pwv_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Drop-PWV Attribution Diagnostic",
        "",
        "This is a trained-model channel-influence diagnostic, not a radar-only baseline. "
        "Deltas are drop-PWV minus normal ab2; positive CSI/FSS and negative CRPS are favorable.",
        "",
        "| split | mode | CSI1 | CSI10 | CSI30 | FSS3 | CRPS |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        for mode, label, formatter in [
            ("full", "ab2 normal", _fmt),
            ("drop", "ab2 with PWV dropped", _fmt),
            ("delta", "delta", _fmt_delta),
        ]:
            metrics = row[mode]
            lines.append(
                "| {split} | {label} | {csi1} | {csi10} | {csi30} | {fss3} | {crps} |".format(
                    split=row["split"], label=label,
                    csi1=formatter(metrics["csi_1mm"]),
                    csi10=formatter(metrics["csi_10mm"]),
                    csi30=formatter(metrics["csi_30mm"]),
                    fss3=formatter(metrics["fss_3px"]),
                    crps=formatter(metrics["crps"]),
                )
            )
    return "\n".join(lines) + "\n"


def format_drop_pwv_booktabs(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        _tex_row(["Split", "Mode", "CSI1", "CSI10", "CSI30", "FSS3", "CRPS"]),
        r"\midrule",
    ]
    for row in rows:
        for mode, label, formatter in [
            ("full", "ab2 normal", _fmt),
            ("drop", "ab2 PWV dropped", _fmt),
            ("delta", r"$\Delta$", _fmt_delta),
        ]:
            metrics = row[mode]
            lines.append(_tex_row([
                _tex_escape(row["split"]), label,
                formatter(metrics["csi_1mm"]), formatter(metrics["csi_10mm"]),
                formatter(metrics["csi_30mm"]), formatter(metrics["fss_3px"]),
                formatter(metrics["crps"]),
            ]))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_sharpness_rows(sharpness: dict[str, Any]) -> list[dict[str, Any]]:
    agg = sharpness["agg"]
    gt_p999 = float(agg["gt_p999"])
    rows = []
    for key, label, claim in SHARPNESS_METHODS:
        if key not in agg:
            continue
        item = agg[key]
        rows.append({
            "lane": "realism",
            "method": label,
            "high_k": float(item["hk"]),
            "p999": float(item["p999"]),
            "gt_p999": gt_p999,
            "frac_over65": float(agg["diff_frac_over65"]) if key == "diff" else None,
            "claim": claim,
        })
    return rows


def format_sharpness_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "## Skill-vs-Sharpness Separation",
        "",
        "CSI/FSS/CRPS skill is not inferred from realism diagnostics. "
        "High-k target is 1.0; p99.9 is compared with the GT p99.9 for the same extreme-event subset.",
        "",
        "| lane | method | high-k | p99.9 dBZ | GT p99.9 dBZ | >65 dBZ frac | allowable claim |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {lane} | {method} | {hk} | {p999} | {gt_p999} | {over} | {claim} |".format(
                lane=row["lane"], method=row["method"],
                hk=_fmt(row["high_k"]), p999=_fmt(row["p999"]),
                gt_p999=_fmt(row["gt_p999"]), over=_fmt(row["frac_over65"]),
                claim=row["claim"],
            )
        )
    return "\n".join(lines) + "\n"


def format_sharpness_booktabs(rows: list[dict[str, Any]]) -> str:
    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        _tex_row(["Lane", "Method", "High-k", "p99.9", "GT p99.9", r"$>65$ frac", "Claim"]),
        r"\midrule",
    ]
    for row in rows:
        lines.append(_tex_row([
            row["lane"], _tex_escape(row["method"]), _fmt(row["high_k"]),
            _fmt(row["p999"]), _fmt(row["gt_p999"]), _fmt(row["frac_over65"]),
            _tex_escape(row["claim"]),
        ]))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_outputs(rows_dir: Path, eval_root: Path, sharpness_json: Path) -> tuple[str, str]:
    metric_reports = _load_metric_reports(rows_dir)
    drop_rows = build_drop_pwv_rows(metric_reports, _load_drop_metrics(eval_root))
    sharp_rows = build_sharpness_rows(_load_json(sharpness_json))
    md = "\n".join([
        "# Pluvian Phase 7d Auxiliary Tables",
        "",
        "Generated from existing holdout and sharpness diagnostics. These tables are intentionally separated from the main deterministic skill table.",
        "",
        format_drop_pwv_markdown(drop_rows),
        format_sharpness_markdown(sharp_rows),
    ])
    tex = "\n".join([
        "% Pluvian Phase 7d auxiliary tables. Requires booktabs.",
        "",
        "% Drop-PWV attribution diagnostic",
        format_drop_pwv_booktabs(drop_rows),
        "% Skill-vs-sharpness separation",
        format_sharpness_booktabs(sharp_rows),
    ])
    return md, tex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_dir", type=Path, default=Path("ckpt/bootstrap_ci/phase7d_skill"))
    ap.add_argument("--eval_root", type=Path, default=Path("ckpt/eval"))
    ap.add_argument("--sharpness_json", type=Path, default=Path("ckpt/figures/sharp_all/sharpness_all.json"))
    ap.add_argument("--out_md", type=Path, default=Path("docs/experiments/pluvian_auxiliary_tables.md"))
    ap.add_argument("--out_tex", type=Path, default=Path("docs/experiments/pluvian_auxiliary_tables.tex"))
    args = ap.parse_args()
    md, tex = build_outputs(args.rows_dir, args.eval_root, args.sharpness_json)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_tex.write_text(tex, encoding="utf-8")
    print(f"[wrote] {args.out_md}")
    print(f"[wrote] {args.out_tex}")


if __name__ == "__main__":
    main()
