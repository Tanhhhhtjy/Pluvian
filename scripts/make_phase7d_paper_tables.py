"""Generate paper-ready Phase 7d metric tables from CI JSON files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_ORDER = ("ab1_p7d", "ab2_p7d", "ab3_era5", "xcoreA", "xcoreB")


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _tex_row(cells: list[str]) -> str:
    return " & ".join(cells) + r" \\" 


def _fmt_point(value: float) -> str:
    return f"{float(value):.3f}"


def _fmt_delta(value: float) -> str:
    return f"{float(value):+.3f}"


def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{float(lo):.3f}, {float(hi):.3f}]"


def _filter_reports(reports: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    by_model = {r["model"]: r for r in reports if r.get("split") == split}
    return [by_model[m] for m in MODEL_ORDER if m in by_model]


def format_metric_markdown(reports: list[dict[str, Any]], split: str) -> str:
    rows = _filter_reports(reports, split)
    lines = [
        f"### Aggregate Skill ({split})",
        "",
        "| model | CSI1 | CSI10 | CSI30 | FSS3 | CRPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in rows:
        est = report["estimate"]
        lines.append(
            "| {model} | {csi1} | {csi10} | {csi30} | {fss3} | {crps} |".format(
                model=report["model"],
                csi1=_fmt_point(est["csi_1mm"]),
                csi10=_fmt_point(est["csi_10mm"]),
                csi30=_fmt_point(est["csi_30mm"]),
                fss3=_fmt_point(est["fss_3px"]),
                crps=_fmt_point(est["crps"]),
            )
        )
    return "\n".join(lines) + "\n"


def format_metric_booktabs(reports: list[dict[str, Any]], split: str) -> str:
    rows = _filter_reports(reports, split)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        _tex_row(["Model", "CSI1", "CSI10", "CSI30", "FSS3", "CRPS"]),
        r"\midrule",
    ]
    for report in rows:
        est = report["estimate"]
        lines.append(_tex_row([
            _latex_escape(report["model"]),
            _fmt_point(est["csi_1mm"]),
            _fmt_point(est["csi_10mm"]),
            _fmt_point(est["csi_30mm"]),
            _fmt_point(est["fss_3px"]),
            _fmt_point(est["crps"]),
        ]))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _filter_delta_reports(reports: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    preferred = (
        "ab2_p7d - ab1_p7d",
        "ab3_era5 - ab1_p7d",
        "ab3_era5 - ab2_p7d",
        "xcoreA - ab1_p7d",
        "xcoreB - ab1_p7d",
        "xcoreA - ab2_p7d",
        "xcoreB - ab2_p7d",
    )
    by_cmp = {r["comparison"]: r for r in reports if r.get("split") == split}
    return [by_cmp[c] for c in preferred if c in by_cmp]


def _fmt_delta_ci(report: dict[str, Any], key: str) -> str:
    lo, hi = report["ci95"][key]
    return f"${_fmt_delta(report['estimate'][key])}$ {_fmt_ci(lo, hi)}"


def _fmt_delta_ci_plain(report: dict[str, Any], key: str) -> str:
    lo, hi = report["ci95"][key]
    return f"{_fmt_delta(report['estimate'][key])} {_fmt_ci(lo, hi)}"


def format_delta_markdown(reports: list[dict[str, Any]], split: str) -> str:
    rows = _filter_delta_reports(reports, split)
    lines = [
        f"### Paired Delta CI ({split})",
        "",
        "Deltas are candidate minus baseline. Positive CSI/FSS and negative CRPS are favorable.",
        "",
        "| comparison | dCSI1 | dCSI10 | dCSI30 | dFSS3 | dCRPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for report in rows:
        lines.append(
            "| {cmp} | {csi1} | {csi10} | {csi30} | {fss3} | {crps} |".format(
                cmp=report["comparison"],
                csi1=_fmt_delta_ci_plain(report, "csi_1mm"),
                csi10=_fmt_delta_ci_plain(report, "csi_10mm"),
                csi30=_fmt_delta_ci_plain(report, "csi_30mm"),
                fss3=_fmt_delta_ci_plain(report, "fss_3px"),
                crps=_fmt_delta_ci_plain(report, "crps"),
            )
        )
    return "\n".join(lines) + "\n"


def format_delta_booktabs(reports: list[dict[str, Any]], split: str) -> str:
    rows = _filter_delta_reports(reports, split)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        _tex_row([
            "Comparison", r"$\Delta$CSI1", r"$\Delta$CSI10",
            r"$\Delta$CSI30", r"$\Delta$FSS3", r"$\Delta$CRPS",
        ]),
        r"\midrule",
    ]
    for report in rows:
        lines.append(_tex_row([
            _latex_escape(report["comparison"]).replace(" - ", " $-$ "),
            _fmt_delta_ci(report, "csi_1mm"),
            _fmt_delta_ci(report, "csi_10mm"),
            _fmt_delta_ci(report, "csi_30mm"),
            _fmt_delta_ci(report, "fss_3px"),
            _fmt_delta_ci(report, "crps"),
        ]))
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _load_metric_reports(rows_dir: Path) -> list[dict[str, Any]]:
    reports = []
    for path in sorted(rows_dir.glob("*_ci.json")):
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def _load_delta_reports(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_outputs(rows_dir: Path, delta_json: Path) -> tuple[str, str]:
    metric_reports = _load_metric_reports(rows_dir)
    delta_reports = _load_delta_reports(delta_json)
    md_parts = [
        "# Pluvian Phase 7d Paper Tables",
        "",
        "Generated from bootstrap CI JSON. Point estimates are full-set metrics; delta tables show candidate minus baseline with 95% paired day-bootstrap CIs.",
        "",
    ]
    tex_parts = [
        "% Pluvian Phase 7d paper tables. Requires booktabs.",
        "",
    ]
    for split in ("event_test", "test_robust"):
        md_parts.append(format_metric_markdown(metric_reports, split))
        md_parts.append(format_delta_markdown(delta_reports, split))
        tex_parts.append(f"% Aggregate skill: {split}\n" + format_metric_booktabs(metric_reports, split))
        tex_parts.append(f"% Paired deltas: {split}\n" + format_delta_booktabs(delta_reports, split))
    return "\n".join(md_parts), "\n".join(tex_parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows_dir", type=Path, default=Path("ckpt/bootstrap_ci/phase7d_skill"))
    ap.add_argument("--delta_json", type=Path, default=Path("docs/experiments/pluvian_bootstrap_delta_ci.json"))
    ap.add_argument("--out_md", type=Path, default=Path("docs/experiments/pluvian_paper_tables.md"))
    ap.add_argument("--out_tex", type=Path, default=Path("docs/experiments/pluvian_paper_tables.tex"))
    args = ap.parse_args()
    md, tex = build_outputs(args.rows_dir, args.delta_json)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    args.out_tex.write_text(tex, encoding="utf-8")
    print(f"[wrote] {args.out_md}")
    print(f"[wrote] {args.out_tex}")


if __name__ == "__main__":
    main()
