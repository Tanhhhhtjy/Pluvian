"""Generate paper Table 1 (event_test) and Table 2 (test_robust) with bootstrap 95% CIs.

Reads bootstrap CI JSON from ``ckpt/bootstrap_ci/phase7d_skill/`` (ab1, ab2, ab3, xcoreA)
and ``ckpt/bootstrap_ci/ab3b_era5_budget/`` (ab3b). Reports CSI@{1,5,10,30} mm/h,
FSS@{3,11}px, and MAE. CRPS is omitted because for deterministic single-member
predictions ``_crps_marginal`` equals MAE.

Outputs:
  docs/paper/tables_1_2.md   (Markdown, bold = best per column)
  docs/paper/tables_1_2.tex  (LaTeX booktabs, \\textbf = best, \\textit{ \\downarrow}
                              tags the ab3b CSI@30 regression)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# (display_name, ci_file_path_relative_to_repo)
MODEL_SPECS: list[tuple[str, dict[str, str]]] = [
    ("ab1 (radar only)", {
        "event_test": "ckpt/bootstrap_ci/phase7d_skill/ab1_p7d_event_test_ci.json",
        "test_robust": "ckpt/bootstrap_ci/phase7d_skill/ab1_p7d_test_robust_ci.json",
    }),
    ("ab2 (+PWV via xcoreA)", {
        "event_test": "ckpt/bootstrap_ci/phase7d_skill/ab2_p7d_event_test_ci.json",
        "test_robust": "ckpt/bootstrap_ci/phase7d_skill/ab2_p7d_test_robust_ci.json",
    }),
    ("ab3 (+ERA5)", {
        "event_test": "ckpt/bootstrap_ci/phase7d_skill/ab3_era5_event_test_ci.json",
        "test_robust": "ckpt/bootstrap_ci/phase7d_skill/ab3_era5_test_robust_ci.json",
    }),
    ("ab3b (+water-budget loss)", {
        "event_test": "ckpt/bootstrap_ci/ab3b_era5_budget/ab3b_era5_event_test_ci.json",
        "test_robust": "ckpt/bootstrap_ci/ab3b_era5_budget/ab3b_era5_test_robust_ci.json",
    }),
]

METRIC_COLS = [
    ("csi_1mm",  "CSI@1",  "max"),
    ("csi_5mm",  "CSI@5",  "max"),
    ("csi_10mm", "CSI@10", "max"),
    ("csi_30mm", "CSI@30", "max"),
    ("fss_3px",  "FSS@3",  "max"),
    ("fss_11px", "FSS@11", "max"),
    ("mae",      "MAE",    "min"),
]


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_cell(est: float, ci: tuple[float, float] | None) -> str:
    if ci is None:
        return f"{est:.3f}"
    lo, hi = ci
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


def _best_index(values: list[float], direction: str) -> int:
    if direction == "max":
        return max(range(len(values)), key=lambda i: values[i])
    return min(range(len(values)), key=lambda i: values[i])


def build_table_rows(split: str, repo_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for display, paths in MODEL_SPECS:
        report = _load(repo_root / paths[split])
        rows.append({
            "name": display,
            "estimate": report["estimate"],
            "ci95": report.get("ci95", {}),
        })
    return rows


def _bold_md(s: str) -> str:
    return f"**{s}**"


def _bold_tex(s: str) -> str:
    return rf"\textbf{{{s}}}"


def _ab3b_csi30_tag_md(s: str) -> str:
    # Italic + down arrow to flag the reverse-direction regression.
    return f"*{s} (down)*"


def _ab3b_csi30_tag_tex(s: str) -> str:
    return rf"\textit{{{s} $\downarrow$}}"


def render_markdown(rows: list[dict[str, Any]], caption: str, split: str) -> str:
    header = ["Model"] + [c[1] for c in METRIC_COLS]
    sep = ["---"] + [":---:" for _ in METRIC_COLS]
    body: list[list[str]] = []
    col_values = {k: [r["estimate"][k] for r in rows] for k, _, _ in METRIC_COLS}
    best = {k: _best_index(col_values[k], d) for k, _, d in METRIC_COLS}
    ab3b_idx = next(i for i, r in enumerate(rows) if r["name"].startswith("ab3b"))
    for i, r in enumerate(rows):
        cells = [r["name"]]
        for key, _, _ in METRIC_COLS:
            est = r["estimate"][key]
            ci = r["ci95"].get(key)
            ci_tuple = (ci[0], ci[1]) if ci else None
            cell = _fmt_cell(est, ci_tuple)
            if i == best[key]:
                cell = _bold_md(cell)
            if i == ab3b_idx and key == "csi_30mm":
                cell = _ab3b_csi30_tag_md(cell)
            cells.append(cell)
        body.append(cells)

    lines = [
        f"### {caption}",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    n_windows = int(rows[0]["estimate"]["n_windows"])
    n_days = int(rows[0]["estimate"]["n_days"])
    lines.append(
        f"_Split: {split} ({n_days} days, {n_windows} forecast windows). "
        f"CSI thresholds in mm h$^{{-1}}$; FSS neighbourhood in grid pixels "
        f"(1 pixel = 0.01$^\\circ$ $\\approx$ 1.1 km). MAE in mm h$^{{-1}}$. "
        f"Brackets show day-stratified bootstrap 95% CI (n_boot = 1000). "
        f"Bold marks the best value per column; ab3b CSI@30 is italicised with "
        f"a down arrow to flag the inverse direction of its regression._"
    )
    return "\n".join(lines) + "\n"


def render_latex(rows: list[dict[str, Any]], caption: str, label: str, split: str) -> str:
    col_values = {k: [r["estimate"][k] for r in rows] for k, _, _ in METRIC_COLS}
    best = {k: _best_index(col_values[k], d) for k, _, d in METRIC_COLS}
    ab3b_idx = next(i for i, r in enumerate(rows) if r["name"].startswith("ab3b"))

    header_cells = ["Model"] + [c[1] for c in METRIC_COLS]
    align = "l" + "c" * len(METRIC_COLS)

    n_windows = int(rows[0]["estimate"]["n_windows"])
    n_days = int(rows[0]["estimate"]["n_days"])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(header_cells) + r" \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        cells = [r["name"].replace("_", r"\_")]
        for key, _, _ in METRIC_COLS:
            est = r["estimate"][key]
            ci = r["ci95"].get(key)
            ci_tuple = (ci[0], ci[1]) if ci else None
            cell = _fmt_cell(est, ci_tuple)
            if i == best[key]:
                cell = _bold_tex(cell)
            if i == ab3b_idx and key == "csi_30mm":
                cell = _ab3b_csi30_tag_tex(cell)
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        rf"\par\medskip",
        rf"\footnotesize Split: \texttt{{{split}}} ({n_days} days, {n_windows} windows). "
        r"CSI thresholds in mm\,h$^{-1}$; FSS neighbourhood in grid pixels "
        r"(1\,pixel $=$ 0.01$^\circ$ $\approx$ 1.1\,km). MAE in mm\,h$^{-1}$. "
        r"Brackets show day-stratified bootstrap 95\% CI ($n_{\text{boot}} = 1000$). "
        r"\textbf{Bold} marks the best value per column; "
        r"\textit{italic} with $\downarrow$ flags the ab3b CSI@30 regression "
        r"(lower is the wrong direction for skill).",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--out_md", type=Path, default=Path("docs/paper/tables_1_2.md"))
    ap.add_argument("--out_tex", type=Path, default=Path("docs/paper/tables_1_2.tex"))
    args = ap.parse_args()

    rows_t1 = build_table_rows("event_test", args.repo_root)
    rows_t2 = build_table_rows("test_robust", args.repo_root)

    md_parts = [
        "# Pluvian Paper Tables 1 and 2",
        "",
        "Headline deterministic skill scores for the four model variants on the "
        "two evaluation splits. CRPS is omitted because the deterministic "
        "single-member CRPS reduces to MAE (see `_crps_marginal` in "
        "`scripts/train.py`).",
        "",
        render_markdown(
            rows_t1,
            "Table 1. Deterministic skill on event_test (4 storm days)",
            "event_test",
        ),
        render_markdown(
            rows_t2,
            "Table 2. Deterministic skill on test_robust (non-storm days)",
            "test_robust",
        ),
    ]
    tex_parts = [
        "% Pluvian paper Tables 1 and 2. Requires booktabs.",
        "",
        render_latex(
            rows_t1,
            "Deterministic skill on \\texttt{event\\_test} (4 storm days, 96 windows). "
            "CSI thresholds in mm\\,h$^{-1}$, FSS in grid pixels.",
            "tab:event_test",
            "event_test",
        ),
        "",
        render_latex(
            rows_t2,
            "Deterministic skill on \\texttt{test\\_robust} (8 non-storm days, 192 windows). "
            "CSI thresholds in mm\\,h$^{-1}$, FSS in grid pixels.",
            "tab:test_robust",
            "test_robust",
        ),
    ]

    out_md = args.repo_root / args.out_md if not args.out_md.is_absolute() else args.out_md
    out_tex = args.repo_root / args.out_tex if not args.out_tex.is_absolute() else args.out_tex
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_parts), encoding="utf-8")
    out_tex.write_text("\n".join(tex_parts), encoding="utf-8")
    print(f"[wrote] {out_md}")
    print(f"[wrote] {out_tex}")


if __name__ == "__main__":
    main()
