# Pluvian Figure Inventory

Updated: 2026-06-02 UTC.

This file tracks paper-ready visual material and the next figures to generate after ab3/Track C results arrive.

## Existing Material

| figure type | current files | status |
|---|---|---|
| ab1 vs ab2 event predictions | `ckpt/figures/event_test_p7c/*.png`, `ckpt/figures/p7d/*.png`, `figures/event_23_7_ab1_vs_ab2.*` | usable for baseline narrative |
| ab2 csi30 caveat case | `ckpt/figures/p7d/cmp_failure_2023-07-29T06.png` | usable failure-case evidence |
| +PWV mechanism demos | `ckpt/figures/mechanism/*.png`, `figures/case_23p7/*.png` | usable for physical motivation |
| per-lead skill audit | `ckpt/figures/per_lead/per_lead_event_test.png`, `ckpt/figures/per_lead/per_lead_test_robust.png` | central evidence for localized csi30 regression |
| xcore per-lead audit | `ckpt/figures/xcore_per_lead/xcoreA/*.png`, `ckpt/figures/xcore_per_lead/xcoreB/*.png` | usable for loss-reweighting tradeoff / appendix |
| bootstrap CI tables | `docs/experiments/pluvian_bootstrap_ci.md`, `docs/experiments/pluvian_bootstrap_delta_ci.md`, `ckpt/bootstrap_ci/phase7d_skill/*.json` | usable uncertainty evidence for metrics and paired deltas |
| paper table snippets | `docs/experiments/pluvian_paper_tables.md`, `docs/experiments/pluvian_paper_tables.tex` | ready-to-copy aggregate and paired-delta tables |
| paired delta CI forest plots | `ckpt/figures/bootstrap_delta_ci/paired_delta_ci_*.png`, `ckpt/figures/bootstrap_delta_ci/paired_delta_ci_*.pdf` | visual uncertainty/tradeoff summary for ab2/xcore |
| sharpness / spectrum audit | `ckpt/figures/sharp_all/sharpness_all.png`, `ckpt/figures/sharp_all/sharpness_all.json` | central evidence for separating realism from skill |
| GAN / diffusion examples | `ckpt/figures/gan*`, `ckpt/figures/gen_eval`, `ckpt/figures/hicore_eval`, `ckpt/figures/diff*` | appendix or negative-result material; do not use as skill evidence |
| case-study narrative | `figures/case_23p7/SUMMARY.md`, `figures/case_23p7/narrative.md` | useful text scaffold |

## Missing After ab3 Finishes

| priority | figure | purpose | likely inputs |
|---:|---|---|---|
| P0 | ab3 summary table | decide if ERA5 changes route | `ckpt/eval/ablation_3_era5_p7c/{event_test,test_robust}/metric.json` |
| P0 | ab3 per-lead csi curves | diagnose event csi30 +18..+42 min | `ckpt/figures/ab3_per_lead/per_lead_*.json` |
| P0 | ab3 bootstrap CI table | uncertainty around ab3 deltas | `scripts/eval_bootstrap_ci.py` after ab3 eval |
| P1 | ab1/ab2/ab3 event case grids | show what ERA5 changes spatially | ab1, ab2, ab3 checkpoints; starts from existing event figures |
| P1 | ab3 error maps | separate displacement error from intensity error | same case grids |
| P1 | CRPS/spread comparison | support calibration/probabilistic claims | eval metric JSON plus MC-dropout samples if saved |
| P1 | drop-PWV diagnostic table | show PWV influence on trained ab2 strong-core output | `ckpt/eval/ablation_2_p7d_droppwv/*/metric.json` |
| P2 | Track C budget residual map | physical-consistency claim | budget-enabled checkpoint plus ERA5 u/v/q |
| P2 | reliability/rank histogram | only for Track B ensemble | generated ensemble samples |

## Reusable Scripts

| script | use |
|---|---|
| `scripts/eval.py` | aggregate CSI/FSS/CRPS on `event_test` and `test_robust` |
| `scripts/eval_per_lead.py` | per-lead CSI/FSS comparison against a baseline checkpoint |
| `scripts/plot_ab1_vs_ab2.py` | 4-row GT/input/ab1/ab2 prediction grid; adapt for ab3 comparison if needed |
| `scripts/plot_prediction.py` | single-checkpoint prediction plot |
| `scripts/plot_pwv_mechanism_demo.py` | PWV gate/mechanism visualization |
| `scripts/eval_sharpness_all.py` | honest high-k and p99.9 sharpness audit |
| `scripts/plot_sharp_compare.py` | visual and RAPSD comparison for sharpening routes |

## Recommended Case Starts

Reuse existing starts first so figures are comparable across model versions:

- `2023-07-31T06:00:00` for PWV mechanism / extreme event.
- `2023-08-01T08:00:00` for ab2-win mechanism.
- `2023-08-08T06:00:00` for ab2-win mechanism.
- `2023-05-19T10:00:00` for sharpness / spectrum examples.

## Writing Guardrails

- Skill figures: CSI/FSS/CRPS only.
- Sharpness figures: high-k, RAPSD, p99/p99.9, and overrun diagnostics only.
- Do not describe GAN or diffusion outputs as accuracy gains unless the corresponding skill metrics support that statement.
- Include at least one failure case; this is important for credibility and for explaining the csi30 short-lead issue.
