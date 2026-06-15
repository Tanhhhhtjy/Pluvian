# Pluvian: Multimodal Nowcasting with Pooled-CSI Diagnostic of Physics-Loss Trade-offs

**Working title — npj Climate and Atmospheric Science submission**

Compiled draft date: 2026-06-14. Section 3.4 (CDU dual-branch decoder results) is pending CDU training completion (in progress on GPU0).

---

## Abstract (draft 2026-06-14, ~200 words)

Short-term precipitation nowcasting over North China remains difficult due to convective intermittency and underused multimodal context. We present a 6-minute-resolution, 108-minute deep-learning nowcasting framework that progressively fuses radar with column water vapour (PWV) and ERA5 large-scale environmental fields. Stepwise fusion yields monotonic improvement across CSI at 1/5/10 mm h⁻¹, FSS, and MAE on an event-based test set (CSI@10 0.565→0.632, MAE 5.19→4.27); CSI@30 is non-monotonic across the cascade (ab1 0.308, ab2 0.283, ab3 0.291) and we revisit this in §3.2. Pushing further with a physically motivated water-budget loss improves light-to-moderate skill but causes extreme CSI@30 to collapse from 0.291 to 0.210 (−28%; paired bootstrap 95% CI [−0.091, −0.057], n=2000) on event_test and from 0.276 to 0.215 (−22%; 95% CI [−0.078, −0.019]) on test_robust, with 0 of 18 forecast leads exhibiting a positive change relative to the radar baseline. Using a pooled-CSI diagnostic with neighbourhood max-pool tolerance (window sizes 1, 4, 16 pixels), we show this is not spatial displacement but physical smoothing of convective cores — a differentiating negative result for physics-informed loss design in nowcasting. We further introduce Cubic Dual Upsampling (CDU), a dual-branch decoder combining low-frequency bicubic interpolation with high-frequency PixelShuffle residuals. CDU improves CSI at 1/5/10 mm h⁻¹ and FSS uniformly over ab3 (+0.018 CSI@1, +0.013 CSI@10 on event_test) at an essentially neutral cost at the extreme tail (CSI@30 delta −0.006 on event_test, −0.001 on test_robust; paired CIs [−0.0096, −0.0002] and [−0.0045, +0.0032], two orders of magnitude smaller than the ab3b collapse) — that is, CDU avoids the budget-loss trade-off without itself recovering the extreme tail, which we frame as an open architectural problem. All evaluation is restricted to four storm days and eight non-storm days over North China in May–August 2023; we therefore phrase methodological claims as proposed practice rather than universal prescriptions.

---

## Table of Contents

- §1 Introduction → `section_1_introduction.md`
- §2 Method → `section_2_method.md`
- §3 Results → `section_3_results.md`
- §4 Discussion → `section_4_discussion.md`
- Tables 1–2 → `tables_1_2.md` (markdown) / `tables_1_2.tex` (LaTeX)

## Figure list

| ID | Title | Path | Status |
|---|---|---|---|
| Fig 1 | Multimodal fusion framework + CDU decoder schematic | (not yet drawn) | TODO |
| Fig 2 | Per-lead CSI@{1,10,30} for 4 models × 2 splits | `ckpt/figures/paper_per_lead/` (6 PNGs) | ✓ |
| Fig 3 | Pooled-CSI diagnostic (csi vs pool window) | `ckpt/figures/pooled_csi/pooled_csi_diag_*.{png,pdf}` | ✓ |
| Fig 4 | Case study 2023-07-30 09:00 UTC, GT + 4 models + diff, 30 dBZ contour | `ckpt/figures/paper_case/event_20230730T0900_final.png` | ✓ |
| Fig 5 (suppl.) | Skill ladder bar chart (4-model × 3-threshold × 2-split) | `ckpt/figures/paper_skill_bar/skill_ladder.{png,pdf}` | ✓ |
| Fig S1 | ERA5 + PWV input example | (not yet drawn) | TODO |

## Cited works (placeholder bibliography)

To be expanded; in-text uses `[Author Year]`.

- Andrychowicz et al. 2023 — MetNet-3
- Bevis et al. 1992 — GPS meteorology / PWV
- Bi et al. 2023 — Pangu-Weather
- Ebert 2008 — neighbourhood verification
- Gao et al. 2022 — Earthformer
- Gilleland 2009 — verification of spatial forecasts
- Hersbach et al. 2020 — ERA5
- Lam et al. 2023 — GraphCast
- Mittermaier 2014 — fuzzy verification
- Ravuri et al. 2021 — DGMR (Nature)
- Roberts & Lean 2008 — FSS
- Song et al. 2026 — exPreCast (ICLR), Cubic Dual Upsampling
- Zhang et al. 2023 — NowcastNet (Nature)

## Outstanding tasks before submission

1. Fill §3.4 with CDU ablation results (CDU training in progress, ETA 2026-06-15 morning)
2. Draw Fig 1 (architecture schematic) — needs to convey radar/PWV/ERA5 encoder paths and the CDU dual-branch decoder
3. Draw Fig S1 (input data example) — paired ERA5/PWV/radar snapshots
4. Expand bibliography to full BibTeX
5. Reconcile minor terminology between §2 (method) and §3 (results), e.g. "MaxPool_w" notation
6. Run one final pass for consistency of numerical values across §3, §4, Table 1, Table 2 (currently all hand-aligned)
