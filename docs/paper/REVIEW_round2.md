# Reviewer #2 Report (Round 2) — npj Climate and Atmospheric Science

Manuscript: *Pluvian: Multimodal Nowcasting with Pooled-CSI Diagnostic of Physics-Loss Trade-offs* (v2, 2026-06-14)

## 1. Summary & Recommendation

This is an independent round-2 review of the revised manuscript. The authors have done substantive work between v1 and v2: §3.4 is now populated with real CDU numbers, a budget-loss weight sweep (Table A1 in §3.2) has been added, the bibliography/CRPS housekeeping is tidied, and the narrative around contribution (iii) has been honestly downgraded from "restores extreme skill" to "avoids the budget-loss trade-off without recovering the tail". The core diagnostic story (pooled-CSI shows ab3b smoothing) is now well-supported and is the strongest part of the paper.

However, two new structural problems surface precisely *because* of the v2 changes. First, the weight-sweep evidence the authors offer in support of §4.2 contains a finding (weight = 0.001 dominates the main ab3b on bulk metrics *and* has only a 5% CSI@30 regression) that, taken at face value, undermines the "structural property of the loss landscape" thesis they now build on it. Second, the v2 reframing leaves the paper as a diagnostic + negative-result manuscript whose flagship architectural intervention is explicitly disavowed in §1, §4.4 and the Abstract — which is intellectually honest but raises a real question about whether contribution (iii) clears the standalone-novelty bar for npj. Two round-1 concerns (C3 external baseline, C6 from-scratch CDU control) remain entirely unaddressed.

**Recommendation: Major Revision.** The paper is closer to publishable than v1; the remaining issues are about framing/scope and one substantive cherry-picking risk, not about missing experiments at the same scale as round 1.

## 2. Round 1 closure

| ID | Status | Evidence |
|---|---|---|
| C1 (CI overlap, weak sig.) | Partially | §3.2 now reports paired window-level CI excluding zero on both splits (n=2000), but day-stratified bootstrap with n_day = 4 / 8 has a combinatorial information ceiling that n_boot cannot lift. |
| C2 (§3.4 empty) | Closed | §3.4 now populated; framing recast to "neutral, not recovery". |
| C3 (no external baseline) | Open | DGMR / NowcastNet / Earthformer still cited but never benchmarked. No anchoring of absolute skill. |
| C4 (pooled-CSI novelty) | Closed | §3.3 last paragraph and §4.3 now reframe as "diagnostic use" of fuzzy verification with Ebert / Gilleland / Mittermaier cited. |
| C5 (failure attribution) | Closed (with caveat — see new concern N1) | Weight sweep added (§3.2 Table A1); attribution shifted from "single-point" to "structural over weight regime". |
| C6 (warm-start confound) | Open | Still only warm-started ab3-cdu, no from-scratch control. §2.5 admits this. |
| C7 (over-general claim) | Partially | Abstract softened ("proposed practice rather than universal prescriptions"); §1 (iii) still reaches discipline-wide ("open architectural problem"). |
| C8 (CRPS) | Closed | Dropped from Tables 1–2 with note in caption. |
| M1 (HTML comments) | Closed | I did not see leftover editorial HTML comments in v2. |
| M2 (Abstract monotonicity) | Closed | Abstract now says "CSI@30 is non-monotonic across the cascade". |
| M3 (unit consistency) | Open | §3.3 still references "30 dBZ contour" while CSI thresholds are in mm h⁻¹; Fig 4 caption inherits the mismatch. |
| M4 (PWV stations) | Open | "Seven near-empty stations dropped" still without IDs. |
| M5 (Beucler / de Bezenac bib) | Open | Still cited in §1, still absent from MAIN.md bibliography list. |

## 3. New concerns

**N1 (substantive — cherry-picking risk in the weight sweep, §3.2 Table A1).** The sweep is intended to support the "structural property of the loss landscape" claim now central to §4.2. But the row at weight = 0.001 reports event_test CSI@1 = 0.507, CSI@10 = 0.650, MAE = 3.97 — *better than the main ab3b at weight = 0.1* (0.492 / 0.641 / 4.10) on every bulk metric — with only a −5% CSI@30 regression vs −28% for the main ab3b. The authors' defence (loss ≈ 1 × 10⁻⁵ vs data-loss ~ 10, "physics term has essentially stopped influencing optimisation") is a post-hoc rationalisation: if a 10⁻⁵-relative-magnitude term meaningfully shifts CSI@1 by +0.015 and MAE by −3%, it is by definition still doing something. A hostile reviewer will ask: why is weight = 0.1 the "main" ab3b configuration when 0.001 looks Pareto-superior, and why isn't the §4.2 "structural" thesis simply "we mis-tuned to 0.1"? **Required:** either (a) demote the main ab3b to weight = 0.001 and restructure §3.2 / §4.2 around that, or (b) provide a quantitative argument — gradient norms, per-pixel contribution to the update, or an ablation that rules out coincidental data-loss optimisation — that the 0.001 bulk gains are *not* attributable to the budget term.

**N2 (§3.4 vs Fig 4 internal inconsistency).** §3.4 reports aggregate CSI@30 deltas of −0.006 / −0.001 and concludes "essentially neutral at the extreme tail". The same paragraph then candidly notes that in the 2023-07-30 09:00 UTC case window, "CDU's reflectivity peaks at 26 dBZ against an observed 46 dBZ, smoothing the strongest core in that single window even though the aggregate is preserved." These two readings can both be true only if other windows compensate with positive deltas — but the manuscript does not report the within-day or per-window variance of the CDU–ab3 difference. Without that, the reader cannot distinguish "CDU is uniformly slightly negative on cores" from "CDU helps some windows and hurts others, averaging to ~0". The latter is a meaningfully different behavioural story and matters for any practitioner deciding whether to deploy CDU. **Required:** a small histogram or per-window scatter (CDU − ab3) for CSI@30 over the 96 event_test windows, and either a softer Fig 4 caption acknowledging this is not a representative window, or an additional contrasting case where CDU sharpens.

**N3 (standalone novelty of contribution (iii)).** V2 honestly recasts CDU as "trade-off avoidance" rather than "tail recovery". But once the architectural intervention is admitted not to solve the headline problem, the contribution reduces to "a decoder variant gives +0.018 CSI@1 over ab3 while keeping CSI@30 within CI". For npj CAS, which favours either a clear methodological advance with broad transferability or a substantive empirical finding, a single-decoder-block ablation that yields sub-2% bulk gains and zero tail gains is borderline. The strongest contributions in v2 are now (a) the diagnostic (pooled-CSI applied to physics-loss failures) and (b) the negative result (budget loss collapses cores). **Suggestion:** either reframe the paper as a *diagnostic / negative-result methods paper* and demote CDU to a confirming ablation in §3.4, or split the architectural piece (CDU) into a separate, more focused application-track submission (MWR / GMD). The current §1 still leads with "we present a … nowcasting framework", positioning the paper as a model paper, but the actual headline finding is methodological. This framing mismatch will draw editor scepticism before reviewer scepticism.

**N4 (information ceiling on day-stratified bootstrap, deepening C1).** Round 1's C1 was about overlap of marginal CIs. V2 fixes the *paired* CI issue (paired CI now excludes zero for ab3 vs ab3b on both splits). What v2 does *not* fix is the underlying information limit: with n_day = 4 on event_test, day-stratified resampling has 4⁴ = 256 distinct draws, and CSI is a non-linear pooled statistic, so the effective resolution of the bootstrap is coarser than n_boot = 2000 suggests. The very tight CI on the CDU–ab3 difference on event_test ([−0.0096, −0.0002]) is particularly suspicious — the lower bound is within 2 × 10⁻⁴ of zero. **Required:** report a sensitivity analysis where (a) the bootstrap is window-stratified (n = 96 / 192) rather than day-stratified as a comparison, and (b) explicitly state in §2.6 / §3 which inferential statement each CI is supposed to support (variance across days vs. variance across windows are different objects).

**N5 (§3.1 ab3 vs ab1 on test_robust CSI@30 is not actually a fusion win).** Table 2 reports ab1 CSI@30 = 0.270 [0.216, 0.297] vs ab3 = 0.276 [0.236, 0.292] — a +0.006 delta with heavily overlapping marginal CIs and within the day-bootstrap noise floor implied by N4. §3.1 nonetheless says "the test_robust split reproduces the ordering" and reports the fusion cascade as monotone-by-modality-except-CSI@30, which is true on event_test but understated on test_robust. **Required:** report the paired ab3 vs ab1 CSI@30 CI on test_robust (not just the marginal CIs from Table 2) and either soften §3.1 to "ordering reproduced on bulk metrics; CSI@30 essentially unchanged on test_robust" or, if the paired CI does exclude zero, surface that number.

**N6 (round-1 C3 still open is the largest remaining gap).** Every quantitative claim in the paper is intra-author. Without one external baseline — pysteps STEPS as a free option, or a re-trained DGMR/Earthformer on the same split — the reader cannot anchor whether ab3's CSI@10 = 0.632 is competitive, mediocre, or state-of-the-art on this domain. The cost of pysteps STEPS on the same windows is low; the cost of *not* having it is that an editor at npj is likely to flag this as a basic methodological prerequisite. I treat this as a round-1 carryover, not a round-2 discovery, but it should not slip out of view simply because v2 added other things.

## 4. Minor comments

- **m1.** §3.3 mixes "30 dBZ contour" (a reflectivity unit) with CSI thresholds in mm h⁻¹ (a rain-rate unit, threshold τ = 30 mm h⁻¹). Pick one convention and apply it everywhere, including Fig 4. Round-1 M3 unfixed.
- **m2.** §1 still cites "Beucler 2021; de Bezenac 2018" in the third paragraph but neither appears in MAIN.md's bibliography list. Round-1 M5 unfixed.
- **m3.** §2.4 budget loss: Δp = 3 × 10⁴ Pa is described as "a 700–1000 hPa column-thickness proxy". This is a load-bearing scaling — the budget residual magnitude scales linearly with Δp — but the choice is not justified or sensitivity-tested. One sentence on why 3 × 10⁴ Pa rather than, e.g., 5 × 10⁴ Pa is appropriate.
- **m4.** §2.5 claim "bit-identical except CDU" between ab3 and ab3-cdu is contradicted in the same paragraph by "warm-started from ab3 best checkpoint, optimizer and LR schedule reset". Strike "bit-identical" and use "identical configuration except decoder block and warm-start".

## 5. Specific suggestions

**Must-do (gating acceptance):**
1. Resolve N1: either redefine the main ab3b as weight = 0.001 (and rewrite §3.2/§4.2 accordingly) or quantitatively justify why weight = 0.1 is the correct anchor despite being Pareto-dominated by weight = 0.001 in the sweep.
2. Resolve N3 (framing): make a clear editorial decision — is this a diagnostic / methods paper (in which case lead with pooled-CSI + budget-loss collapse and demote CDU) or a model paper (in which case the model needs to solve the headline problem). The current hybrid stance will read as overclaim.
3. Add at least one external baseline (round-1 C3 carryover, N6). pysteps STEPS on the same windows is the lowest-effort option.
4. Address N4: window-stratified bootstrap as a sensitivity check, with an explicit statement of what each CI is conditioned on.

**Nice-to-have:**
5. Resolve N2 with a per-window CDU–ab3 distribution panel.
6. Resolve N5 with the paired CI on test_robust for ab3 vs ab1 at CSI@30.
7. From-scratch ab3-cdu control (round-1 C6 carryover) — even one seed is enough to discharge the warm-start confound.
8. Minor consistency / bibliography sweep (m1–m4).
