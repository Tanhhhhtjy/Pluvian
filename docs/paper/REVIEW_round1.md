# Reviewer #1 Report — npj Climate and Atmospheric Science

Manuscript: *Pluvian: Multimodal Nowcasting with Pooled-CSI Diagnostic of Physics-Loss Trade-offs*

## 1. Summary & Recommendation

The authors present a three-stream nowcasting model (radar + GNSS PWV + ERA5) over North China, run a four-step ablation (ab1/ab2/ab3/ab3b/ab3-cdu), and use a max-pooled CSI sweep to argue that a water-budget PDE loss degrades extreme skill via *physical smoothing* rather than *displacement*. The negative-result framing and the smoothing-vs-displacement diagnostic are genuinely useful contributions.

**Recommendation: Major Revision.** Three issues block acceptance in current form: (i) §3.4 (CDU recovery), which Introduction explicitly promises as contribution (iii), is empty; (ii) statistical evidence for the headline "collapse" is weaker than claimed — the 95% bootstrap CIs for ab3 and ab3b CSI@30 overlap; (iii) no external baseline (DGMR, NowcastNet, Earthformer, MetNet-3) is included, so the absolute skill level is unanchored.

## 2. Strengths

- **S1. Honest negative result.** Reporting a physics-loss configuration that breaks the very metric a domain reader would care about (CSI@30, 0/18 leads positive) is uncommon and valuable. §4.2's mechanism argument (pointwise quadratic penalty on a noisy reanalysis derivative biases the optimiser towards smooth fields) is well-stated.
- **S2. Disciplined ablation skeleton.** ab1→ab2→ab3 share an identical backbone, schedule, loss and seed (§2.5), and the per-lead reporting on event_test makes the "0/18 vs 2/18" comparison auditable.
- **S3. Pooled-CSI is genuinely cheap to add.** A one-parameter sweep over pool window w that any group can drop into existing pipelines is a low-friction community asset, regardless of novelty (see C4).
- **S4. Method writing is concrete.** Domain (36–42.6°N, 113–120°E), grid (661×701), Z–R, frame split (12+18) and budget-loss form are all reproducible.

## 3. Major Concerns

**C1. Headline "collapse" is not supported by the reported uncertainty.**
Table 1 gives ab3 CSI@30 = 0.291 [0.264, 0.327] and ab3b CSI@30 = 0.210 [0.188, 0.270]; Table 2 gives 0.276 [0.236, 0.292] vs 0.215 [0.178, 0.244]. The 95% bootstrap intervals overlap on both splits. The Abstract, §3.2 and §4.1 nonetheless state "collapses by 28% / 22%" as a definitive effect. With n=4 storm days (event_test) and n=8 (test_robust), day-stratified resampling has at most 4⁴ = 256 / 8⁸ distinct draws — the n_boot = 1000 figure overstates information content. Required: (a) a paired test on day-level (or window-level) score differences, not just marginal CIs; (b) explicit acknowledgement that the effect size, while directionally consistent, is within bootstrap noise on event_test.

**C2. §3.4 is empty but Introduction sells it as a contribution.**
Introduction contribution (iii) and the Abstract both claim CDU "restores extreme skill without reweighting losses". §3.4 reads verbatim "Results pending". §4.4 then constructs an "if CDU recovers... if it does not..." disjunction that is rhetorically self-insulating. As reviewed, the paper has *no* evidence for the architectural prescription it advertises. Either complete §3.4 and align Abstract/§1, or remove the CDU claim from contributions and rewrite §4.4 as a stated next step.

**C3. No comparison against published nowcasters.**
DGMR, NowcastNet, MetNet-3, Earthformer, and exPreCast are cited but never benchmarked. Every reported gain is intra-author. A 6.66M-param model on a 4-storm-day test set, without an external reference, cannot support phrases like "closed much of this gap" (§1). At minimum, one open-source baseline (pysteps STEPS, or a public DGMR/Earthformer checkpoint retrained on the same split) is needed to anchor absolute skill.

**C4. Pooled-CSI novelty is overstated.**
Neighbourhood / fuzzy verification is a mature literature (Ebert 2008; Gilleland 2009; Roberts & Lean 2008; Mittermaier 2014). Max-pooled binary fields at multiple windows is essentially the "minimum coverage" / "upscaling" family. The contribution here is the *diagnostic use* (smoothing vs displacement) rather than the construction. §3.3 and §4.3 should reframe accordingly and cite the prior fuzzy-verification literature. Additionally, the choice of non-overlapping stride-w max-pool (vs the more common stride-1 sliding) is justified by alignment with one citation (Song 2026); a side-by-side with stride-1 results in the appendix would let readers calibrate the difference.

**C5. ab3b failure attribution is not unique.**
§4.2 attributes the collapse to "pointwise quadratic penalty geometry". An equally simple explanation — that the budget-loss weight (0.1) and Huber δ (1e-4) are simply mis-tuned — is not ruled out. Required: a sweep of the budget weight (e.g., 0.01, 0.001) or a single run at a smaller weight showing the same qualitative trade-off. Without this, "physics-informed loss design is structurally biased" (§1, §4.2) is one experiment short of the claim.

**C6. ab3-cdu is not a clean ablation against ab3.**
§2.5 states ab3-cdu is *warm-started from the best ab3 checkpoint* with optimiser and LR reset, while ab1/ab2/ab3/ab3b are from-scratch. Parameter count also rises (+0.25M). The Method section calls ab3→ab3-cdu "bit-identical except CDU"; that is true at the code level but not at the training-trajectory level. When §3.4 is filled, a from-scratch ab3-cdu control is needed, otherwise any recovery is confounded with the warm-start regularisation effect.

**C7. Generalisability claim outruns the data.**
Train (74 days) / test (4 + 8 days), May–Aug 2023, single region. The framing in §1 and §4 generalises to "physics-informed loss design in nowcasting" — a discipline-wide claim from twelve evaluation days in one warm season. §4.5 (a) flags this but the abstract and contribution statements do not. Recommend softening (i) and (iii) in §1 to scope-bounded language.

**C8. CRPS degeneracy admitted in §2.4 should be cleaned up, not parenthesised.**
"`_crps_marginal` degenerates to single-member CRPS, which is algebraically equal to MAE" is a code bug in the verification pipeline. Either compute a real ensemble CRPS or drop CRPS from §2 entirely.

## 4. Minor Comments

- **M1.** Editorial HTML comments (`<!-- editorial notes -->`) leak through every section — strip before submission. They also reveal unresolved internal disagreement (Z–R citation, hardware, pool stride).
- **M2.** Abstract says "monotonic improvement across CSI at 1/5/10". CSI@30 is *not* monotonic and ab1 actually wins on event_test (0.308, bolded in Table 1). Rephrase to "monotonic at 1/5/10; non-monotone at 30".
- **M3.** Table 1 caption mixes mm h⁻¹ thresholds with a 30 dBZ contour reference in Fig 4 caption (§3.3). Pick one and stay consistent.
- **M4.** PWV preprocessing — "seven near-empty stations are dropped per our data audit" (§2.1) — is a hand-tuned filter; list the station IDs in supplementary for reproducibility.
- **M5.** §1 cites "Beucler 2021; de Bezenac 2018" but the bibliography in MAIN.md lists neither. Reconcile before submission.

## 5. Specific Suggestions

**Must-do (gating acceptance):**
1. Fill §3.4 with CDU results, or remove the architectural prescription from contributions/abstract.
2. Add paired day-level significance test for the ab3 vs ab3b CSI@30 gap; report effect with proper uncertainty.
3. Add at least one external baseline on the same split.
4. Loss-weight sensitivity run for ab3b (to discharge C5).

**Nice-to-have:**
5. From-scratch ab3-cdu control alongside warm-started version.
6. Stride-1 pooled-CSI variant in appendix.
7. One non-summer or non-North-China day as a stress test, even if results are negative.
8. Drop CRPS column or implement real ensemble CRPS.
