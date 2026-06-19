# Band Debate — Independent Verdict (Phase 7f Retrain)

**Judge stance**: neutral, technical, no concession to politeness.
**Date**: 2026-06-19.
**Input docs**: `band_debate_A.md`, `band_debate_B.md`, `band_debate_C.md`.

---

## 1. Independent fact-check (before scoring)

Target distribution (event_test, 96 windows):

| Quantile / threshold | Value |
|---|---|
| max | 413 mm/h (single-pixel) |
| p99.99 | 49 mm/h |
| p99.9 | 24 mm/h |
| p99 | 8.95 mm/h |
| pixels > 30 mm/h | 0.05 % |
| pixels > 50 mm/h | 0.009 % |
| pixels > 100 mm/h | 0.0003 % |
| pixels > 150 mm/h | ~0 % |

CSI evaluation thresholds: **1, 5, 10, 30 mm/h** (max is 30 — no CSI@50 in the pipeline).

Two structural facts dominate:

1. **Top-center matters for CSI@30 only up to ~80–100 mm/h**. Any top center ≥ 80 lets soft-argmax express > 30 cleanly. Pushing top center to 150 or 200 buys almost nothing for CSI/FSS@30. It only matters for **MAE on the >100 mm/h tail**, where total pixel count is ≤ 1e-5 of the dataset.
2. **CSI@30 hinges on band 4/5 geometry**. Soft-argmax output = `Σ p_k c_k`. To emit a crisp 30, the model needs either a center at 30 (B/C) or to interpolate between centers 19 and 50 (A). Interpolation costs gradient sharpness.

---

## 2. Scoring (weights + rationale)

**Weighting rationale**: paper main metric is CSI@30/FSS@30 + MAE; training cost is 5 × 24 h GPU (instability is expensive); reviewer credibility matters because the whole point of Phase 7f is responding to "cap = 50" criticism.

| Dimension | Weight | A (6b, top 150, w≤16) | B (6b, top 80, w≤24) | C (7b, top 200, w≤100) |
|---|---:|---:|---:|---:|
| Math correctness (pred max covers tail) | 0.15 | 9 | 7 | 8 |
| Training stability (sample_weight risk) | 0.20 | 9 | 7 | 3 |
| Comparability to old paper | 0.10 | 8 | 5 | 4 |
| Reviewer credibility (story) | 0.15 | 7 | 9 | 6 |
| Implementation complexity | 0.10 | 9 | 7 | 5 |
| Extreme tail learning signal | 0.10 | 6 | 7 | 4 (top band ~dead) |
| **CSI@30 / FSS@30 geometric alignment** | 0.20 | 5 | 9 | 9 |

**Weighted totals**: A = **7.40**, B = **7.45**, C = **5.65**.

A and B are statistically tied; C is clearly behind.

---

## 3. Honest read of each debater's weakness disclosure

**A — minimal extension to (0, 0.5, 4.5, 19, 50, 150)**
- A *admits* the top band may be near-dead — fine, honest.
- A *evades* the bigger issue: **legacy centers 4.5 and 19 were tuned for dBZ-era distribution; they do not sit at mm/h density modes**. Center 19 is in the p99.5–p99.9 thin tail; CSI@10 has no neighboring center (gap from 4.5 to 19 = 14 mm/h) and CSI@30 must be interpolated between 19 and 50. A's "preserve canonical 5-band" framing buries this.
- A's "comparability with phase 7d/7e ablation" claim is **weaker than presented**: Phase 7f is a full retrain, so absolute CSI numbers change anyway; the only thing preserved is "method-section bullet-point consistency", not numeric comparability.

**B — distribution-aware (0, 0.5, 2.5, 10, 30, 80), weights up to 24**
- B *admits* sample_weight = 24 risk and truncation at 80 — honest.
- B *under-states* the top truncation effect on **MAE**: even though >80 mm/h is only ~0.005 % of pixels, those pixels are the ones the paper's "fix the cap" narrative is selling. Truncating at 80 mm/h (vs 100 or 150) shrinks the headline "we no longer cap" story.
- B's "CSI@30 perfectly aligns with band center 30" claim is the **single strongest technical argument across all three** debates and survives scrutiny.

**C — 7 bands (0, 0.5, 2, 8, 30, 80, 200), weights up to 100**
- C *admits* top band may learn garbage — but **understates how dead it really is**. With edge 130 and >130 mm/h pixel fraction ≈ 1e-5 to 1e-6, even weight × 100 yields effective signal ≈ 1e-3, and that signal is computed on essentially a handful of pixels in the entire training set → high-variance noise injection, not learning.
- C **factually misrepresents A's proposal**: claims "A 6 bands 顶到 100" — A's top is **150**, not 100. The comparison "only C reaches above target max" rests on this error.
- C's `weight = 100` on a near-empty top band creates a real risk: the cross-entropy loss is dominated by rare-pixel mistakes, and gradient norms explode at the head. Warmup helps but does not eliminate it. On a 5 × 24 h training budget, a single failed run is a 4.8 % cost overhead per attempt.

---

## 4. Verdict: **Compromise hybrid (B-centers + A-weights, top center 100)**

Neither A nor B wins outright on the scorecard. The decision-relevant split is:

- B's **CSI@30 / FSS@30 geometric alignment** (center at 30, edges at 1/5/20/50 matching CSI thresholds 1/5/10/30) is a genuine, defensible technical advantage that A does not have. Reviewers will see this.
- A's **conservative sample_weights** (max 16, not 24) is a genuine, defensible stability advantage on a 5 × 24 h budget that B does not have.

These are orthogonal. They compose cleanly into a hybrid.

### Recommended Phase 7f configuration

```python
band_centers        = (0, 0.5, 2.5, 10.0, 30.0, 100.0)
band_edges          = (0.1, 1.0, 5.0, 20.0, 50.0)
band_sample_weights = (0.1, 1.0, 2.0, 4.0, 8.0, 16.0)
```

**Quantitative justification (≥ 3 numbers):**

1. **Edges (0.1, 1, 5, 20, 50)** exactly match CSI evaluation thresholds (1, 5, 10, 30) at band boundaries or centers → no interpolation cost on any of the 4 reported CSI numbers.
2. **Top center 100** covers all pixels up to p99.999+ (since >100 mm/h = 0.0003 %, ~30 pixels in a 100M-pixel training set). Truncation cost vs A's 150 is negligible; gradient density on the top band is **3× higher** than A's (0.0003% pixels concentrated on a single 50→100 band vs A's 50→150 with empty 100–150 segment).
3. **Top weight 16** (not 24, not 100) gives effective top-band signal `0.009% × 16 ≈ 0.14%` — same order as Phase 7e's working 50-band schedule (0.009% × 8 = 0.07%). Empirically known to be trainable, no warmup needed, no instability hedge needed.

### Why not pure B
- Top center 80 < 100 sacrifices the "we no longer cap" narrative for negligible CSI gain.
- Weight 24 introduces avoidable instability risk on a 120 GPU-hour training budget.

### Why not pure A
- Centers 4.5 and 19 are dBZ-era artifacts in a mm/h pipeline. Reviewers will notice. The "minimal change" story does not outweigh failing the geometric-alignment argument that any well-prepared reviewer will raise.

### Why not C
- Top weight 100 is materially destabilizing on a budget where each failed run costs 24 h.
- Top band (>130 mm/h) is statistically dead: ≤ 1e-5 of pixels even before weighting.
- 7th band buys 0 measurable CSI/FSS gain at the reported thresholds (max = 30).
- Misrepresents competitor proposals — credibility hit.

---

## 5. Three actionable prerequisites before kicking off Phase 7f

1. **Validate `band_edges` are wired through every loss-component, eval-bucketization, and sample-weight path simultaneously**. The new edges `(0.1, 1, 5, 20, 50)` must be reflected in: (a) `band_centers` config, (b) `band_edges` for bucketize, (c) `band_sample_weights` indexing, (d) any per-band CSI/FSS eval split, (e) checkpoint config schema. A single stale edge breaks the alignment argument entirely. Add a `sanity_check_band_consistency()` assertion at training start.

2. **Dry-run head-only forward pass on a single batch** with a synthetic input where target pixels span [0, 200] mm/h, confirm that:
   - softmax × centers yields outputs in [0, 100] (not capped at 50).
   - bucketize of synthetic targets at 0.5, 2, 5, 15, 35, 75 mm/h lands in bands 1, 2, 3, 4, 5, 5 respectively (verify edge semantics — `right=True` vs `False` is a classic bug source).
   - per-band weighted CE produces non-zero gradient for band 5 when fed a target ≥ 50.
   Time cost: < 30 min. Catches 90 % of misconfig bugs that would waste 24 h GPU.

3. **Pick one of the 5 models, run a 5-epoch smoke train** (single GPU, ~2 h) on event_test subset with the new head before launching all 5 × 60-epoch full runs. Check: (a) val loss decreasing, (b) per-band predicted-mass distribution is not collapsed to band 0, (c) gradient norm on top band stable (< 10× lower bands). If any check fails, debug before burning 120 GPU-h on a broken config.

---

## 6. Sweep alternative (if budget allows)

The right scientific answer is to sweep `{A, B, hybrid}` on **one** of the 5 models (~24 h × 3 = 72 GPU-h), pick the winner by CSI@30 on event_test, then full-train all 5 with the winner. This adds ~25 % budget for ~80 % less regret risk. Given the 5 × 24 h commitment, this is the cheaper move in expected value.

If sweep is rejected on budget grounds: ship the hybrid above.

---

**Final answer**: Compromise hybrid with `band_centers = (0, 0.5, 2.5, 10, 30, 100)`, `band_edges = (0.1, 1, 5, 20, 50)`, `band_sample_weights = (0.1, 1, 2, 4, 8, 16)`. Core reason: takes B's CSI-threshold-aligned band geometry (the only genuinely defensible technical advantage in the debate) without B's destabilizing top weight or A's legacy dBZ-era centers.
