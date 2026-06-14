# Pluvian Route Decision Notes

Updated: 2026-06-02 UTC.

This note records the current decision logic for the Phase 7d goal. It should
be refreshed after ab3 ERA5 holdout evaluation finishes.

## 1. csi30 short-lead regression: loss reweighting vs lead-aware gate

Current evidence:

- The regression is localized: event_test csi30 is worse mainly at +18..+42 min,
  while event_test csi1/csi5/csi10 improve broadly and test_robust csi30 is flat
  to slightly positive.
- xcoreB shows loss reweighting can recover most of the event csi30 gap
  (`csi30_short_delta`: -0.0384 -> -0.0099), but it also damages event csi1,
  csi5, csi10, FSS, and CRPS relative to xcoreA.
- xcoreA is a safer fallback but still leaves a meaningful short-lead csi30 gap
  (`csi30_short_delta`: -0.0267).

Judgment:

- Loss reweighting is cheap and already partially validated, but it is blunt:
  it changes the global objective and tends to buy extreme-core hits by spending
  light/moderate skill and calibration.
- A lead-aware gate is the cleaner mechanism if the final ab3 diagnosis still
  says the problem is specifically "PWV dilutes radar extrapolation at early
  leads." It targets the suspected failure mode directly: keep radar dominance
  at short leads, let PWV/ERA5 contribute more as advection/thermodynamic
  uncertainty grows.
- Do not build the gate before ab3 holdout unless ab3 clearly fails in the same
  localized way. It is an architecture change and creates checkpoint/interface
  risk; the project already has a lower-risk xcore fallback.

Success probability and cost estimate:

| route | expected success | cost | main risk |
|---|---:|---:|---|
| xcoreA fallback | medium for aggregate skill, low for fully closing csi30 gap | already done | csi30 gap remains |
| stronger xcore | medium for csi30, low for balanced paper metric | 1-3 GPU h per small FT | sacrifices csi1/csi10/FSS/CRPS |
| lead-aware gate | medium-high if ab3 repeats localized early-lead failure | ~1 day coding + 5-20 GPU h validation | architecture confound; may reduce genuine PWV benefit |

Recommended trigger: if ab3/ab3b does not fix event_test csi30_short_delta and
still preserves the same +PWV light/moderate gains, run one lightweight
lead-aware gate experiment on validation only before touching holdout again.

Implementation note from code inspection:

- Current PWV/radar fusion happens in `model/fusion.py::GatedAsymmFusion`: the
  model computes a PWV cross-attention delta and adds `g_pwv * pwv_delta` to the
  radar latent. This gate has input-history time and latent-pixel axes, but no
  forecast-lead axis.
- Therefore the minimal lead-aware fix should not be a tweak to the existing
  `gate_pwv` alone. The forecast leads are only created later in the decoder's
  lead-time attention.
- Lowest-risk architecture experiment: expose the PWV contribution as a
  separable latent term, compute decoder output with and without that PWV term,
  then interpolate by a fixed per-lead scale. Use no learned parameters and keep
  default disabled so old checkpoints remain compatible.
- Candidate fixed schedule: attenuate only lead indices 2..6, i.e. +18..+42 min,
  with an alpha around 0.5. Validate against xcoreA/B on the same per-lead
  event_test/test_robust metrics.
- Key implementation hazard: validation calls the optimized `_encode_and_fuse`
  and decoder path directly, so any lead-aware path must update train forward,
  validation, eval, and MC-dropout consistently.

## 2. Track C physics vs Track B generative ensemble

Judgment: prioritize Track C physics for the main npj claim; keep Track B as a
secondary uncertainty/sharpness lane.

Reasons:

- Track C is closer to the paper's unique claim: radar + GNSS-PWV + ERA5 moisture
  physics. A water-vapor budget residual is directly tied to why PWV should help
  precipitation nowcasting.
- Track C gives reviewers two kinds of evidence: forecast skill and physical
  consistency. Track B mainly gives calibrated/sharp samples, which is valuable
  but less specific to Pluvian's dual-modal water-vapor contribution.
- Current Track B state is not ready for a main claim: diffusion residuals are
  invalid due to amplitude overrun, and GAN-coreA is currently a sharpness result
  rather than a deterministic-skill result.
- The budget-loss code path is now smoke-tested on remote GPU. The remaining
  dependency is the ab3 holdout result, not a code blocker.

Recommended order:

1. Finish ab3 ERA5 and clean holdout/per-lead evaluation.
2. If ab3 is neutral or positive, launch clean ab3b budget loss from the final
   ab3 checkpoint.
3. Generate physical diagnostics: budget residual histograms/maps, residual vs
   rain-intensity bins, and residual changes on event cases.
4. Only then return to Track B CRPS/reliability/sharpness ensembles.

Reference anchors to cite in the manuscript/literature table:

- GNSS/PWV motivation: Bevis et al. GPS meteorology; storm-nowcasting GNSS IWV
  and tomography studies; recent radar + GNSS-PWV deep-learning nowcasting.
- Moisture physics: moisture-flux convergence / water-vapor conservation as a
  convective-initiation diagnostic; ERA5 as the gridded u/v/q source.
- Physics-AI nowcasting precedent: NowcastNet and hybrid physics-AI extreme
  precipitation nowcasting work.
- Probabilistic/sharpness precedent for Track B: DGMR, STEPS/pySTEPS, and CRPS
  scoring-rule literature.

## 3. Sharpness vs skill manuscript lanes

Keep them separate.

- Skill lane: CSI, FSS, CRPS/MAE, per-lead, event_test/test_robust, ablations.
- Sharpness lane: RAPSD/high-k, p99/p99.9, overrun rate, qualitative samples.
- GAN-coreA can support "spectral realism / sharper samples" but not "better
  forecast skill" unless the same checkpoint wins CSI/FSS/CRPS under the fixed
  holdout protocol.
- Diffusion residuals stay a negative result until amplitude control is fixed.

## 4. Higher-ROI additions

- Bootstrap confidence intervals for the main ab1/ab2/ab3 deltas. This is cheap
  and helps defend small CSI gains.
- Physical-consistency diagnostics even before full budget training: compute the
  water-budget residual for ab2/ab3 predictions to show whether better skill also
  corresponds to lower physical residual.
- Error decomposition for event cases: hits/misses/false alarms by lead and
  intensity band, plus displacement vs amplitude examples.
- Lead-aware gate only as a targeted follow-up if ab3/ab3b confirms the same
  short-lead extreme-core failure.

Bootstrap CI status: implemented for ab1/ab2/xcoreA/xcoreB in
`docs/experiments/pluvian_bootstrap_ci.md`, with paired candidate-minus-baseline
deltas in `docs/experiments/pluvian_bootstrap_delta_ci.md`. The paired table is
the cleaner manuscript evidence: ab2 improves event_test light/moderate metrics
and CRPS versus ab1, but csi30 remains a negative delta. xcoreB mostly buys
csi30 at the cost of csi1/csi10/FSS/CRPS.

## 5. Visualization queue

Can do before ab3 finishes:

- Sync xcore aggregate/per-lead JSON and figures from remote so local docs have
  the same evidence as remote.
- Generate one honest ab2 failure-case grid for the csi30 caveat, e.g.
  `2023-07-29T06:00:00`.
- Run drop-PWV counterfactual eval for ab2 on event_test/test_robust. This is a
  diagnostic for PWV influence, not a replacement for radar-only ab1.

Require ab3/ab3b outputs:

- ab3 holdout summary table and per-lead curves.
- ab1/ab2/ab3 case grids and error maps.
- ab3b budget residual maps/histograms and residual-vs-intensity diagnostics.
