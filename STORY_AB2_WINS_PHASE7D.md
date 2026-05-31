# Phase 7d — ab2 (radar+PWV) advantage cases & PWV mechanism story

Branch `phase7d/ab2-win-cases`. Built on the Phase-7c re-split (May–Aug
radar∩PWV, Sept dropped) and the 60-epoch retrains `ablation_{1,2}_new_7b_p7c`.

## TL;DR (honest framing)

The +PWV benefit is **mixed, not a clean ab2 sweep**. Aggregate metrics: ab2
edges ab1 on `csi_10mm / crps / fss`, but ab1 is *higher* on `csi_30mm`
(heavy rain) on the small event sets — `gated_fusion` appears to trade some
heavy-rain skill for light-rain/crps. The only clean causal evidence that PWV
contributes heavy-rain skill is the **`drop_pwv` counterfactual inside ab2**
(remove PWV → `csi_30mm` collapses ≈3×). The raw ab1-vs-ab2 gap is confounded
by architecture, so the paper argument leads with the counterfactual, not the
bare ablation. See `project_phase7c_results` for the metric tables.

This phase produces the **case-level visual evidence** to accompany that
argument: representative forecast starts where ab2 genuinely beats ab1, plus
the per-pixel PWV mechanism panels on those same starts.

## How the cases were selected

`scripts/scan_ab2_advantage.py` scored **480 forecast starts** across
`val + test_robust + event_test`. For each start it computes ab2−ab1 CSI
advantage and the storm-region error reduction `(|ab1−GT| − |ab2−GT|)`. Output
`eval/scan_ab2_advantage.csv`, sorted by `err_red_storm`. Filtering for clean
wins (`csi_adv>0 & err_red_all>0 & err_red_storm>0 & storm_frac>2% &
gt_max>50 dBZ`) leaves 64 starts; the strongest cluster on three days.

Three representative starts (span all three splits, all heavy-rain `gt_max>60`):

| start | split | gt_max | csi_adv | err_red_all | err_red_storm |
|-------|-------|-------:|--------:|------------:|--------------:|
| 2023-05-19T10:00 | val | 66.9 | **+0.057** | +0.40 | +1.33 |
| 2023-08-08T06:00 | test_robust | 65.9 | +0.043 | **+0.67** | +1.46 |
| 2023-08-01T08:00 | event_test | 60.6 | +0.026 | +0.38 | **+1.77** |

`2023-05-19T10:00` is the cleanest lead figure: GT has a compact 50+ dBZ core
that ab1 under-predicts and over-spreads; ab2 restores the core intensity, and
the `ab2−ab1` diff is positive exactly over the storm.

## Figures

- **4-row comparison** `ckpt/figures/ab2_wins/cmp_<start>.png`
  (`scripts/plot_ab1_vs_ab2.py`): rows GT / ab1-new / ab2-new / (ab2−ab1),
  columns +18/+36/+72/+108 min, dBZ NWS colormap, diff in RdBu_r ±30 dBZ.
  Same paradigm as `figures/event_23_7_ab1_vs_ab2.png`.
- **PWV mechanism** `ckpt/figures/mechanism/win_<start>.png`
  (`scripts/plot_pwv_mechanism_demo.py`): (a) PWV input field, (b) learned
  per-pixel gate `g_pwv`, (c) counterfactual ab2_PWV − ab2_noPWV, (d) error
  reduction |ab1−GT|−|ab2−GT|.

## Mechanism — what the panels actually support (and what they don't)

Per-pixel spatial correlations on the three win cases:

| start | corr(gate, \|cf\|) | corr(PWV, err_red) | corr(gate, err_red) |
|-------|------------------:|-------------------:|--------------------:|
| 2023-05-19T10 | **+0.31** | +0.09 | −0.01 |
| 2023-08-08T06 | +0.28 | −0.31 | −0.01 |
| 2023-08-01T08 | +0.18 | −0.00 | −0.10 |

**Defensible claim:** `corr(gate, |counterfactual|)` is consistently positive
(~0.2–0.3). Where the learned gate opens, PWV actually moves the forecast — the
gate is a real, interpretable control on the PWV branch. On `2023-05-19T10` this
is visually obvious: the bright gate band and the counterfactual band coincide.

**What we do NOT claim:** a clean "PWV-high ⇒ ab2-better" co-location.
`corr(PWV, err_red)` is weak and even negative on `2023-08-08T06`. Error
reduction is driven by storm structure, not by raw moisture amount. Overselling
a moisture↔skill spatial story would not survive review.

**Suggested next step for stronger attribution:** Integrated Gradients
(∂rain_pred/∂PWV, baseline = zeroed PWV) is more stable than bare gradients/ERF
when the gate saturates — add it as panel (e) to answer "where does the signal
come from" without relying on Grad-CAM/saliency.

## Caveat

`csi_30mm` and storm-region stats are noisy: each 8-day val/test set has only
1–2 heavy-rain days. These cases are illustrative, not statistically
conclusive; aggregate claims still rest on the `drop_pwv` counterfactual plus
(recommended) bootstrap CIs.
