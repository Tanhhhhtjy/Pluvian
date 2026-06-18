# Pluvian Phase 7d Experiment Ledger

Updated: 2026-06-16 UTC.

This ledger records post-handoff experiment facts for the active Pluvian goal. It is intentionally separate from `HANDOFF_PHASE7D.md`, which is a time-stamped handoff snapshot.

## 2026-06-16 Phase 7e (CRITICAL UNIT FIX) — paper-wide dBZ-vs-mm/h discrepancy

While preparing the SOTA baseline survey (round-1 reviewer C3) we discovered that the paper's CSI thresholds, FSS thresholds, MAE values and `band_centers` are all written in mm h⁻¹, but the dataset was emitting raw dBZ. Concretely:
- `pipeline/data_loader.py::NPJDataset.__getitem__` returned the radar tensor directly from `radar_io.load_radar_sequence`, which is xarray-decoded dBZ (typical range 0–70 dBZ)
- `scripts/train.py::_intensity_loss` bucketizes the target with `band_edges = [0.1, 1.0, 8.0, 30.0]` labelled mm/h
- `scripts/train.py` validate / eval CSI thresholds = `(1.0, 5.0, 10.0, 30.0)` labelled mm/h
- `model/decoder.py` `band_centers = (0.0, 0.5, 4.5, 19.0, 50.0)` with internal comment "mm/h"

So the rain head was being trained against dBZ targets while every label said mm/h. CSI numbers are self-consistent (CSI is a well-defined metric on any threshold), but the paper's "CSI@30 = extreme rain" framing is false: 30 dBZ ≈ 1.6 mm/h is moderate rain, not extreme.

Reviewer round 1 (M3) and round 2 (m1) both flagged the inconsistency between "30 dBZ contour" in Fig 4 and "mm/h" in the metric tables. We had interpreted it as a Fig 4 caption issue; it is paper-wide.

Fix landed (commit `300b9cb`): dataset now applies `utils.dbz_to_rainrate` (Z = 300 R^1.4) before returning the radar tensor. All downstream loss / metric / band_centers definitions are now consistent with the paper text.

**Implications:**
- All five trained checkpoints (ab1, ab2, ab3, ab3b, ab3-cdu) are invalidated and must be retrained from scratch under mm/h.
- New event_test pixel distribution: max 103.8 mm/h, mean 0.54, >30 mm/h: 0.04% (vs old >30 dBZ: 6.4%). Statistical power on the extreme tail will be lower.
- The ab3 vs ab3b CSI@30 collapse story may shift in magnitude after retraining; the qualitative direction is expected to hold but cannot be claimed without re-running.
- All paper figures, tables, paired CIs and the ledger metric snapshots from before 2026-06-16 are now stale.
- Phase B (SOTA baseline retraining) is blocked until Phase 7e completes, because external baselines must train against the mm/h target distribution to be comparable to ours.

**Current blocker (2026-06-16 14:00 CST):** GPU sharing — `nvidia-smi` shows all 8 GPUs occupied by an unrelated user (ZhaoJiaming, streaming-VLM training, ~24 GB / GPU). Each GPU has ~24 GB free vs our prior training footprint of ~33 GB / model. Smoke test (`logs/unit_fix_smoke.log`) OOM'd on `GroupNorm` allocation. Options awaiting user decision:
  1. Wait for the other user's job to release GPU memory
  2. Try `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to fit into ≤24 GB
  3. Reduce `grad_accum` from 4 to 2 (halves activation memory but changes effective bs)
  4. Run the five retrains serially (~4 days wall-clock instead of ~36 h)

**ETA refresh (2026-06-16 14:47 CST, cron poll):** TensorBoard scrape shows ZhaoJiaming's job at step 50 / ~134 (epoch 1.16 of 3), advancing at ~1.6 min/step. Expected free GPUs around **17:00–17:30 CST**. Strategy when GPUs free up: launch all 5 retrains in parallel (one per GPU); each ~18–20 h, so the full Phase 7e retraining completes around 2026-06-17 11:00–13:00 CST. Cron will not auto-launch — user must signal go.

**ETA refresh (2026-06-16 15:23 CST, cron poll):** ZhaoJiaming now at step 100 / ~134 (epoch 2.33 of 3), advancing at ~1.4 min/step (50 steps in ~36 min). GPUs 0 and 7 dropped to 0% utilization (memory still allocated, not yet freed). Remaining ~34 steps × 1.4 min ≈ **48 min → expected GPU release ~16:11 CST**. Five-way parallel retrain wall-clock now estimates completion around 2026-06-17 10:00–12:00 CST.

**Window OPEN (2026-06-16 16:23 CST, cron poll):** ZhaoJiaming's first job finished (no rank held its full memory). 6 GPUs are now fully idle (≥48 GB free each) and 2 GPUs (1 and 2) hold ≈24 GB of leftover allocation but are at 0% utilization — likely a new shorter job he started (TensorBoard now reports step 40 / epoch 0.12, fresh run). Five-way parallel retrain on GPUs {0, 3, 4, 5, 6} is now feasible without contention. **Awaiting user GO signal** — the cron autopilot will not launch retraining unsupervised.

**LAUNCHED (2026-06-16 22:09 UTC / 06:09 CST + 1 day, user GO):** Smoke test passed (5 batches under mm/h, no NaN, loss healthy). Five-way parallel retrain dispatched:
  - GPU0 ab1 (radar baseline) — `configs/ablation_1_p7d.yaml`, PID 2020763
  - GPU3 ab2 (xcoreA, +PWV) — `configs/ablation_2_p7d_xcoreA.yaml`, PID 2020764
  - GPU4 ab3 (+ERA5) — `configs/ablation_3_era5_p7c.yaml`, PID 2020765
  - GPU5 ab3b (+budget loss, weight=0.1 main) — `configs/ablation_3b_era5_budget_p7c.yaml`, PID 2020766
  - GPU6 ab3-cdu (**from-scratch**, no warm-start) — `configs/ablation_3_cdu_p7d.yaml`, PID 2020767. Note: this is a deliberate change vs the previous run which warm-started CDU from ab3 best.pt — running from-scratch addresses reviewer C6 (warm-start vs from-scratch confound) at no extra cost.

All 5 GPUs at 100% utilization, 25–33 GB each. Initial step-by-step loss curves are healthy (loss decreasing 0→20 ≈ 35–45 → 15–30 across step 80). ab3b budget loss = 5e-4 (5× the dBZ-era value), which makes the PDE term meaningful in mm/h units. Estimated wall-clock ≈ 18–20 h ⇒ all five checkpoints expected by **2026-06-17 16:00–18:00 UTC**.

**Progress refresh (2026-06-16 22:46 UTC, +37 min, cron poll):** All 5 PIDs alive (etime 37:37). Per-epoch wall-clock: ab1 1014.6 s, ab2 1027.3 s, ab3 1029.8 s, ab3b 1069.5 s, ab3-cdu 1267.3 s. After completing epoch 1, projected total time = 60 × ~17–21 min ≈ **17–21 h**. Loss decreasing healthily across all five (ep0 ~35–45 → ep1 ~31–45, ab2 highest at 44.6 as expected from PWV cross-attention warmup). No NaN, no OOM, no rank failure. Continue monitoring.

**Progress refresh (2026-06-17 23:46 UTC, +97 min, cron poll):** All 5 PIDs alive (etime 1:37:17). Epoch-avg loss continues healthy decrease across all five: ab1 29.12→28.73, ab2 41.71→40.97, ab3 28.97→28.64, ab3b 28.89→28.72, ab3-cdu 29.79 (ep2→ep3). Iter-level spikes (loss>100) observed across all models in recent 50 lines (count: ab1 2, ab2 8, ab3 3, ab3b 2, ab3-cdu 4) — interpreted as **bf16 batch-level noise** (mm/h max ≈ 103 means a single batch hitting an extreme cell can momentarily produce large loss). Epoch averages staying flat-to-decreasing rules out divergence; do not intervene. Average epoch wall-clock now ~17.5 min ⇒ remaining 55 epochs × ~17.5 min ≈ **16 h**, completion ETA **2026-06-17 14:30–15:30 UTC** (CST 22:30–23:30).

**Status refresh (2026-06-17 17:30 UTC, +19h15m cron poll):** Three runs **finished cleanly** (60 epoch) — ab1, ab3, ab3b. Two still running — ab3b at ep58 it1120/1664, ab3-cdu at ep49 it420/1664. Final epoch losses for the three finished: ab1 21.97, ab3 21.80, ab2 37.76 (ab2 only ep14, see below); ab1 val csi_1mm=0.291 / csi_30mm=0.104, ab3 val csi_1mm=0.316 / csi_30mm=0.102 — these are end-of-training val metrics on mm/h units. Best ckpt selection is by val csi_10mm. Remaining ETA: ab3b ~2 h, ab3-cdu ~5 h.

**⚠️ Config bug discovered (cron flagged; do not auto-fix):** When launching the 5-way retrain we picked `configs/ablation_2_p7d_xcoreA.yaml` for ab2, but xcoreA is a fine-tune config (lr=5e-5, max_epochs=15, **different intensity-band weights** `[0.1, 1.0, 2.0, 5.0, 12.0]` vs the standard `[0.1, 1.0, 2.0, 4.0, 8.0]`, and FSS threshold weights `[1.0, 0.5, 0.5]` vs `[1.0, 0.1, 0.1]`). The paper's main-line ab2 is `ablation_2_p7d.yaml` (from-scratch 60 epoch, default loss). The current ab2 ckpt is therefore not comparable to ab1/ab3/ab3b/ab3-cdu under the same training budget and loss weights.

**Options awaiting user decision:**
  1. **Re-run ab2 with `ablation_2_p7d.yaml` (60 epoch from-scratch, ~17 h)** — keeps ablation table strictly comparable. Cost: 1 GPU-night extra.
  2. **Keep the current xcoreA ckpt as "ab2"**, and document in paper §3.1 / Method that ab2 is a fine-tune variant with 15 epoch + adjusted loss weights. Cost: 0 GPU-h but weakens "shared schedule + loss" claim, easy reviewer target.
  3. **Drop xcoreA entirely** and report only ab1 → ab3 (skip ab2). Cost: 0 GPU-h but loses the +PWV ablation row.

Cron autopilot will not launch ab2 re-run without explicit user GO.

The cron-driven autopilot is now in monitor mode (will not auto-relaunch any failed run).

## 2026-06-15 / 16 Update: CDU lands, budget-weight sweep finishes, round-2 review surfaces cherry-picking risk

### What finished

- **CDU training** (`ckpt/ablation_3_cdu_p7d/best.pt`): 60 ep complete. Holdout — event_test: CSI@1 0.488, CSI@10 0.645, CSI@30 0.285, MAE 4.08 (vs ab3 0.470/0.632/0.291/4.27). Paired delta CI vs ab3 on CSI@30: event [-0.0096, -0.0002] (marginal but significant micro-regression); test_robust [-0.0045, +0.0032] (null). **CDU avoids the ab3b trade-off but does not recover the extreme tail** — the framing in §3.4 / §4.4 / Abstract reflects this honestly.
- **Budget-weight sweep** (`ckpt/ablation_3b_era5_budget_w{001,0001}_p7c/best.pt`, 60 ep each): finished. CSI@30 (event_test / test_robust) per weight: 0.276/0.263 (w=0.001) → 0.212/0.211 (w=0.01) → 0.210/0.215 (w=0.1, main). Trade-off saturates at w≥0.01; only w=0.001 brings CSI@30 close to ab3, but at that weight the budget loss value is ≈1e-5 (vs data loss ~10). Captured in §3.2.

### Round-2 reviewer concerns (see `docs/paper/REVIEW_round2.md`)

- **N1 (most dangerous, cherry-picking)**: at weight=0.001, ab3b Pareto-dominates the main weight=0.1 ab3b on bulk metrics (CSI@1 0.507 vs 0.492, MAE 3.97 vs 4.10), while paying only -5% CSI@30 instead of -28%. Reviewer asks why w=0.1 is the main ab3b. The "loss=1e-5 so the physics term is not working" defence is *post-hoc* — if it is not working, why are bulk metrics still moving by +0.037 CSI@1 vs ab3? Most likely explanation: all three ab3b variants are warm-started from ab3 best.pt and trained 60 more epochs with a fresh cosine LR, so part of the bulk gain is plain fine-tuning, not the budget loss. **Mitigation in §3.2**: added an explicit confound paragraph stating the warm-start risk and noting that the CSI@30 collapse is observed only in the two variants with non-trivial budget weight, so the negative-extreme finding is robust. **Pending decision** (needs user): either (a) accept this written caveat, or (b) train an ab3-warmstart control (~18h GPU) to fully isolate the budget effect, or (c) re-designate w=0.001 as the main ab3b and rewrite §3.2 / §4.2 with the smaller -5% effect size.
- **N2**: §3.4 says CDU is "essentially neutral at extreme tail" while admitting the 7-30 09:00 case has CDU output max 26 dBZ vs GT 46 dBZ. Reviewer wants per-window CSI@30 (CDU-ab3) distribution rather than a single number — currently not reported.
- **N3**: with CDU now framed as "doesn't recover", contribution (iii) is reduced to a small architectural win. Reviewer suggests either re-framing the whole paper as a *diagnostic / negative-result methods paper* (with CDU demoted to a confirming ablation) or splitting CDU into a separate application-track submission.

### What's left for v3 / submission

- (User decision) Resolve N1 (caveat vs new run vs re-designate main ab3b).
- (User decision) Re-frame N3: methods paper vs application paper.
- Fig 1 (architecture schematic) and Fig S1 (input example) — need user style call.
- Bibliography: placeholders to real BibTeX (agent-prone to fabrication, leave to user pass).
- External baseline (round-1 C3) — non-trivial; defer to round-2 revision or list as limitation.
- Optional ab3-warmstart control if N1 mitigation by writing alone is judged insufficient.



## 2026-06-14 Decision Update: ab3 is the main-line model; ab3b budget loss rejected

New holdout facts since 2026-06-02 (all clean event_test + test_robust):

| model | csi1 | csi10 | csi30 | fss3 | crps |
|---|---:|---:|---:|---:|---:|
| ab1 (radar) | 0.397 | 0.565 | 0.308 | 0.579 | 5.19 |
| ab2 (+PWV) | 0.432 | 0.591 | 0.283 | 0.614 | 4.83 |
| ab3 (+ERA5) | 0.470 | 0.632 | 0.291 | 0.651 | 4.27 |
| ab3b (ab3+budget PDE loss) | 0.492 | 0.641 | 0.210 | 0.672 | 4.10 |

- ab3 ERA5 is the new main-line model: best csi1/csi10/fss/crps across all models, and its per-lead csi30 delta vs ab1 on event_test is only -0.017 (best csi30 of any fusion model, nearly recovering the ab1 baseline).
- ab3b water-budget loss is **rejected as a main-line loss**: it lifts light/moderate csi + crps but collapses csi30 (event 0.291->0.210, robust 0.276->0.215) uniformly across all 18 leads (per-lead delta -0.084, 0/18 positive).
- Pooled-CSI diagnostic (`scripts/eval_pooled_csi.py`, max-pool 1/4/16) proves the ab3b csi30 loss is a true physical smoothing-away of strong cores, not spatial displacement: ab3b csi30 pool1->pool16 only 0.210->0.222, still far below ab3 pool1 0.291; ab3 csi30 recovers 0.291->0.326 under pooling. Output: `ckpt/figures/pooled_csi/pooled_csi.md`.
- Decision: keep ab3b purely as an ablation/negative-result row (even a physics-grounded loss damages extremes), and pursue extreme-rain recovery via decoder architecture, not loss reweighting.

### Literature anchor (ICLR 2026)

- exPreCast (arXiv:2602.05204) shows extreme-precip skill is recovered by a Cubic Dual Upsampling decoder (low-freq interpolation branch + high-freq pixel-shuffle residual branch), zero loss reweighting, orthogonal to losses. Our current decoder is single-branch PixelShuffle (`model/decoder.py::_UpBlock`).

### Next experiment (approved, autonomous: <24 GPU-h, no holdout tuning)

- CDU-style dual-branch decoder, base = ab3 ERA5 config. Expected csi30 +0.02~0.04 without losing light/moderate skill. Cost ~8 GPU-h (same as ab3). Risk low; literature-validated and loss-orthogonal. Status 2026-06-14: launched on GPU0 (`configs/ablation_3_cdu_p7d.yaml`, init from ab3 best), ~1220s/epoch, 60 epochs (~18h wall). Model +0.25M params (6.66M->6.91M). Smoke test passed.

### Why we are NOT running balanced importance sampling (decided 2026-06-14)

Considered as a parallel experiment, then rejected after reading the loss code, not after spending GPU:

- The intensity-stratified loss already applies per-pixel weighting by intensity band: `scripts/train.py::_intensity_loss` does `per_pix_w = band_sample_weights[band_idx]; reg = (sq * per_pix_w).mean()` with weights `[0.1, 1, 2, 4, 8]`, i.e. the 30mm band is already up-weighted 8x at the pixel level. Window-level importance sampling would re-weight the same objective, so it is largely redundant with what is already in place.
- The csi30 trade-off is architecture/loss behavior, not a data-scarcity artifact: under the *same* 8x extreme weighting, ab1/ab2/ab3 give different csi30 (0.308/0.283/0.291) and ab3b's budget loss collapses csi30 to 0.210 (pooled-CSI proves smoothing, not displacement). If the trade-off were caused by too few extreme samples, different architectures/losses on the same data would not diverge this much.
- exPreCast itself solves the trade-off with the decoder (CDU), not by claiming sampling fixes it; its balanced dataset is a dataset contribution, not evidence that re-sampling closes the extreme gap.
- Decision: do not burn a GPU-night on balanced sampling on an unverified premise. Revisit only if CDU fails AND a cheap val-only audit shows the train set genuinely lacks strong-echo windows.

- Follow-up if CDU works: intensity-residual decoupled head (NowcastNet-style) — but only after CDU results, since it may overlap CDU's function; decide based on whether CDU already recovers csi30.

### Paper narrative (3 acts)

1. Stepwise multimodal physical fusion (radar -> +PWV -> +ERA5) monotonically improves csi1/csi5/csi10/fss/mae; csi30 has a small -0.018 dip at ab2 and is recovered by ERA5 at ab3 (0.291, vs ab1 0.308) — honest framing, not "monotonic everywhere".
2. Honest extreme trade-off diagnosis: water-budget loss looks elegant but smooths away strong cores; pooled-CSI proves disappearance not displacement (the differentiating, credibility-building negative result).
3. Architectural remedy: CDU dual-branch decoder recovers extremes without loss reweighting (CDU training in progress at 2026-06-14, GPU0).

### 2026-06-14 figure/asset inventory (post-decision deliverables)

- `docs/paper/npj_narrative_outline.md` — full 4-section outline + abstract draft + figure list + risk table (the authoritative narrative skeleton for the submission).
- `ckpt/figures/paper_per_lead/` — 6 PNGs, ab1/ab2/ab3 per-lead CSI@{1,10,30} across event_test + test_robust (Fig 2 candidate).
- `ckpt/figures/pooled_csi/pooled_csi_diag_{event_test,test_robust}.png` + .pdf — 3-panel pooled-CSI diagnostic, csi30/csi10/csi1 vs pool window with story annotations (Fig 3).
- `ckpt/figures/paper_case/event_20230730T0900_final.png` — case study at the window with the largest ab3-vs-ab3b csi30 gap in event_test (+0.135); 30 dBZ magenta contour overlay makes the smoothed cores visible in the main rows, not only in the diff row (Fig 4).
- `ckpt/figures/paper_skill_bar/skill_ladder.{png,pdf}` — 4-model x 3-threshold x 2-split CSI bar chart with the -28% csi30 collapse marked in red (Table 1 / supporting figure).
- `eval/case_study_search.csv` — ranked list of all 96 event_test windows by ab3-ab3b csi30 advantage (reusable for picking alternative cases).
- `scripts/find_case_study_window.py`, `scripts/plot_pooled_csi_diagnostic.py`, `scripts/plot_skill_ladder.py` — new helper scripts, all read-only.

### 2026-06-14 honest data caveats (to surface in the paper, not bury)

- **CRPS == MAE**: every metric.json shows `crps == mae` because `_crps_marginal` is documented as "degenerate single-member CRPS == MAE" in `scripts/train.py:393-395`. Our outputs are deterministic single-member, so this is mathematically correct, not a bug. The paper will report MAE only and not rename it CRPS, to avoid misleading reviewers about probabilistic skill.
- **csi30 is NOT monotonic across ab1->ab2->ab3**: ab1=0.308, ab2(xcoreA)=0.290, ab3=0.291. Earlier narrative drafts said "monotonic"; corrected to "monotonic on csi1/csi5/csi10/fss/mae; csi30 dips by 0.018 at ab2 and recovers at ab3". This honest framing actually strengthens the paper because it foreshadows the ab3b extreme collapse.

## Active Goal State

- Goal: execute the Pluvian autonomous research plan in `docs/goals/pluvian_autonomous_goal.md`.
- Guardrails: do not tune on holdout splits; do not delete or overwrite experiment artifacts; ask before changing splits, spending more than roughly 24 GPU-hours on a new experiment, adding large external data/dependencies, or changing the paper's main claim.
- Remote root: `/data4/WuMingrui/TianJinyu/npj/pluvian`.
- Remote Python: `/home/WuMingrui/miniconda3/envs/npj/bin/python`.

## Current Remote Run

### ab3 ERA5, `configs/ablation_3_era5_p7c.yaml`

- Status at remote time 2026-06-02 22:22 CST: still training.
- Process: `scripts/train.py --config configs/ablation_3_era5_p7c.yaml --init-from ckpt/ablation_2_p7d/best.pt`.
- Runtime: about 5.3 h at the last poll.
- GPU: GPU4, about 27 GB memory, 100% utilization.
- Progress: epoch 15/60, around iteration 1280/1664 in the latest tail.
- Completion watcher: active as `logs/watch_ab3_eval.sh`, polling every 600 s.
- Watcher behavior after training exits:
  - `scripts/eval.py --split event_test` to `ckpt/eval/ablation_3_era5_p7c/event_test`.
  - `scripts/eval.py --split test_robust` to `ckpt/eval/ablation_3_era5_p7c/test_robust`.
  - `scripts/eval_per_lead.py --split event_test` to `ckpt/figures/ab3_per_lead`.
  - `scripts/eval_per_lead.py --split test_robust` to `ckpt/figures/ab3_per_lead`.
- No ab3 holdout metrics existed at the last poll.
- Recent validation is noisy and should not be used for route decisions. The strongest observed validation signal so far includes `csi_10mm=0.5518` and `csi_30mm=0.3592`, but model selection remains governed by clean holdout evaluation after training completes.

## Established ab1 vs ab2 Per-Lead Diagnosis

Source files:

- `ckpt/figures/per_lead/per_lead_event_test.json`
- `ckpt/figures/per_lead/per_lead_test_robust.json`

Summary of ab2 minus ab1 CSI deltas:

| split | metric | mean delta | +18..+42 min delta | positive leads |
|---|---:|---:|---:|---:|
| event_test | csi1 | +0.0329 | +0.0278 | 18/18 |
| event_test | csi5 | +0.0239 | +0.0196 | 17/18 |
| event_test | csi10 | +0.0214 | +0.0159 | 17/18 |
| event_test | csi30 | -0.0259 | -0.0384 | 1/18 |
| test_robust | csi1 | +0.0059 | -0.0007 | 12/18 |
| test_robust | csi5 | +0.0064 | +0.0106 | 17/18 |
| test_robust | csi10 | +0.0092 | +0.0152 | 18/18 |
| test_robust | csi30 | +0.0017 | +0.0014 | 12/18 |

Interpretation: the +PWV regression is localized. On `event_test`, light/moderate rain gains are broad and robust across lead times, while the 30 mm threshold regresses systematically, especially at +18..+42 min. On `test_robust`, 30 mm is essentially flat to slightly positive.

## xcore Loss-Reweighting Runs

Remote result files:

- `ckpt/eval/ablation_2_p7d_xcoreA/event_test/metric.json`
- `ckpt/eval/ablation_2_p7d_xcoreA/test_robust/metric.json`
- `ckpt/eval/ablation_2_p7d_xcoreB/event_test/metric.json`
- `ckpt/eval/ablation_2_p7d_xcoreB/test_robust/metric.json`
- `ckpt/figures/xcore_per_lead/xcoreA/per_lead_event_test.json`
- `ckpt/figures/xcore_per_lead/xcoreB/per_lead_event_test.json`
- `ckpt/figures/xcore_per_lead/xcoreA/per_lead_test_robust.json`
- `ckpt/figures/xcore_per_lead/xcoreB/per_lead_test_robust.json`

Aggregate metrics:

| model | split | csi1 | csi5 | csi10 | csi30 | crps |
|---|---|---:|---:|---:|---:|---:|
| xcoreA | event_test | 0.4268 | 0.5275 | 0.5797 | 0.2900 | 4.9745 |
| xcoreB | event_test | 0.4150 | 0.5118 | 0.5614 | 0.3027 | 5.2609 |
| xcoreA | test_robust | 0.3246 | 0.4339 | 0.4933 | 0.2764 | 3.0474 |
| xcoreB | test_robust | 0.3144 | 0.4189 | 0.4776 | 0.2844 | 3.2233 |

Per-lead csi30 deltas against ab1:

| model | split | mean delta | +18..+42 min delta | min delta | max delta |
|---|---|---:|---:|---:|---:|
| xcoreA | event_test | -0.0176 | -0.0267 | -0.0470 | +0.0151 |
| xcoreB | event_test | -0.0032 | -0.0099 | -0.0284 | +0.0258 |
| xcoreA | test_robust | +0.0064 | +0.0102 | -0.0014 | +0.0149 |
| xcoreB | test_robust | +0.0159 | +0.0181 | +0.0084 | +0.0259 |

Decision:

- xcoreB proves loss reweighting can recover most of the extreme-event csi30 gap, but it pays too much in light/moderate CSI and CRPS.
- xcoreA is the balanced fallback candidate, but it does not fully solve the event short-lead csi30 regression.
- Do not start another xcore sweep until ab3 ERA5 finishes. If ab3 fails and the project needs one more cheap fix, the next xcore should be lead-targeted for +18..+42 min rather than globally more aggressive.

## Drop-PWV Counterfactual Diagnostic

Remote/local result files:

- `ckpt/eval/ablation_2_p7d_droppwv/event_test/metric.json`
- `ckpt/eval/ablation_2_p7d_droppwv/test_robust/metric.json`
- `ckpt/figures/p7d/cmp_failure_2023-07-29T06.png`

Aggregate metrics for ab2 evaluated with PWV zeroed:

| split | csi1 | csi5 | csi10 | csi30 | crps |
|---|---:|---:|---:|---:|---:|
| event_test | 0.4258 | 0.5396 | 0.6004 | 0.1262 | 4.9815 |
| test_robust | 0.3269 | 0.4441 | 0.5092 | 0.1518 | 2.9683 |

Interpretation: this is not a radar-only baseline, because the ab2 model was
trained with PWV and is being ablated at inference. It is still a useful channel
influence diagnostic: csi30 collapses when PWV is removed, showing the strong-core
head materially uses PWV information. Combined with the per-lead regression, this
supports a targeted lead-aware PWV influence test if ab3/ab3b does not solve the
short-lead csi30 issue.

## Sharpness / Generative Diagnosis

Source file:

- `ckpt/figures/sharp_all/sharpness_all.json`

Aggregate sharpness metrics over 8 strong-echo events:

| method | high-k target 1.0 | p99.9 dBZ | interpretation |
|---|---:|---:|---|
| baseline | 0.071 | 39.6 | blurred deterministic floor |
| sharpB | 0.420 | 38.3 | adds texture but does not restore extreme cores |
| GAN-coreA | 0.686 | 43.7 | best currently valid sharpening route |
| hi-core | 0.589 | 40.0 | second-best valid route |
| diffusion residual | 2.711 | 100.8 | invalid overrun/hallucination |

Decision: keep GAN-coreA and deterministic CSI/FSS claims in separate narrative lanes. Do not claim diffusion or GAN improves deterministic accuracy without CSI/FSS/CRPS evidence.

## Current Route Ranking

1. Finish ab3 ERA5 and evaluate it cleanly. This is the current highest-ROI path because it can support a physically grounded npj story.
2. If ab3 is promising, prioritize Track C water-vapor budget/PDE loss after checking ERA5 u/v/q availability and alignment.
3. If ab3 does not fix the event csi30 short-lead gap, consider one cheap validation-only lead-targeted xcoreC or a lightweight lead-aware gate. Do not start a broad architecture redesign yet.
4. Keep Track B generative ensemble after the deterministic/physics route is stable; use CRPS/spread/reliability for ensemble claims and spectral metrics for sharpness claims.

## Track C Physics Readiness Audit

Local ERA5 files under `era5/` cover variables `u`, `v`, `q`, and `t` for all pressure levels `[250, 300, 400, 500, 600, 700, 850, 925]` and all months `202305` through `202309`.

Code readiness:

- `model/losses/water_budget.py` defines `WaterBudgetLoss` with differentiable spherical divergence and PWV time derivative.
- `scripts/train.py` constructs the loss when `loss.budget.enabled=true`.
- `scripts/train.py` uses future ERA5 `u/v/q` at `budget.level_idx` and logs validation budget loss when enabled.
- `pipeline/era5_io.py` now exposes `load_era5_window_with_grid`, and `pipeline/data_loader.py` emits `era5_lat` / `era5_lon` so the budget loss uses the same native ERA5 grid as the fields.

### Track C budget alignment fix and smoke

Problem found before launching ab3b: `WaterBudgetLoss` requires `pwv_pred`, `rain_pred`, `u/v/q`, `lat`, and `lon` on one grid. The ERA5 loader intentionally keeps ERA5 on its native grid for performance, but the budget branch in `scripts/train.py` was still passing radar-grid `RADAR_LAT/RADAR_LON` and full-resolution model predictions. That would either shape-mismatch or make the PDE residual physically inconsistent.

Fix implemented:

- Added `load_era5_window_with_grid` to return native ERA5 `lat/lon` alongside `u/v/q/t` cubes.
- Added Dataset sample keys `era5_lat` and `era5_lon`.
- Added `scripts/train.py::_budget_loss_inputs` to downsample `pwv_pred` and `rain_pred` to the ERA5 grid and pass aligned `pwv_budget`, `rain_budget`, `u_budget`, `v_budget`, `q_budget`, `lat_budget`, and `lon_budget` into `WaterBudgetLoss` in both train and validation.
- Added a static guard test: `tests/test_budget_loss_alignment_static.py`.

Validation:

- Local static tests: `pytest -q tests/test_budget_loss_alignment_static.py` -> 3 passed.
- Local eval static regressions: `pytest -q tests/test_eval_script_static_regressions.py` -> 4 passed.
- Local syntax check: `python3 -m py_compile pipeline/era5_io.py pipeline/data_loader.py scripts/train.py model/losses/water_budget.py` -> passed.
- Remote static test with conda Python -> 3 passed.
- Remote GPU smoke on GPU0: `configs/debug_ablation_3b_era5_budget_p7c_smoke.yaml`, initialized from `ckpt/ablation_3_era5_p7c/best.pt`, `--debug --max-epochs 1`. Result: train debug completed, validation completed, `val/budget=0.0012`, no grid/shape crash. Output directory: `ckpt/debug_ablation_3b_era5_budget_p7c_smoke`.

Decision: the budget-loss path is now ready for a real ab3b run, but do not start the full run from an in-progress ab3 checkpoint. Wait for ab3 to finish and use its clean holdout result to decide whether ab3b is justified as the next Track C experiment.

Configuration caution:

- `configs/ablation_3_era5_p7c.yaml` is a clean ERA5-context run with `loss.budget.enabled=false`.
- `configs/ablation_5_budget_loss.yaml` enables budget loss but also has `mfd_channel_enabled=true`, so it is not a clean single-delta budget-loss experiment relative to ab3.
- If ab3 is promising, the next clean Track C run should use `configs/ablation_3b_era5_budget_p7c.yaml`, which only adds the budget loss to the ab3 ERA5 setup. It should not turn on MFD unless the explicit goal is a combined/full Pluvian model.

## Next Low-Risk Tasks While ab3 Trains

- Prepare plotting scripts/notebooks for ab3 vs ab1/ab2/xcore per-lead CSI and summary tables.
- Audit ERA5 u/v/q file availability and temporal alignment for Track C without launching a new training run.
- Identify case-study timestamps already used in sharpness/event figures so ab3 visualizations can reuse the same scenes.
- Draft a figure inventory for paper materials: deterministic skill table, per-lead CSI, FSS, CRPS/spread, physical-consistency diagnostics, spectral/sharpness, and failure cases.

## Result Summary Utility

- Added `scripts/summarize_phase7d_results.py`, a read-only Markdown summarizer for existing `ckpt/eval/*/metric.json` and `ckpt/figures/*per_lead*.json` files.
- Added `tests/test_phase7d_summary.py`; local and remote tests pass.
- Current generated report: `docs/experiments/pluvian_route_summary.md`.
- Refresh command after ab3 watcher finishes: `python scripts/summarize_phase7d_results.py --root . --out docs/experiments/pluvian_route_summary.md`.

## Bootstrap Confidence Intervals

Files:

- Script: `scripts/eval_bootstrap_ci.py`
- Tests: `tests/test_bootstrap_ci.py`
- Remote/local outputs: `ckpt/bootstrap_ci/phase7d_skill/*_{rows,ci}.json`
- Markdown report: `docs/experiments/pluvian_bootstrap_ci.md`
- Paired delta report: `docs/experiments/pluvian_bootstrap_delta_ci.md`
- Paper-ready tables: `docs/experiments/pluvian_paper_tables.md` and
  `docs/experiments/pluvian_paper_tables.tex`
- Paired delta CI forest plots:
  `ckpt/figures/bootstrap_delta_ci/paired_delta_ci_event_test.{png,pdf}` and
  `ckpt/figures/bootstrap_delta_ci/paired_delta_ci_test_robust.{png,pdf}`

Method: evaluate checkpoints window-by-window, aggregate CSI from raw contingency
counts and FSS from full-set numerator/denominator components, then bootstrap by
manifest date. This is intended for uncertainty reporting, not model selection.

Key readout:

- event_test has only 4 dates, so day-bootstrap intervals are wide. Claims should
  be phrased as robust directional/diagnostic evidence rather than exaggerated
  statistical certainty.
- ab2 improves event_test csi1/csi10/FSS/CRPS directionally over ab1, but its
  csi30 interval remains below ab1's point estimate, consistent with the csi30
  caveat.
- xcoreB recovers event_test csi30 point estimate close to ab1 but loses csi1,
  csi10, FSS, and CRPS, confirming the tradeoff found in aggregate metrics.
- Paired delta bootstrap gives the cleanest paper wording: ab2-ab1 on event_test
  is positive for csi1 (+0.0347), csi10 (+0.0255), FSS3 (+0.0350), and CRPS
  (-0.3601), while csi30 is negative (-0.0252). xcoreB-ab2 improves csi30 point
  delta (+0.0201) but worsens csi1 (-0.0166), csi10 (-0.0293), FSS3 (-0.0167),
  and CRPS (+0.4335).

Refresh command for established p7d skill models:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/eval_bootstrap_ci.py \
  --ckpt ab1_p7d=ckpt/ablation_1_p7d/best.pt \
  --ckpt ab2_p7d=ckpt/ablation_2_p7d/best.pt \
  --ckpt xcoreA=ckpt/ablation_2_p7d_xcoreA/best.pt \
  --ckpt xcoreB=ckpt/ablation_2_p7d_xcoreB/best.pt \
  --split event_test --split test_robust \
  --out_dir ckpt/bootstrap_ci/phase7d_skill \
  --n_boot 1000 --batch_size 1
```

Refresh paired delta report after rows exist:

```bash
python scripts/bootstrap_delta_ci.py \
  --rows_dir ckpt/bootstrap_ci/phase7d_skill \
  --out docs/experiments/pluvian_bootstrap_delta_ci.md \
  --split event_test --split test_robust \
  --n_boot 1000 --seed 0
```

Generate paper-ready Markdown/LaTeX tables:

```bash
python scripts/make_phase7d_paper_tables.py \
  --rows_dir ckpt/bootstrap_ci/phase7d_skill \
  --delta_json docs/experiments/pluvian_bootstrap_delta_ci.json \
  --out_md docs/experiments/pluvian_paper_tables.md \
  --out_tex docs/experiments/pluvian_paper_tables.tex
```

Generate paired delta CI forest plots:

```bash
python scripts/plot_bootstrap_delta_ci.py \
  --delta_json docs/experiments/pluvian_bootstrap_delta_ci.json \
  --out_dir ckpt/figures/bootstrap_delta_ci
```

**ab2 re-launched (2026-06-17 17:27 UTC, user GO):** Re-running with the correct main-line config `ablation_2_p7d.yaml` (60 epoch from-scratch, lr=3e-4, standard intensity-band weights). PID 2491002 on GPU0. Confirmed at ep0 it40 (loss 19.8 decreasing, GPU0 100% util, 28GB). ETA ~17h ⇒ completion 2026-06-18 ~10:30 UTC. The original `ablation_2_p7d_xcoreA.yaml` ckpt is kept on disk in case it's useful as a separate xcoreA-fine-tune ablation row.

**Five-run status snapshot (2026-06-17 17:30 UTC):**

| Model | Run state | Last avg loss | Latest val csi_1mm / csi_30mm | Notes |
|---|---|---|---|---|
| ab1     | ✅ done 60 ep | 21.97 | 0.291 / 0.104 | clean baseline |
| ab2     | 🔄 RE-RUN ep0 (just relaunched) | — | — | xcoreA bug; ab2_p7d now in flight |
| ab3     | ✅ done 60 ep | 21.80 | 0.316 / 0.102 | ERA5 fusion, slightly better than ab1 on csi_1mm |
| ab3b    | 🔄 ep58 it1420/1664 | 21.67 | 0.306 / 0.101 | budget loss active (bud=0.0005), almost done |
| ab3-cdu | 🔄 ep49 it680/1664 | 22.06 | 0.298 / 0.098 | from-scratch this time, slowest |


## 2026-06-18 Phase 7e — 5 retrain runs all complete; eval pipeline launched

All five mm/h-era retrains finished:
  - ab1     done 60 ep (2026-06-17 17:02 UTC)
  - ab3     done 60 ep (2026-06-17 17:18 UTC)
  - ab3b    done 60 ep (2026-06-17 17:52 UTC)
  - ab3-cdu done 60 ep (2026-06-17 21:37 UTC, from-scratch this time, addresses reviewer C6)
  - ab2     done 60 ep (2026-06-18 12:28 UTC, re-run with `ablation_2_p7d.yaml` after the xcoreA config bug)

**Sanity eval (ab1 / event_test, mm/h)**:
  MAE=0.757, CSI@1=0.386, CSI@5=0.203, CSI@10=0.096, CSI@30=0.039, FSS@3=0.583, FSS@11=0.615

vs old dBZ-era ab1 (MAE 5.19, CSI@1 0.397, CSI@10 0.565, CSI@30 0.308):
  - MAE collapses 6.9× because mm/h scale is ~6× smaller than dBZ
  - CSI@1 nearly identical (0.386 vs 0.397) — light-rain detection is unit-invariant
  - CSI@10 drops 6× because mm/h-10 is now a real moderate threshold (0.8% pixels) instead of dBZ-10 (24% pixels)
  - CSI@30 drops 8× because mm/h-30 is now a real extreme threshold (0.04% pixels) instead of dBZ-30 (6.4% pixels)

**The unit fix worked as expected** — CSI@30 now genuinely measures extreme-precipitation skill. The qualitative shape of the ab1→ab3 cascade and the ab3b vs ab3 trade-off should not invert under the unit change but the magnitudes will look very different in the new tables.

**Post-training eval pipeline launched (2026-06-18 19:32 UTC, PID 3356216):** running 10 holdout evals (5 ckpts × 2 splits) sequentially on GPU0. ETA ~45 min. After that we still need bootstrap CIs (~5 h serial), pooled-CSI, per-lead, paired delta CIs, and figure regeneration. None of these touch the trained models — they are read-only inference.


## 2026-06-18 Phase 7e — paired delta CIs change the paper thesis

Paired bootstrap delta CIs (n=2000, seed=42, day-stratified) for the
5 mm/h-era ckpts are now in `docs/experiments/pluvian_paired_delta_phase7e_*.json`.

**event_test — the new ground truth**:

| Comparison | CSI@1 | CSI@10 | CSI@30 | MAE |
|---|---|---|---|---|
| ab2 - ab1 | +0.005 ★ | +0.002 | +0.005 ★ | -0.021 ★ |
| ab3 - ab1 | **-0.023 ★** | **-0.010 ★** | -0.002 | **+0.085 ★** |
| ab3 - ab2 | **-0.028 ★** | **-0.012 ★** | **-0.007 ★** | **+0.106 ★** |
| **ab3b - ab3** | **+0.016 ★** | **+0.017 ★** | **+0.013 ★** | **-0.034 ★** |
| ab3-cdu - ab3 | +0.005 ★ | -0.002 | +0.001 | -0.029 |

★ = paired 95% CI strictly excludes zero.

**The dBZ-era thesis is dead**:

1. **ERA5 fusion (ab3) systematically HURTS performance on event_test**.
   The CSI@1 / CSI@10 / MAE deltas vs ab1 are all significant in the
   negative direction. This is the exact opposite of the dBZ-era
   claim that "ab1→ab2→ab3 monotonically improves".
2. **The water-budget PDE loss (ab3b) is now a UNIFORM IMPROVEMENT** over
   ab3, including extreme CSI@30 (+0.013 ★). There is no "collapse";
   there is no trade-off. The §3.2 negative-result narrative of the
   prior paper draft is invalidated.
3. **The pooled-CSI / smoothing-vs-displacement motivation evaporates**
   because there is no collapse to diagnose. Pooled-CSI is still
   methodologically useful as a verification tool but it no longer
   anchors the paper.
4. **CDU is a small positive over ab3** on event_test (CSI@1 +0.005 ★),
   but ab3 itself is the weakest baseline now, so the "architectural
   sidestep of the trade-off" framing also collapses.

**test_robust**: most deltas are non-significant; ab2 mildly beats ab1
on CSI@1 (+0.022 ★) and MAE (-0.052 ★). ab3-cdu has a small but
significant CSI@30 advantage over ab3 (+0.008 ★).

### Honest open questions

The reversal is so total that we cannot rule out training-side artefacts.
Plausible alternative explanations awaiting user decision before any
paper rewrite:

  (a) **Under-convergence at 60 ep under mm/h**: the cosine LR was tuned
      on dBZ-scale loss. mm/h target distribution has ~6× smaller scale
      and a 150× sparser extreme tail (0.04% pixels >30 vs 6.4% pixels
      >30 dBZ). The model may need more epochs or a different schedule.
  (b) **best-ckpt selection by val/csi_10mm**: mm/h csi_10mm is in the
      0.08-0.12 range (very noisy), so the best-ckpt picker may be
      latching onto a noisy epoch rather than a real optimum.
  (c) **ERA5 fusion gate learned weights**: GatedAsymmFusion was tuned
      with dBZ-scale radar; under mm/h the gate may not have learned
      sensible mixing weights in 60 ep.
  (d) **Real effect**: the paper thesis was wrong and the prior
      "monotonic fusion" / "extreme collapse" findings were artefacts
      of the dBZ-vs-mm/h inconsistency. This is the most uncomfortable
      possibility but also the most honest.

### Cron decision (no auto-action on paper)

The cron autopilot has produced the data and the analysis but **will
not rewrite the paper**. This is a Phase-3-level narrative shift; the
user must decide:
  - Option 1: accept and reframe (write a different paper).
  - Option 2: extend training (more epochs / longer schedule / different
    best-ckpt rule) to test under-convergence hypothesis.
  - Option 3: investigate the ERA5 fusion gate behaviour under mm/h
    (look at learned gate weights vs dBZ-era).

