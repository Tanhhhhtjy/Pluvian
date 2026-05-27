# Phase 7b Red-Team Audit — `phase7b/integrated`

> Date: 2026-05-27
> Branch: `phase7b/integrated` (HEAD 900e024 — after U-Net skip + intensity head + gated fusion merge)
> Auditor: `red-team-7b` (team `pluvian-phase7a`)
> Scope: only NEW risks introduced by the three-way merge. Phase 7a items (best.pt, val=23·7, MC autocast, etc.) not re-listed unless regressed.

---

## TL;DR

**Must-fix (do NOT launch training until these are resolved): 4**
**Should-fix: 4**
**Nice-to-have: 3**

---

## Must-fix (block training)

### M1. `gated_fusion` is hard-on for ab1 and ab3 — silent architecture swap

`model/pluvian.py:65` defaults `gated_fusion=True`. `scripts/train.py:727-743` constructs `Pluvian(...)` and **never passes `gated_fusion` from cfg**. Configs `_base.yaml`, `ablation_1`, `ablation_2`, `ablation_3`, `ablation_7b_intensity` all have **no `gated_fusion` key** (verified empirically — see repro). Effect:

- **ab1** (`pwv_enabled=False, pwv_concat_only=False`): `self.gated_fusion = True and not False = True` → uses `GatedAsymmFusion` with PWV tokens zeroed. The `_PwvRefiner` (2 spatial + 1 temporal self-attn) runs on zero inputs every step; per-pixel gates start at sigmoid(-2)≈0.12 and inject station deltas back into the radar feature. **ab1 is no longer a clean radar-only baseline** — comparing the next ab1-new run to the historical ab1 (`CSI@10mm=0.596`) measures both the architecture change *and* the new training fixes.
- **ab3**: same path → ab3 silently becomes the asymmetric-gated variant rather than the legacy `CrossAttentionFusion`. The whole Phase 7 roadmap framed gated fusion as a *7b* upgrade vs the *7a* cross-attn baseline; that A/B is now confounded.
- ab2 is safe: `pwv_concat_only=True` forces `self.gated_fusion = False` (`pluvian.py:86`).

Repro:
```
python -c "from scripts.train import load_config; from pathlib import Path; \
  print([('gated_fusion' in load_config(Path('configs')/c)['model']) \
         for c in ['ablation_1_radar_only.yaml','ablation_3_pwv_channel.yaml']])"
# -> [False, False]
```
Then `grep gated_fusion scripts/train.py` → no hits. Fix: thread `gated_fusion: bool = False` through `_base.yaml` and pass `gated_fusion=m["gated_fusion"]` at `train.py:743`. Keep `True` only in `ablation_7b_intensity.yaml` and the planned `ab1-new`.

### M2. Old ab1/ab2/ab3 ckpts cannot resume on this branch

`scripts/train.py:766` calls `model.load_state_dict(state["model"])` with **default `strict=True`**. The new model has new parameters:

- `decoder.lead_attn.queries`, `decoder.lead_attn.attn.*`, `decoder.lead_attn.q_norm.*`, `decoder.lead_attn.kv_norm.*`
- `decoder.skip{1,2}_time.weight`, `decoder.skip{1,2}_proj.*`, `decoder.fuse_stage{1,2}.*`
- For ab3 specifically: all `GatedAsymmFusion` keys (`fusion.pwv_refiner.*`, `fusion.pwv_cross.*`, `fusion.sta_cross.*`, `fusion.gate_*`) replace the legacy `fusion.layers.*`.
- Old keys (`decoder.time_proj.weight (18,12)`) are gone.

Resume from `ckpt/ablation_*/last.pt` of the historical runs will raise `RuntimeError: Missing key(s)/Unexpected key(s)`. Watchdog's auto-resume after a mid-epoch crash on a fresh 7b run is OK (same arch), but operationally — if anyone reruns ab2 from disk to extract `best.pt` for paper figures, it crashes. Fix: either pin `strict=False` with an explicit warn-list, or document "ab1-new / ab7b ckpts only".

### M3. `intensity_stratified=True` silently bypasses the configured data loss

`scripts/train.py:481-491`: when `intensity_on and "rain_logits" in out`, the code calls `_intensity_loss(...)` and assigns to `loss_data`. The configured `cfg["loss"]["data"]["type"]` (which can be `weighted_mse | bmae | focal_rain | tweedie`) is **never evaluated** in this branch — `losses["data"]` is built (`train.py:200-216`) and untouched. Implications:

- If the user sets `loss.data.type: tweedie` *and* `model.intensity_stratified: true`, Tweedie is silently dropped. Watchdog won't notice. Train log shows `data=…` which is actually `ce_w*ce + reg_w*reg`.
- `running["data"]` accumulator and `[ep…] avg data=` printout still claim to be "data loss" — same label, different quantity. Comparing data-loss curves across ablations now mixes apples/oranges.

Worse: at val time (`train.py:619`) `agg["data"].append(losses["data"](rain_pred, rain_tgt_f).item())` runs the *configured* (e.g. WeightedMSE) loss on the **expectation** of the band softmax. So `train/data` and `val/data` are different functions when intensity is on. `val/csi_*` is unaffected (uses raw `rain_pred`). Fix: either explicitly fail in `build_losses` when both are set, or log `train/data_ce` + `train/data_reg` separately.

### M4. Manifest still emits 24 starts/day (Phase 7a item #2 not regressed but still load-bearing for cost)

Confirmed `pipeline/manifest.csv` train rows still carry `starts_per_day=24` per Phase 7a expansion. With the new architecture: extra `_LeadTimeCrossAttn` (per-spatial cross-attn over T_in×T_out queries, attn batch dim `B·H·W = 2·83·88 = 14608`) + skip-fuse 3×3 convs at H/4 = 166×176 + 2× extra encoder skip tensors materialized in memory. Phase 5 baseline of 7 min/epoch will not hold. **Required: 50-iter smoke (`--debug` extended to ~50) on AutoDL before the full 60-epoch launch.** User's `feedback_pretraining_speed_check.md` rule explicitly demands this; the ¥30+ figure assumes ~7 min/epoch and will not survive a 2× slowdown.

---

## Should-fix

### S1. `mc_dropout_predict` is *not* fully cached — encoder still runs 2× per val batch

`scripts/train.py:598-605`: `out = model(model_in)` runs the full encoder+fusion+decoder once for the deterministic forward; then `model.mc_dropout_predict(...)` calls `self._encode_and_fuse(...)` *again* (`pluvian.py:274`) before looping the K decoder passes. Audit #11's "cache fused" was implemented for the inner loop but the *outer* val loop still computes the encoder twice. With the new heavier model this costs another ~30-50% of val time. Fix: in val, call `_encode_and_fuse` once, hand `(fused, skips)` to both branches.

### S2. CRPS spread is artificially compressed when `intensity_stratified=True`

MC-dropout perturbation happens at `decoder.py:188` (`F.dropout(x, ...)`) *before* `rain_head` and softmax. The softmax expectation `Σ probs * band_centers` is a smooth function of `x`, so dropout-perturbed samples ride on the same fixed band lattice. Empirically the per-pixel std (recorded as `val/spread`) will be smaller than for the continuous head, biasing CRPS comparisons against the intensity model. Not a bug per se — but flag in the metric_log so we don't read "CRPS dropped 30%" as a win when the cause is head topology, not skill. Fix: add a `val/crps_method` string field, or run a one-time sanity print of `spread/mean(rain_pred)` for both heads on the same val batch.

### S3. `decoder.softmax` and `_intensity_loss` cross-entropy under bf16 autocast

`decoder.py:222`: `F.softmax(logits, dim=2)` on `(B, T, K=5, H, W)` runs in bf16 under autocast. logit range is unbounded; for an over-confident pixel `exp(20)≈4.85e8` is fine in bf16 (max ~3.39e38), so no overflow. **But** `(probs * centers).sum(dim=2)` where `centers[-1]=50.0` and bf16 precision is ~3 decimal digits → expected rain quantization noise ≈ 50 * 2^-7 ≈ 0.4 mm at the high-intensity band. Comparable to the CSI@30mm decision boundary error budget; recommend an `.float()` cast on `probs * centers` (`decoder.py:223-224`) so the expectation lives in fp32. `F.cross_entropy` upcasts internally, so CE itself is OK.

### S4. `_oom_retry` still dead (Phase 7a #8 still standing)

`scripts/train.py:429` defined; zero call sites (`grep _oom_retry scripts/train.py` returns only the def). Remove or wire into the `train_one_epoch` forward call. Carrying it is misleading — the Phase 7 roadmap claims it's protecting OOM.

---

## Nice-to-have

### N1. `_collate` keeps `pwv_coords`/`station_coords` as tensors but no shape check

If a future manifest row has a different station count, batch stacking silently fails. Add a one-time assert in `_collate` against the first batch's shape. Low-pri.

### N2. `running["data"]` mixes scales across loss types

When swapping `weighted_mse → tweedie → intensity` between ablations, the `[ep…] avg data=` printout is on a different scale each run. Hard to spot a regression. Add a `loss_type` tag to log header.

### N3. `band_centers` default vs `band_edges` default off-by-one risk

`_base.yaml` lists 5 centers `[0.0, 0.5, 4.5, 19.0, 50.0]` and 4 edges `[0.1, 1.0, 8.0, 30.0]`. `_intensity_loss:259` checks `len(sample_w) == K` but **does not** validate `len(band_edges) == K-1`. If a future config lists 5 edges by mistake, `torch.bucketize` returns indices 0..5 → out-of-range index into `w_t` (size 5) → CUDA assert. Add a runtime check.

---

## Pre-relaunch checklist (in order)

1. **Pin `gated_fusion` per config**. Add `gated_fusion: false` to `_base.yaml`; override `true` only in `ablation_7b_intensity.yaml` and the new `ab1-new` config. Pass through at `train.py:743`. Verify with `python -c "from scripts.train import load_config; from pathlib import Path; print(load_config(Path('configs/ablation_1_radar_only.yaml'))['model'].get('gated_fusion'))"` → `False`. **Without this, M1 ruins the headline ablation A/B.**
2. **Mark old ab1/ab2/ab3 ckpts as legacy**. Move `ckpt/ablation_{1,2,3}_*` to `ckpt/legacy_phase5/`. Do not attempt `--resume` against the new branch.
3. **Pick the data-loss policy**. For `ab1-new` (planned full 7b): explicitly choose either intensity-stratified OR Tweedie/FocalRain — not both. Document the choice in the config header comment.
4. **5-iter CPU smoke** (already done per task description) — re-run with `intensity_stratified=true` AND `gated_fusion=true` to confirm forward+backward + MC-dropout end-to-end. Time the iter.
5. **AutoDL 50-iter dry-run** with full B=2, H×W=664×704, bf16. Compare iter-time to Phase 5 ab1 (7 min/epoch ≈ 0.21 s/it at 2448 samples/grad_accum=2). >2× regression → stop and diagnose before launching 60 epoch.
6. **Watchdog cleanup**: `rm -rf ckpt/ablation_1_new* logs/ablation_1_new*` on remote before `setsid`-launching, per HANDOVER §9.4 stale-`[done]` gotcha.
7. **Confirm `cfg["loss"]["data"]["weight"]` is what you want when intensity is on** — it scales the entire `ce + reg` total (`train.py:489`).

---

## Recommended `ab1-new` config (most stable starting point)

```yaml
name: ablation_1_new_7b
inherits: _base.yaml
model:
  # Same data routing as legacy ab1: radar-only.
  pwv_enabled: false
  pwv_concat_only: false
  era5_enabled: false
  mfd_channel_enabled: false
  # Phase 7b architecture (skip decoder is unconditional; gated fusion is
  # vacuous when pwv_enabled=false but kept off to avoid the _PwvRefiner-on-
  # zeros path; see M1).
  gated_fusion: false
  intensity_stratified: true
  band_centers: [0.0, 0.5, 4.5, 19.0, 50.0]
loss:
  data:
    type: weighted_mse        # placeholder; intensity_loss takes over (M3)
    weight: 1.0
  intensity:
    ce_weight: 1.0
    reg_weight: 1.0
    band_edges: [0.1, 1.0, 8.0, 30.0]
    band_sample_weights: [0.1, 1.0, 2.0, 4.0, 8.0]
  fss:
    enabled: true
    thresholds: [1.0, 10.0, 30.0]
    threshold_weights: [1.0, 0.1, 0.1]
    window: 9
    weight: 0.1
  budget:
    enabled: false
hardware:
  precision: bf16
  grad_accum: 2
data:
  batch_size: 2
  num_workers: 2
```

Rationale: isolates **U-Net skip + intensity head** as the only architecture deltas vs legacy ab1. Holding `gated_fusion=false` (since PWV is off anyway) makes the next experiment (ab1-new + PWV + gated_fusion) a clean follow-on A/B. If M1 is not fixed first, `gated_fusion: false` here will still be ignored — that is why M1 is a hard blocker.
