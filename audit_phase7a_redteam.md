# Phase 7a Red-Team Audit — `phase7a/integrated`

> Audit date: 2026-05-27
> Branch: `phase7a/integrated` (post-merge of loss-fixes + val-expansion)
> Auditor: red-team-auditor agent (pluvian-phase7a team)

## TL;DR

**5 must-fix (stop training until these are addressed):**

1. **`best.pt` is not best — it always equals current epoch's state.** `scripts/train.py:378` writes `torch.save(state, best_link)` where `state` is the *current* epoch's model dict; the historically-best `path` is computed (`records[0][1]`) but never used. Verified empirically on ab2: `metric_log.json` says best CSI@10mm = 0.625 at epoch 24, but `best.pt` contains epoch 59 (CSI 0.619). **All Phase 5 "best.pt" eval numbers, including the +12% CSI uplift from the CPU re-eval, are confounded by this.** Repro:
   ```bash
   python -c "import torch,json; sd=torch.load('ckpt/ablation_2_pwv_concat/best.pt',weights_only=False); log=json.load(open('ckpt/ablation_2_pwv_concat/metric_log.json')); print(sd['epoch'], max(log,key=lambda e:e['val/csi_10mm'])['epoch'])"
   # prints: 59 24
   ```

2. **Silent 3× train-set explosion.** New manifest sets `starts_per_day=24` on **all** rows (`pipeline/manifest.csv`). Train set went from 102 × 8 = 816 to 102 × 24 = **2448** samples/epoch. With HANDOVER's 7 min/epoch baseline, ab4–6 will now take ~21 min/epoch → ~21 h × ¥1.5/h ≈ **¥30 → ¥90 per ablation**. No commit message announces this; the train.py speed-check rule will catch it but only after launch.

3. **Val = the 23·7 case study (data leakage).** `manifest.csv` val rows = `2023-07-29 ~ 2023-08-01` — exactly the marquee event used in `figures/case_23p7/` and `scripts/plot_prediction.py`. Reporting val CSI/FSS as "benchmark" then visualizing the same days as Figure 1 is reviewer-bait. `test_robust` (14 days, 336 starts) is built but never referenced in `train.py` / eval.

4. **MC-dropout CRPS runs outside `autocast` → 5×–10× val slowdown + bf16/fp32 inconsistency.** `scripts/train.py:548-550` only wraps the deterministic forward inside `with torch.autocast(...)`; the MC-dropout call at `train.py:571-572` (`model.mc_dropout_predict(...)`) executes in fp32. With `crps_samples=4` and 48 val batches, each val epoch does **5 forwards/batch with 4 in fp32**. Expect val time to dominate epoch budget. Also: the deterministic `rain_pred` (bf16) and the MC mean (fp32) won't agree — CSI vs CRPS are computed from different precision passes.

5. **`_intensity_weight` triple-counts the 10–30 mm/h band.** `model/losses/data_losses.py:23-24` applies `where(>10, 3.0)` then `where(>30, 5.0)` to a tensor that started at 1.0. For y=20, weight = 3 ✓. For y=40, the first `where` lifts it to 3, then the second to 5 ✓. Actually correct on second read — keep, but the implementation reads fragile; future devs may "fix" it by accident.

## Should-fix (8)

6. **Val FSS only computes threshold=1.0** (`scripts/train.py:567`) despite the multi-threshold `[1,10,30]` FSS being wired in *training* loss. The FSS@10mm and FSS@30mm advertised in the Phase 7 roadmap are not actually being measured at val time.

7. **Eval thresholds 1/5/10/30 mm vs training-loss FSS thresholds 1/10/30 mm** — keys `val/csi_5mm` exist with no FSS counterpart; minor but trips report scripts that expect aligned thresholds.

8. **`_oom_retry` defined but never invoked anywhere** (`scripts/train.py:392`, dead code). Phase 7 roadmap claims it's wired.

9. **`CkptManager` deletes ckpts using `if p != best`** — since `best` is the record path and `best.pt` is overwritten with current state, the "best" epoch file (e.g. `epoch024.pt`) is what survives, but `best.pt` next to it is *not* that file. Confusing semantics for downstream eval scripts.

10. **Tweedie loss under bf16 autocast.** `model/losses/tweedie.py:53`: `mu = softplus(pred) + eps`. With bf16, eps=1e-6 is representable, but `pow(mu, -0.5)` for `mu` near eps yields ~1000; `y * 1000 * 2` for y=50 mm → ~1e5. `2-p` and `1-p` floats are bf16-precision. Tweedie has not been smoke-tested under autocast; recommend a single-epoch dry-run on ab3 best.pt (no training) before launching ab4/5/6 with `loss.data.type: tweedie`.

11. **`mc_dropout_predict` re-runs encoders K times.** `model/pluvian.py:209-227`: each MC sample re-runs RadarEncoder + Era5Encoder + SparseTokenEncoder + Fusion (all deterministic). Only the decoder is stochastic. Should cache the `fused` tensor and only re-run decoder. ~4× val speedup attainable.

12. **Old ckpt loads will warn but not fail** — `decoder_dropout` is functional (`F.dropout`, no params), so old state dicts load cleanly. Confirmed: ab2 `best.pt` has `decoder.time_proj.weight: (18,12)` matching new `input_frames=12`. Still — add a `strict=True` assertion in eval script to fail loudly if a future ckpt key drifts.

13. **`CkptManager` skips `last.pt` write on NaN-score epochs** — `update()` is only called when score not NaN (`train.py:743`). When val NaN, no checkpoint is saved at all, including `last.pt`. If watchdog resumes from `last.pt`, a NaN'd val epoch means the resume point is stale by one epoch.

## Nice-to-have (4)

14. **Val rows have `pre_total_mm=NaN`** (manifest), so can't filter "rainy starts" — 24/day means many starts have zero rain in the 18-frame target window. Won't break, but noise floor for CSI@30mm stays high.

15. **`split_batch` MFD fallback pads with zeros** silently (`scripts/train.py:182-189`) — if dataset emits 0 MFD channels but config says 3, model sees zero MFD without any warning. Add a one-time log.

16. `_collate` doesn't handle list-of-strings batch items distinctly; minor robustness gap.

17. `_csi_counts` docstring is duplicated (`scripts/train.py:257-259`) — copy-paste artefact, doesn't affect behaviour.

## Pre-relaunch checklist (do in order)

1. **Fix `best.pt` bug** (`scripts/train.py:368-385` — copy `best` path to `best.pt` instead of `state`). Until then **don't use any current `best.pt` for eval**; load `epoch{NNN}.pt` matching `metric_log.json` argmax.
2. **Decide val set policy**: either rebuild val from non-23·7 days, or rename current val→`val_event` and use `test_robust` as the headline val. Update `splits_val` accordingly in all ablation configs.
3. **Smoke 5-iter dry-run on `phase7a/integrated`** with `ablation_1_radar_only.yaml --debug` to measure iter-time vs Phase 5 baseline (HANDOVER §7: ~7 min/epoch). If >2× regression, diagnose before launching ab4/5/6.
4. **Move `mc_dropout_predict` inside autocast** OR document it as fp32-by-design. Time the resulting val epoch.
5. **Wire multi-threshold val FSS** (`scripts/train.py:526-532, 605-607`): add `fss_nbrs × thresholds` cross-product.
6. **Tweedie dry-run** under bf16 on 5 iters — confirm no NaN/Inf in loss or grads.
7. **Re-evaluate ab1/2/3 using corrected `best.pt` resolution** (i.e., load the `epoch{NNN}.pt` file pointed at by `argmax(metric_log)`). The +12% CPU re-eval number will move.
8. **Cache encoder output in MC-dropout** (optional but ~4× val speedup).
9. Commit the manifest `starts_per_day=24` change with a clear message explaining 3× train growth + cost implication.
10. Then — and only then — launch ab4 on AutoDL with watchdog.
