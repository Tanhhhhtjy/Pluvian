# §2 Method (draft 2026-06-14)

## 2.1 Data sources and preprocessing

The model ingests three streams over a North-China domain (36.00–42.60 °N, 113.00–120.00 °E) on a 0.01° (~1 km) regular lon–lat mesh of shape H × W = 661 × 701.

The primary observation is the operational S-band radar mosaic, stored as 6 min NetCDF tiles in physical dBZ (1 dBZ quantization, 0 – ~70 dBZ); gaps < 30 min are forward-filled and longer gaps marked invalid. Reflectivity is converted to rain rate via the China operational Z–R relation R = (Z/300)^(1/1.4), Z = 10^(dBZ/10) [Fulton et al. 1998]; all losses and metrics are in mm h⁻¹.

The secondary observation is ground-based GNSS precipitable water vapour (PWV) from a regional reference network at 30 min cadence; where a PWV station is colocated with a surface met. station, its precise (lon, lat) replaces the coarse 0.1° PWV coordinate, and seven near-empty stations are dropped per our data audit. PWV is delivered as a sparse token bag with a per-step validity mask rather than as a gridded field.

Reanalysis context is taken from ERA5 at hourly cadence on the native 0.25° grid: four variables (u, v, q, T) at eight pressure levels (925, 850, 700, 600, 500, 400, 300, 250 hPa) — 32 channels per frame — bilinearly interpolated to the radar window and time-nearest-neighbour to the 6 min grid.

Each sample is a 180 min window of 30 frames split into 12 history (72 min) and 18 forecast (108 min) frames. Calendar days are partitioned into train (74), val (8), test_robust (8 non-storm days, 192 windows at 24 starts day⁻¹) and event_test (4 consecutive storm days 2023-07-29 to 2023-08-01, 96 windows); 29 days are dropped for data quality. All four splits lie within May–August 2023 so that PWV is not silently zeroed on held-out evaluation.

## 2.2 Multimodal encoder–decoder architecture

The Pluvian backbone (~6.66 M parameters in the ab3 ERA5 configuration) is a three-branch encoder, a cross-attention fusion stage, and a single shared multi-scale decoder.

The **radar branch** uses a conv patch stem (factor 4) plus one stride-2 stage to reach H/8 × W/8, followed by two windowed self-attention blocks (8 px window, non-shifted) and one temporal block. Three intermediate feature maps — `stage0` (H/2), `stage1` (H/4), `stage2` (H/8) — are exposed as U-Net skips. The **ERA5 branch** stacks vars × levels into 32 channels and runs a small 2-D conv tower on the coarse 0.25° grid; the resulting feature is bilinearly upsampled and added residually to the H/8 radar feature, keeping the heavy computation on the coarse grid. The **sparse-token branch** tokenises PWV and surface-station observations with Fourier positional encoding over (lon, lat) and a per-step value embedding; missing stations are masked out.

The default fusion in the ab3 family is `GatedAsymmFusion`: a learnable gate weighs a dedicated PWV cross-attention branch against a separate surface-station branch, both querying the dense radar feature as Q. The legacy `CrossAttentionFusion` (single token bag) is retained for ab1/ab2.

The decoder begins with a per-spatial-location cross-attention mapping T_in fused tokens onto T_out = 18 learnable lead-time queries, so each forecast frame attends to its own subset of history. The resulting (B, T_out, C, H/8, W/8) tensor is upsampled ×8 through three U-Net stages with skip fuses at H/8, H/4, H/2, ending in two native-resolution heads: a single-channel PWV head and an intensity-stratified rain head (§2.4).

## 2.3 Cubic Dual Upsampling (CDU) decoder

In the CDU variant (ab3-cdu, ~6.91 M parameters; the +0.25 M delta is entirely in the three decoder upsample blocks) the standard `PixelShuffle ×2 → Conv → GN → GELU` block is replaced by a dual-branch `_CDUUpBlock` adapted from exPreCast [Song et al. ICLR 2026]. Each block has a **low-frequency branch** (bicubic interpolation ×2 + 1×1 channel projection) for the smooth field and a **high-frequency branch** (3×3 conv to 4·out_ch channels + `PixelShuffle ×2`) for strong-convective texture as a learnable residual; the two outputs are concatenated and fused by 3×3 conv + GroupNorm + GELU. The motivation is that a single PixelShuffle path tends to smooth high-intensity cores during upsampling; the bicubic branch lets the network preserve the envelope explicitly, freeing the PixelShuffle branch to specialise in residual sharpness.

The CDU substitution is the **only** delta between ab3 and ab3-cdu: data pipeline, loss configuration, optimizer, LR schedule, training horizon, mixed precision and random seed are bit-identical. The CDU run is warm-started from the converged ab3 checkpoint; the eighteen new CDU-branch parameter tensors are randomly initialised, and the optimizer and LR schedule are reset.

## 2.4 Loss function

The rain head emits five intensity bands with centres (0, 0.5, 4.5, 19, 50) mm h⁻¹. The data loss is an **intensity-stratified loss** combining (i) multinomial cross-entropy with hard band labels obtained by `torch.bucketize` on the target at edges (0.1, 1.0, 8.0, 30.0) mm h⁻¹, and (ii) a per-pixel weighted squared error on the softmax-expectation rain field with band weights (0.1, 1.0, 2.0, 4.0, 8.0). The heavy upper-band weights explicitly compensate for the long-tailed rain-rate distribution that would otherwise collapse the model to climatology. Both terms enter with weights `ce_weight = reg_weight = 1.0`.

An auxiliary **fractions skill score (FSS)** term is added at thresholds τ ∈ {1, 10, 30} mm h⁻¹ with relative weights (1.0, 0.1, 0.1) and a 9 × 9 averaging window, contributing a fixed multiplier 0.1 to the total loss.

For the budget variant (ab3b) we add a soft physics constraint  ε = ∂PWV/∂t + (Δp / g) · ∇·(q V) + P  whose residual ε is penalised with a Huber loss (δ = 10⁻⁴ mm s⁻¹) [Trenberth 1991]. Here PWV is the predicted column water vapour, V = (u, v) and q are the ERA5 fields at 850 hPa (`level_idx = 1`), Δp = 3 × 10⁴ Pa is a 700–1000 hPa column-thickness proxy that converts single-level horizontal moisture flux divergence to column-integrated VIMFD, and P is the predicted rain rate divided by 3600. Divergence is computed on the sphere with a σ = 2 grid-point Gaussian pre-smoothing. The budget term enters the total loss with weight 0.1 in ab3b and 0.0 elsewhere.

## 2.5 Training protocol

All ablations share the same recipe: AdamW (lr = 3 × 10⁻⁴, weight decay 0.05), cosine LR with 2-epoch linear warm-up and 60-epoch horizon, gradient clip 1.0, effective batch size 4 (physical batch 1 + 4 grad-accum micro-batches), bf16 mixed precision. Training was performed on a single NVIDIA GeForce RTX 4090 (24 GB); mean wall-clock per epoch is ≈ 1100 s (range 1020–1390 s, variance dominated by ERA5 disk I/O), totalling ≈ 18 h for 60 epochs. Checkpoints are selected by best `val/csi_10mm` (top-3 retained).

The ab1, ab2, ab3 and ab3b runs are randomly initialised. The ab3-cdu run is warm-started from the best ab3 checkpoint (strict-loader diff: 6 unexpected single-branch keys, 18 missing CDU-branch keys); the new parameters are randomly initialised and the optimizer and LR schedule are reset.

## 2.6 Evaluation protocol

Skill metrics are computed full-set: per-batch hit/false-alarm/miss counts are accumulated across the whole split and CSI/FSS evaluated once at the end, eliminating the ~0.02–0.05 batch-imbalance drift seen with the per-batch-mean estimator during Phase 7a.

We report **CSI** at τ ∈ {1, 5, 10, 30} mm h⁻¹, **FSS** at neighbourhood window w ∈ {3, 11} px on τ = 1 mm h⁻¹, and **MAE** on the rain field. To distinguish amplitude failure from spatial displacement at high intensities we additionally report a **pooled CSI**: for prediction ŷ, ground truth y, threshold τ and pool size w,

  pooled-CSI(ŷ, y, τ, w) = CSI( MaxPool_w(𝟙{ŷ ≥ τ}), MaxPool_w(𝟙{y ≥ τ}) )

where MaxPool_w is a w × w **non-overlapping** max-pool with `ceil_mode=True` — we follow this convention (rather than the more common stride-1 sliding window) to align with exPreCast [Song et al. ICLR 2026]. With w ∈ {1, 4, 16}, w = 1 reduces to exact-grid CSI; a CSI that recovers as w grows isolates a displacement failure, while one that stays low indicates an amplitude failure. **Per-lead diagnostics** report CSI/FSS separately at each lead 6 min ≤ t ≤ 108 min.

`event_test` (96 windows) probes peak-intensity behaviour; `test_robust` (192 windows) probes calibration on near-climatology conditions. Uncertainty is quantified by paired **day-bootstrap** confidence intervals (n_boot = 1000, fixed seed) over manifest dates, recomputing CSI/FSS from raw counts within each resample.
