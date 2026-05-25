"""End-to-end CPU integration test for Pluvian (T4.5).

Runs a single real sample through the full pipeline:
  NPJDataset -> Pluvian model -> data_loss + budget_loss -> backward -> opt.step

To keep CPU memory/time tractable we:
  - center-crop the radar grid (661x701) to 128x128
  - shrink T_in=8, T_out=6
  - shrink hidden_dim=64, n_layers=2
  - use fp32 (CPU has no bf16)

If you want to re-run after a code change:
    python -u tests/test_e2e_cpu.py

Writes the report to tests/e2e_cpu_report.md.
"""
from __future__ import annotations

import math
import os
import resource
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model import Pluvian  # noqa: E402
from model.losses import WeightedMSE, WaterBudgetLoss  # noqa: E402
from pipeline.data_loader import NPJDataset  # noqa: E402
from pipeline.utils import RADAR_LAT, RADAR_LON  # noqa: E402


REPORT_PATH = REPO_ROOT / "tests" / "e2e_cpu_report.md"

# --- Mini-config ------------------------------------------------------------
SEED = 42
START_TIME = "2023-07-28T12:00:00"
CROP = 128
HIDDEN_DIM = 64
N_LAYERS = 2
N_HEADS = 4
T_IN = 8
T_OUT = 6
BUDGET_WEIGHT = 0.1
BUDGET_LEVEL_IDX = 1  # 850 hPa


# --- Helpers ----------------------------------------------------------------

def _center_crop_hw(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    H, W = x.shape[-2], x.shape[-1]
    h0 = (H - h) // 2
    w0 = (W - w) // 2
    return x[..., h0:h0 + h, w0:w0 + w].contiguous()


def _crop_sample(sample: dict, h: int, w: int) -> dict:
    out = {}
    out["radar"] = _center_crop_hw(sample["radar"], h, w)
    out["radar_mask"] = sample["radar_mask"]  # per-frame scalar, no spatial dim
    out["pwv_grid"] = sample["pwv_grid"]
    out["pwv_mask"] = sample["pwv_mask"]
    out["pwv_coords"] = sample["pwv_coords"]
    out["station_grid"] = sample["station_grid"]
    out["station_mask"] = sample["station_mask"]
    out["station_coords"] = sample["station_coords"]
    out["era5"] = {k: _center_crop_hw(v, h, w) for k, v in sample["era5"].items()}
    if "era5_mfd" in sample:
        out["era5_mfd"] = _center_crop_hw(sample["era5_mfd"], h, w)
    return out


def _add_batch_dim(sample: dict) -> dict:
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        elif isinstance(v, dict):
            out[k] = {kk: vv.unsqueeze(0) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _split_batch(batch: dict, T_in: int, T_out: int, mfd_channels: int):
    """Mirror of scripts/train.py::split_batch but tolerant to T_in+T_out < n_frames."""
    radar = batch["radar"].float()
    radar_in = radar[:, :T_in]
    rain_tgt = radar[:, T_in:T_in + T_out]
    era5 = batch["era5"]
    era5_in = {k: era5[k][:, :T_in].float() for k in era5}
    era5_fut = {k: era5[k][:, T_in:T_in + T_out].float() for k in era5}

    model_in = {
        "radar": radar_in,
        "radar_mask": batch["radar_mask"][:, :T_in],
        "pwv_grid": batch["pwv_grid"],
        "pwv_mask": batch["pwv_mask"],
        "pwv_coords": batch["pwv_coords"],
        "station_grid": batch["station_grid"],
        "station_mask": batch["station_mask"],
        "station_coords": batch["station_coords"],
        "era5": era5_in,
    }
    if mfd_channels > 0 and "era5_mfd" in batch:
        mfd = batch["era5_mfd"].float()
        mfd_in = mfd[:, :T_in]
        if mfd_in.shape[2] < mfd_channels:
            pad = torch.zeros(
                mfd_in.shape[0], mfd_in.shape[1],
                mfd_channels - mfd_in.shape[2], *mfd_in.shape[-2:],
                dtype=mfd_in.dtype,
            )
            mfd_in = torch.cat([mfd_in, pad], dim=2)
        elif mfd_in.shape[2] > mfd_channels:
            mfd_in = mfd_in[:, :, :mfd_channels]
        model_in["era5_mfd"] = mfd_in
    return model_in, rain_tgt, era5_fut


def _check_finite(name: str, t: torch.Tensor):
    if not torch.isfinite(t).all():
        n_nan = torch.isnan(t).sum().item()
        n_inf = torch.isinf(t).sum().item()
        raise RuntimeError(f"{name} has NaN/Inf: nan={n_nan} inf={n_inf}")


def _group_grad_norm(model: torch.nn.Module, prefix: str) -> float:
    total = 0.0
    found = False
    for name, p in model.named_parameters():
        if name.startswith(prefix) and p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
            found = True
    return math.sqrt(total) if found else float("nan")


# --- Main test --------------------------------------------------------------

def run() -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    device = torch.device("cpu")

    print(f"[setup] device={device}  torch={torch.__version__}  threads={torch.get_num_threads()}")

    # ---- 1. Real sample ----
    print(f"[data] loading sample at {START_TIME} via NPJDataset (load_mfd=True) ...")
    t = time.time()
    ds = NPJDataset(
        manifest=REPO_ROOT / "pipeline" / "manifest.csv",
        window_minutes=180,
        starts=[START_TIME],
        drop_pwv=False,
        load_mfd=True,
    )
    sample = ds[0]
    print(f"[data] loaded in {time.time() - t:.1f}s")
    print(f"[data] radar       shape={tuple(sample['radar'].shape)} dtype={sample['radar'].dtype}")
    print(f"[data] era5['u']   shape={tuple(sample['era5']['u'].shape)}")
    if "era5_mfd" in sample:
        print(f"[data] era5_mfd    shape={tuple(sample['era5_mfd'].shape)}")
    print(f"[data] pwv_grid    shape={tuple(sample['pwv_grid'].shape)}")
    print(f"[data] station_grid shape={tuple(sample['station_grid'].shape)}")

    # ---- 2. Crop to CROPxCROP and add batch dim ----
    sample = _crop_sample(sample, CROP, CROP)
    batch = _add_batch_dim(sample)
    print(f"[crop] batch radar shape={tuple(batch['radar'].shape)} "
          f"era5['u']={tuple(batch['era5']['u'].shape)}")

    # ---- 3. Cropped lat/lon for budget loss ----
    H_full = len(RADAR_LAT)
    W_full = len(RADAR_LON)
    h0 = (H_full - CROP) // 2
    w0 = (W_full - CROP) // 2
    lat_crop = np.asarray(RADAR_LAT[h0:h0 + CROP], dtype=np.float32)
    lon_crop = np.asarray(RADAR_LON[w0:w0 + CROP], dtype=np.float32)
    lat_t = torch.from_numpy(lat_crop)
    lon_t = torch.from_numpy(lon_crop)

    # ---- 4. Mini model ----
    model = Pluvian(
        radar_channels=1,
        pwv_enabled=True,
        pwv_concat_only=False,
        era5_enabled=True,
        mfd_channel_enabled=True,
        forecast_frames=T_OUT,
        input_frames=T_IN,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_era5_vars=4,
        n_era5_levels=8,
        mfd_channels=3,
    ).to(device).float()

    n_params = model.num_parameters()
    print(f"[model] params: {n_params/1e6:.3f}M")

    # ---- 5. Losses + optim ----
    data_loss = WeightedMSE().to(device)
    budget_loss = WaterBudgetLoss(
        delta_p_pa=30000.0, smooth_sigma=2.0, huber_delta=1e-4,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05,
                            betas=(0.9, 0.95))

    model_in, rain_tgt, era5_fut = _split_batch(batch, T_IN, T_OUT, mfd_channels=3)

    expected_shape = (1, T_OUT, CROP, CROP)
    assert rain_tgt.shape == expected_shape, f"rain_tgt shape {rain_tgt.shape} != {expected_shape}"

    # ---- 6. Forward (timed + tracemalloc + RSS) ----
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    tracemalloc.start()
    t0 = time.time()
    out = model(model_in)
    fwd_time = time.time() - t0
    snap_fwd, peak_fwd = tracemalloc.get_traced_memory()
    rss_after_fwd = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    rain_pred = out["rain_pred"]
    pwv_pred = out["pwv_pred"]
    print(f"[fwd]  rain_pred shape={tuple(rain_pred.shape)} "
          f"pwv_pred shape={tuple(pwv_pred.shape)} ({fwd_time:.2f}s)")

    assert rain_pred.shape == expected_shape, f"rain_pred shape mismatch: {rain_pred.shape}"
    assert pwv_pred.shape == expected_shape, f"pwv_pred shape mismatch: {pwv_pred.shape}"
    _check_finite("rain_pred", rain_pred)
    _check_finite("pwv_pred", pwv_pred)

    # ---- 7. Losses ----
    l_data = data_loss(rain_pred, rain_tgt.float()) * 1.0
    u = era5_fut["u"][:, :, BUDGET_LEVEL_IDX]
    v = era5_fut["v"][:, :, BUDGET_LEVEL_IDX]
    q = era5_fut["q"][:, :, BUDGET_LEVEL_IDX]
    bud_out = budget_loss(
        pwv_pred=pwv_pred, rain_pred=rain_pred,
        u_era5=u, v_era5=v, q_era5=q, lat=lat_t, lon=lon_t,
    )
    l_bud = bud_out["loss"] * BUDGET_WEIGHT
    l_total = l_data + l_bud
    print(f"[loss] data={l_data.item():.6f}  budget={l_bud.item():.6f}  total={l_total.item():.6f}")
    _check_finite("loss", l_total)

    # ---- 8. Backward ----
    t0 = time.time()
    l_total.backward()
    bwd_time = time.time() - t0
    snap_bwd, peak_bwd = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after_bwd = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is in kilobytes on Linux
    rss_peak_mb = max(rss_after_fwd, rss_after_bwd) / 1024.0
    rss_delta_mb = (max(rss_after_fwd, rss_after_bwd) - rss_before) / 1024.0
    print(f"[bwd]  done in {bwd_time:.2f}s")
    print(f"[mem]  RSS peak={rss_peak_mb:.1f} MB (Δ since pre-fwd: {rss_delta_mb:+.1f} MB)")

    # ---- 9. Grad sanity ----
    grad_norms = {
        "encoder.radar": _group_grad_norm(model, "radar_encoder."),
        "encoder.era5":  _group_grad_norm(model, "era5_encoder."),
        "encoder.sparse": _group_grad_norm(model, "sparse_encoder."),
        "fusion":        _group_grad_norm(model, "fusion."),
        "decoder":       _group_grad_norm(model, "decoder."),
    }
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            _check_finite("grad", p.grad)
            total_sq += p.grad.detach().float().pow(2).sum().item()
    total_grad_norm = math.sqrt(total_sq)
    print(f"[grad] total_norm={total_grad_norm:.4f}")
    for k, v_ in grad_norms.items():
        print(f"[grad] {k:>18s}: {v_:.4f}")

    if not (1e-6 <= total_grad_norm <= 1e3):
        raise RuntimeError(f"grad_norm {total_grad_norm} outside [1e-6, 1e3]")
    for k, v_ in grad_norms.items():
        if math.isnan(v_) or v_ <= 0:
            raise RuntimeError(f"group {k} received no gradient (norm={v_})")

    # ---- 10. Step + re-eval loss on the same sample ----
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    with torch.no_grad():
        out2 = model(model_in)
        l_data2 = data_loss(out2["rain_pred"], rain_tgt.float()) * 1.0
        u2 = era5_fut["u"][:, :, BUDGET_LEVEL_IDX]
        v2 = era5_fut["v"][:, :, BUDGET_LEVEL_IDX]
        q2 = era5_fut["q"][:, :, BUDGET_LEVEL_IDX]
        bud_out2 = budget_loss(
            pwv_pred=out2["pwv_pred"], rain_pred=out2["rain_pred"],
            u_era5=u2, v_era5=v2, q_era5=q2, lat=lat_t, lon=lon_t,
        )
        l_total2 = l_data2 + bud_out2["loss"] * BUDGET_WEIGHT
    delta = l_total2.item() - l_total.item()
    print(f"[step] loss before={l_total.item():.6f} after={l_total2.item():.6f}  Δ={delta:+.6f}")
    # Allow a small uphill jump (single sample, no scheduling), but not a blow-up.
    if delta > 0.5 * abs(l_total.item()) + 1e-3:
        raise RuntimeError(f"loss increased too much after step: Δ={delta}")

    peak_mb = max(peak_fwd, peak_bwd) / (1024 * 1024)
    print(f"[mem]  peak (tracemalloc, Python-side only) {peak_mb:.1f} MB")

    # ---- 11. GPU estimate at full resolution ----
    # Heuristics (rough scaling — A800 40GB):
    #   * memory scales ~ B * (H*W) * hidden_dim * (n_layers/2 + decoder_factor)
    #   * time scales linearly with HW (conv-heavy) plus quadratic on the
    #     latent grid (window attention is ~linear with window=8 fixed)
    full_h, full_w = H_full, W_full
    full_bs = 4
    # Compare current peak HW vs full HW at hidden=192 base config.
    hw_ratio = (full_h * full_w) / (CROP * CROP)
    hidden_ratio_mem = 192 / HIDDEN_DIM
    n_layers_ratio = 4 / N_LAYERS
    bs_ratio = full_bs / 1
    # Memory estimate (very approximate): activations scale linearly with HW,
    # hidden, n_layers and bs. Use RSS delta during fwd+bwd as a proxy for
    # the mini activation footprint (PyTorch allocs live in C++, not visible
    # to tracemalloc).
    act_mb = max(rss_delta_mb, 1.0)
    gpu_mem_gb = act_mb / 1024.0 * hw_ratio * hidden_ratio_mem * n_layers_ratio * bs_ratio
    # Time per step on GPU ~ CPU time scaled by HW & hidden, divided by ~50x speedup.
    gpu_time_s = (fwd_time + bwd_time) * hw_ratio * hidden_ratio_mem * n_layers_ratio * bs_ratio / 50.0
    starts_per_day = 8  # NPJDataset uses 3h step -> 8 starts/day
    n_days = 118
    steps_per_epoch = n_days * starts_per_day / full_bs
    epoch_hours = gpu_time_s * steps_per_epoch / 3600.0
    ablation_days = epoch_hours * 60 / 24.0
    print(f"[gpu_est] mem~{gpu_mem_gb:.1f} GB  step~{gpu_time_s:.2f}s  "
          f"epoch~{epoch_hours:.2f}h  60-epoch ablation~{ablation_days:.1f}d")

    return {
        "n_params_M": n_params / 1e6,
        "fwd_time_s": fwd_time,
        "bwd_time_s": bwd_time,
        "peak_mb": peak_mb,
        "rss_peak_mb": rss_peak_mb,
        "rss_delta_mb": rss_delta_mb,
        "l_data": l_data.item(),
        "l_bud": l_bud.item(),
        "l_total": l_total.item(),
        "l_total_after": l_total2.item(),
        "delta": delta,
        "grad_norm": total_grad_norm,
        "grad_groups": grad_norms,
        "gpu_mem_gb": gpu_mem_gb,
        "gpu_step_s": gpu_time_s,
        "gpu_epoch_h": epoch_hours,
        "gpu_ablation_d": ablation_days,
    }


def _write_report(r: dict):
    md = f"""# Pluvian E2E CPU Test Report

(Generated by `tests/test_e2e_cpu.py`, single real sample at {START_TIME}.)

## Config (mini)
- hidden_dim = {HIDDEN_DIM}, n_layers = {N_LAYERS}, n_heads = {N_HEADS}
- batch_size = 1, T_in = {T_IN}, T_out = {T_OUT}
- H = W = {CROP} (center-crop of full {len(RADAR_LAT)}x{len(RADAR_LON)})
- device = cpu, dtype = fp32, seed = {SEED}
- mfd_channel_enabled = True, budget_loss = on (weight {BUDGET_WEIGHT})

## Model
- Parameters: **{r['n_params_M']:.3f} M**
- Forward time:  {r['fwd_time_s']:.2f} s
- Backward time: {r['bwd_time_s']:.2f} s
- Peak host memory (tracemalloc, Python only): {r['peak_mb']:.1f} MB
- Process RSS peak: {r['rss_peak_mb']:.1f} MB (Δ during fwd+bwd: {r['rss_delta_mb']:+.1f} MB)

## Loss
- Data loss (weighted MSE): {r['l_data']:.6f}
- Budget loss (weight {BUDGET_WEIGHT}): {r['l_bud']:.6f}
- Total loss: {r['l_total']:.6f}
- After 1 optim step: {r['l_total_after']:.6f}  (Δ = {r['delta']:+.6f})

## Gradient stats (post-backward, pre-clip)
- Total grad norm: {r['grad_norm']:.4f}
"""
    for k, v in r["grad_groups"].items():
        md += f"- {k}: {v:.4f}\n"
    md += f"""

## GPU estimate (A800 40GB, full resolution H={len(RADAR_LAT)}, W={len(RADAR_LON)}, bs=4, hidden=192, n_layers=4)
(Very rough — linear extrapolation from this CPU mini-run; expect ±2x error.)
- Memory: ~{r['gpu_mem_gb']:.1f} GB
- Time per step: ~{r['gpu_step_s']:.2f} s
- Time per epoch (~118 days × 8 starts/day ÷ bs=4): ~{r['gpu_epoch_h']:.2f} h
- Time per ablation (60 epochs): ~{r['gpu_ablation_d']:.1f} days

## Assertions
- All output shapes match declarations
- No NaN/Inf in outputs, losses, or grads
- Total grad_norm in [1e-6, 1e3]
- Every key module (encoder.radar, encoder.era5, encoder.sparse, fusion, decoder) received non-zero grad
- Loss after one step did not blow up

## Conclusion
**PASS** ✓
"""
    REPORT_PATH.write_text(md)
    print(f"[report] written to {REPORT_PATH}")


if __name__ == "__main__":
    r = run()
    _write_report(r)
    print("=== Pluvian E2E CPU Test: PASS ✓ ===")
