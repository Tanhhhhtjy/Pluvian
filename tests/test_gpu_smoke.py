"""
GPU smoke test for Pluvian — uses MAY 2023 data that's already on the GPU.
Validates real H=661 W=701 forward+backward on A800 80GB with bf16.

Run on GPU host:
    python tests/test_gpu_smoke.py
"""
import sys, time, os
from pathlib import Path
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.data_loader import NPJDataset
from model.pluvian import Pluvian
from model.losses import WaterBudgetLoss, WeightedMSE


def main():
    torch.manual_seed(42)
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Pick a May sample we know is available
    start = "2023-05-27T06:00:00"  # 5-27 was a big rain day in our data

    print(f"\n=== Loading sample at {start} ===")
    t0 = time.time()
    ds = NPJDataset(
        manifest="pipeline/manifest.csv",
        starts=[start],
        load_mfd=True,
        window_minutes=180,
    )
    batch = ds[0]
    print(f"Loaded in {time.time()-t0:.1f}s")
    print(f"  radar:   {batch['radar'].shape}  max={batch['radar'].max():.1f}")
    print(f"  pwv:     {batch['pwv_grid'].shape}  mask_frac={batch['pwv_mask'].mean():.3f}")
    print(f"  station: {batch['station_grid'].shape}")
    for k in ['u', 'v', 'q', 't']:
        if k in batch['era5']:
            print(f"  era5/{k}: {batch['era5'][k].shape}")
    print(f"  era5_mfd: {batch['era5_mfd'].shape}")

    # Add batch dim
    def to_dev(x):
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(0).to(device)
        if isinstance(x, dict):
            return {k: to_dev(v) for k, v in x.items()}
        return x

    batch_dev = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch_dev[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, dict):
            batch_dev[k] = {kk: vv.unsqueeze(0).to(device) if isinstance(vv, torch.Tensor) else vv
                            for kk, vv in v.items()}
        else:
            batch_dev[k] = v

    # Build model — use small preset to be safe
    print(f"\n=== Build model (small preset, mfd_channel=True for full I-2) ===")
    model = Pluvian(
        pwv_enabled=True,
        pwv_concat_only=False,
        era5_enabled=True,
        mfd_channel_enabled=True,
        forecast_frames=18,
        input_frames=12,
        hidden_dim=96,   # smaller for first smoke
        n_layers=3,
        n_heads=8,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params/1e6:.2f} M")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

    # Losses
    data_loss = WeightedMSE()
    budget_loss = WaterBudgetLoss(huber_delta=1e-4, smooth_sigma=2.0)

    # Forward
    print(f"\n=== Forward (bf16 autocast) ===")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out = model(batch_dev)
    fwd_time = time.time() - t0
    fwd_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"  rain_pred: {out['rain_pred'].shape}  range [{out['rain_pred'].min():.2f}, {out['rain_pred'].max():.2f}]")
    print(f"  pwv_pred:  {out['pwv_pred'].shape}")
    print(f"  forward time: {fwd_time:.2f}s")
    print(f"  forward peak VRAM: {fwd_peak:.2f} GB")

    # Loss (need ground truth for data — use the input radar last frames as a placeholder)
    # For smoke we just check forward+backward works
    # Use input radar's last 18 frames as fake target
    rain_target = batch_dev['radar'][:, -18:] if batch_dev['radar'].shape[1] >= 18 else batch_dev['radar']
    # If sizes don't match, pad
    if rain_target.shape[1] != out['rain_pred'].shape[1]:
        rain_target = rain_target[:, :out['rain_pred'].shape[1]]

    print(f"\n=== Compute losses ===")
    l_data = data_loss(out['rain_pred'], rain_target.float())
    print(f"  data loss: {l_data.item():.4f}")

    # Budget loss
    era5_fut = batch_dev['era5']
    # Take last 18 frames of u/v/q at level 1 (850hPa)
    T_out = out['rain_pred'].shape[1]
    u = era5_fut['u'][:, -T_out:, 1] if era5_fut['u'].ndim == 5 else era5_fut['u']
    v = era5_fut['v'][:, -T_out:, 1] if era5_fut['v'].ndim == 5 else era5_fut['v']
    q = era5_fut['q'][:, -T_out:, 1] if era5_fut['q'].ndim == 5 else era5_fut['q']
    # Match HW
    if u.shape[-2:] != out['rain_pred'].shape[-2:]:
        u = torch.nn.functional.interpolate(u.float(), size=out['rain_pred'].shape[-2:], mode='bilinear')
        v = torch.nn.functional.interpolate(v.float(), size=out['rain_pred'].shape[-2:], mode='bilinear')
        q = torch.nn.functional.interpolate(q.float(), size=out['rain_pred'].shape[-2:], mode='bilinear')

    lat = torch.linspace(36.0, 42.6, out['rain_pred'].shape[-2], device=device)
    lon = torch.linspace(113.0, 120.0, out['rain_pred'].shape[-1], device=device)

    bud_out = budget_loss(
        pwv_pred=out['pwv_pred'],
        rain_pred=out['rain_pred'],
        u=u, v=v, q=q,
        lat=lat, lon=lon,
    )
    l_bud = bud_out['loss']
    print(f"  budget loss: {l_bud.item():.6f}")

    total = l_data + 0.1 * l_bud
    print(f"  total: {total.item():.4f}")

    # Backward
    print(f"\n=== Backward + step ===")
    t0 = time.time()
    total.backward()
    bwd_time = time.time() - t0
    bwd_peak = torch.cuda.max_memory_allocated() / 1e9

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"  backward time: {bwd_time:.2f}s")
    print(f"  total peak VRAM: {bwd_peak:.2f} GB")
    print(f"  grad_norm (pre-clip): {grad_norm.item():.3f}")

    opt.step()

    # Check no NaN
    nans = any(torch.isnan(p).any().item() for p in model.parameters())
    infs = any(torch.isinf(p).any().item() for p in model.parameters())
    print(f"  any NaN in params: {nans}")
    print(f"  any Inf in params: {infs}")

    # Second forward (verify loss decreases or at least stays sane)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        out2 = model(batch_dev)
    l2_data = data_loss(out2['rain_pred'], rain_target.float())
    print(f"\n=== After 1 AdamW step ===")
    print(f"  data loss: {l2_data.item():.4f}  (Δ={l2_data.item()-l_data.item():+.4f})")

    print(f"\n=== SMOKE TEST PASS ===" if not (nans or infs) else "FAIL")
    print(f"Peak VRAM: {bwd_peak:.2f} / 80 GB ({100*bwd_peak/80:.1f}%)")
    print(f"Forward+backward total: {fwd_time+bwd_time:.2f}s per sample @ bs=1, hidden=96")
    print(f"Estimated bs=1 + hidden=192 + n_layers=4: ~{(fwd_time+bwd_time)*4:.1f}s/step")


if __name__ == "__main__":
    main()
