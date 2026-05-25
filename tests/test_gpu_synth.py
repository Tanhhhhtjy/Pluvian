"""
Pure synthetic GPU smoke test for Pluvian — no real data needed.

Verifies the model architecture works end-to-end on A800 80GB at FULL resolution
(H=661, W=701), bf16, with realistic batch composition.

Run on GPU host:
    python tests/test_gpu_synth.py
"""
import sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.pluvian import Pluvian
from model.losses import WaterBudgetLoss, WeightedMSE


def make_synth_batch(B=1, T_in=12, T_out=18, H=661, W=701,
                     n_pwv=176, n_sta=289, n_levels=8, device="cuda"):
    """Build a batch shaped exactly like NPJDataset would output."""
    torch.manual_seed(42)

    # Choose realistic temporal split: input frames are most recent T_in, but
    # full window has T_in + T_out timestamps. For most data the dataset gives
    # T_total = T_in + T_out frames of radar; encoder uses first T_in, target uses last T_out.
    T_total = T_in + T_out

    return {
        "radar": torch.rand(B, T_total, H, W, device=device) * 50.0,        # dBZ-like
        "pwv_grid": torch.rand(B, T_total // 5, n_pwv, device=device) * 60.0,  # mm-like, 30-min
        "pwv_mask": torch.ones(B, T_total // 5, n_pwv, device=device),
        "pwv_coords": torch.stack([
            torch.rand(n_pwv, device=device) * 7 + 113,  # lon
            torch.rand(n_pwv, device=device) * 6 + 36,   # lat
        ], dim=-1),  # (N, 2)
        "station_grid": torch.randn(B, T_total // 10, n_sta, 6, device=device),
        "station_mask": torch.ones(B, T_total // 10, n_sta, device=device),
        "station_coords": torch.stack([
            torch.rand(n_sta, device=device) * 7 + 113,
            torch.rand(n_sta, device=device) * 6 + 36,
            torch.rand(n_sta, device=device) * 1500,
        ], dim=-1),  # (N, 3)
        "era5": {
            "u": torch.randn(B, T_total, n_levels, H, W, device=device) * 10,
            "v": torch.randn(B, T_total, n_levels, H, W, device=device) * 10,
            "q": torch.rand(B, T_total, n_levels, H, W, device=device) * 0.02,
            "t": 250 + torch.rand(B, T_total, n_levels, H, W, device=device) * 50,
        },
        "era5_mfd": torch.randn(B, T_total, 3, H, W, device=device) * 1e-7,
        "lat": torch.linspace(36.0, 42.6, H, device=device),
        "lon": torch.linspace(113.0, 120.0, W, device=device),
    }


def main():
    device = torch.device("cuda")
    print(f"=== GPU: {torch.cuda.get_device_name(0)} ===")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {total_mem:.1f} GB")

    # Test configurations (escalating model size)
    configs = [
        ("small", dict(hidden_dim=96, n_layers=2, n_heads=4)),
        ("base", dict(hidden_dim=192, n_layers=4, n_heads=8)),
    ]

    for name, cfg in configs:
        print(f"\n{'='*60}\n=== Config: {name} ({cfg}) ===\n{'='*60}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            model = Pluvian(
                pwv_enabled=True,
                pwv_concat_only=False,
                era5_enabled=True,
                mfd_channel_enabled=True,
                forecast_frames=18,
                input_frames=12,
                **cfg,
            ).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"Params: {n_params/1e6:.2f} M")

            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
            data_loss = WeightedMSE()
            budget_loss = WaterBudgetLoss(huber_delta=1e-4, smooth_sigma=2.0).to(device)

            batch = make_synth_batch(B=1, T_in=12, T_out=18, device=device)
            print(f"Batch shapes:")
            print(f"  radar: {batch['radar'].shape}")
            print(f"  era5.u: {batch['era5']['u'].shape}")
            print(f"  era5_mfd: {batch['era5_mfd'].shape}")

            # Forward
            t0 = time.time()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(batch)
            fwd_time = time.time() - t0
            fwd_peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"\nForward: {fwd_time:.2f}s, peak VRAM after fwd: {fwd_peak:.2f} GB")
            print(f"  rain_pred: {out['rain_pred'].shape}")
            print(f"  pwv_pred:  {out['pwv_pred'].shape}")

            # Losses (interpolate target to model output shape — model pads to 8x multiple)
            import torch.nn.functional as F
            target_rain = batch['radar'][:, -18:].float()
            if target_rain.shape[-2:] != out['rain_pred'].shape[-2:]:
                target_rain = F.interpolate(target_rain, size=out['rain_pred'].shape[-2:], mode='bilinear', align_corners=False)
            l_data = data_loss(out['rain_pred'], target_rain)

            u850 = batch['era5']['u'][:, -18:, 1].float()
            v850 = batch['era5']['v'][:, -18:, 1].float()
            q850 = batch['era5']['q'][:, -18:, 1].float()
            # Interpolate u,v,q to model output shape
            if u850.shape[-2:] != out['rain_pred'].shape[-2:]:
                u850 = F.interpolate(u850, size=out['rain_pred'].shape[-2:], mode='bilinear', align_corners=False)
                v850 = F.interpolate(v850, size=out['rain_pred'].shape[-2:], mode='bilinear', align_corners=False)
                q850 = F.interpolate(q850, size=out['rain_pred'].shape[-2:], mode='bilinear', align_corners=False)

            # Recompute lat/lon for the model output shape
            lat_model = torch.linspace(36.0, 42.6, out['rain_pred'].shape[-2], device=device)
            lon_model = torch.linspace(113.0, 120.0, out['rain_pred'].shape[-1], device=device)

            bud = budget_loss(
                pwv_pred=out['pwv_pred'],
                rain_pred=out['rain_pred'],
                u_era5=u850, v_era5=v850, q_era5=q850,
                lat=lat_model, lon=lon_model,
            )
            l_bud = bud['loss']
            total = l_data + 0.1 * l_bud
            print(f"Losses: data={l_data.item():.3f} bud={l_bud.item():.3e} total={total.item():.3f}")

            # Backward + step
            t0 = time.time()
            total.backward()
            bwd_time = time.time() - t0
            bwd_peak = torch.cuda.max_memory_allocated() / 1e9
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            print(f"Backward: {bwd_time:.2f}s, peak VRAM after bwd: {bwd_peak:.2f} GB, grad_norm: {grad_norm.item():.2f}")

            opt.step()

            # No NaN check
            nans = any(torch.isnan(p).any().item() for p in model.parameters())
            print(f"NaN in params: {nans}")

            print(f"\n>>> {name} PASSED <<<")
            print(f"VRAM headroom: {total_mem - bwd_peak:.1f} GB ({100*(total_mem-bwd_peak)/total_mem:.1f}%)")
            print(f"Time per step (bs=1, bf16): {fwd_time+bwd_time:.2f}s")

            del model, opt, out, batch, l_data, l_bud, total
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            print(f">>> {name} OOM: {e}")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f">>> {name} ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== ALL CONFIGS TESTED ===")


if __name__ == "__main__":
    main()
