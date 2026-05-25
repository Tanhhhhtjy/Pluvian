"""Shape-and-gradient smoke test for the Pluvian model.

Run with::

    python -m model.test_shapes

Uses tiny H=W=128 inputs so it runs on CPU in a few seconds. The full
H=661/W=701 forward should only be attempted on GPU.
"""
from __future__ import annotations

import torch

from model import Pluvian


def _fake_batch(B: int = 1, T_in: int = 30, H: int = 128, W: int = 128,
                n_pwv: int = 176, n_sta: int = 289,
                era5_h: int = 30, era5_w: int = 30) -> dict:
    rng = torch.Generator().manual_seed(0)
    T_pwv = max(1, T_in * 6 // 30)       # 30-min PWV
    T_sta = max(1, T_in * 6 // 60)       # 1-h stations
    return {
        "radar": torch.randn(B, T_in, H, W, generator=rng),
        "radar_mask": torch.ones(B, T_in, H, W),
        "pwv_grid": torch.randn(B, T_pwv, n_pwv, generator=rng),
        "pwv_mask": torch.ones(B, T_pwv, n_pwv),
        "pwv_coords": torch.stack([
            torch.linspace(113.5, 119.5, n_pwv),
            torch.linspace(36.5, 42.0, n_pwv),
        ], dim=-1),
        "station_grid": torch.randn(B, T_sta, n_sta, 6, generator=rng),
        "station_mask": torch.ones(B, T_sta, n_sta),
        "station_coords": torch.stack([
            torch.linspace(113.5, 119.5, n_sta),
            torch.linspace(36.5, 42.0, n_sta),
            torch.linspace(0.0, 1500.0, n_sta),
        ], dim=-1),
        "era5": {
            "u": torch.randn(B, T_in, 8, era5_h, era5_w, generator=rng),
            "v": torch.randn(B, T_in, 8, era5_h, era5_w, generator=rng),
            "q": torch.randn(B, T_in, 8, era5_h, era5_w, generator=rng),
            "t": torch.randn(B, T_in, 8, era5_h, era5_w, generator=rng),
        },
        "era5_mfd": torch.randn(B, T_in, 3, era5_h, era5_w, generator=rng),
    }


def _check(model: Pluvian, batch: dict, label: str, B: int, H: int, W: int):
    model.train()
    out = model(batch)
    assert out["rain_pred"].shape == (B, model.forecast_frames, H, W), \
        f"[{label}] rain shape {out['rain_pred'].shape}"
    assert out["pwv_pred"].shape == (B, model.forecast_frames, H, W), \
        f"[{label}] pwv shape {out['pwv_pred'].shape}"
    loss = out["rain_pred"].pow(2).mean() + out["pwv_pred"].pow(2).mean()
    loss.backward()
    grad_norm = sum(
        p.grad.detach().norm().item()
        for p in model.parameters()
        if p.grad is not None
    )
    print(f"  [{label}] OK  rain={tuple(out['rain_pred'].shape)} "
          f"pwv={tuple(out['pwv_pred'].shape)}  loss={loss.item():.4f}  "
          f"grad_norm_total={grad_norm:.2f}")
    # clear grads for next variant
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def main():
    B, T_in, H, W = 1, 30, 128, 128
    batch = _fake_batch(B=B, T_in=T_in, H=H, W=W)

    # --- 1. main "base" model ---
    print("== Pluvian(base) — main proposal ==")
    model = Pluvian(input_frames=T_in, forecast_frames=18, size="base",
                    mfd_channel_enabled=True)
    n_params = model.num_parameters()
    print(f"  params: {n_params/1e6:.2f}M")
    assert 20e6 < n_params < 80e6, f"param budget out of bounds: {n_params/1e6:.1f}M"
    _check(model, batch, "base + MFD", B, H, W)

    # --- 2. small variant ---
    print("== Pluvian(small) ==")
    model = Pluvian(input_frames=T_in, forecast_frames=18, size="small")
    print(f"  params: {model.num_parameters()/1e6:.2f}M")
    _check(model, batch, "small", B, H, W)

    # --- 3. ablations ---
    print("== Ablations on base ==")
    for kwargs, label in [
        (dict(pwv_enabled=False), "no-pwv"),
        (dict(pwv_concat_only=True), "pwv-concat-only"),
        (dict(era5_enabled=False), "no-era5"),
        (dict(mfd_channel_enabled=False), "base-no-mfd"),
    ]:
        m = Pluvian(input_frames=T_in, forecast_frames=18, size="base", **kwargs)
        print(f"  [{label}] params: {m.num_parameters()/1e6:.2f}M")
        _check(m, batch, label, B, H, W)

    print("\nAll shape / backward checks passed.")


if __name__ == "__main__":
    main()
