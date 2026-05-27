"""Tests for Phase 7b T2: asymmetric gated radar-PWV fusion."""
from __future__ import annotations

import torch

from model.fusion import GatedAsymmFusion, CrossAttentionFusion


def _make_inputs(B=1, T=4, C=64, H=8, W=8, N_pwv=12, N_sta=20, device="cpu"):
    radar = torch.randn(B, T, C, H, W, device=device)
    pwv = torch.randn(B, T, N_pwv, C, device=device)
    sta = torch.randn(B, T, N_sta, C, device=device)
    pwv_kpm = torch.zeros(B, T, N_pwv, dtype=torch.bool, device=device)
    sta_kpm = torch.zeros(B, T, N_sta, dtype=torch.bool, device=device)
    return radar, pwv, pwv_kpm, sta, sta_kpm


def test_gated_fusion_forward_shape():
    fusion = GatedAsymmFusion(dim=64, n_heads=4, n_cross_layers=1)
    radar, pwv, pwv_kpm, sta, sta_kpm = _make_inputs(C=64)
    out = fusion(radar, pwv, pwv_kpm, sta, sta_kpm)
    assert out.shape == radar.shape, f"shape mismatch: {out.shape} vs {radar.shape}"
    assert torch.isfinite(out).all()


def test_gated_fusion_param_budget():
    fusion = GatedAsymmFusion(dim=192, n_heads=8, n_cross_layers=1)
    n = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    # Task budget: <= 1.5M increase. Old CrossAttentionFusion(dim=192,
    # n_layers=2) is ~890k; the gated variant must stay within ~1.5M absolute.
    assert n <= 1_500_000, f"GatedAsymmFusion has {n} params, budget 1.5M"


def test_gated_fusion_backward_finite_grads():
    fusion = GatedAsymmFusion(dim=32, n_heads=4, n_cross_layers=1)
    radar, pwv, pwv_kpm, sta, sta_kpm = _make_inputs(C=32, H=4, W=4, N_pwv=6, N_sta=8)
    out = fusion(radar, pwv, pwv_kpm, sta, sta_kpm)
    out.pow(2).mean().backward()
    grads = [(name, p.grad) for name, p in fusion.named_parameters()
             if p.requires_grad]
    assert len(grads) > 0
    for name, g in grads:
        assert g is not None, f"{name} has no grad"
        assert torch.isfinite(g).all(), f"{name} grad has non-finite values"
    # Gates specifically must receive non-zero gradient.
    gate_grads = {n: g for n, g in grads if "gate_" in n}
    assert any(g.abs().sum().item() > 0 for g in gate_grads.values()), \
        "all gate gradients are zero"


def test_gated_fusion_init_close_to_radar():
    # With near-zero gates the fusion output should start within a small
    # delta of the radar input (i.e. the residual path dominates).
    torch.manual_seed(0)
    fusion = GatedAsymmFusion(dim=32, n_heads=4, n_cross_layers=1,
                              gate_init_bias=-4.0)
    fusion.eval()
    radar, pwv, pwv_kpm, sta, sta_kpm = _make_inputs(C=32, H=4, W=4, N_pwv=6, N_sta=8)
    with torch.no_grad():
        out = fusion(radar, pwv, pwv_kpm, sta, sta_kpm)
    delta = (out - radar).abs().mean().item()
    radar_scale = radar.abs().mean().item()
    assert delta < 0.2 * radar_scale, \
        f"init output drifts from radar by {delta:.3f} vs scale {radar_scale:.3f}"


def test_gated_fusion_handles_all_masked_rows():
    # Fully-masked PWV rows must not produce NaNs.
    fusion = GatedAsymmFusion(dim=32, n_heads=4, n_cross_layers=1)
    radar, pwv, pwv_kpm, sta, sta_kpm = _make_inputs(C=32, H=4, W=4, N_pwv=6, N_sta=8)
    pwv_kpm[:] = True  # everything masked
    out = fusion(radar, pwv, pwv_kpm, sta, sta_kpm)
    assert torch.isfinite(out).all()


def test_pluvian_pwv_concat_only_uses_legacy_fusion():
    # Backward-compat: pwv_concat_only=True must keep the original combined-bag
    # CrossAttentionFusion module.
    from model import Pluvian
    m = Pluvian(
        radar_channels=1, pwv_enabled=True, pwv_concat_only=True,
        era5_enabled=False, mfd_channel_enabled=False,
        forecast_frames=2, input_frames=4, hidden_dim=32, n_layers=2,
        n_heads=4, gated_fusion=True,  # request gated, but flag should be overridden
    )
    assert isinstance(m.fusion, CrossAttentionFusion)
    assert not m.gated_fusion


def test_pluvian_default_uses_gated_fusion():
    from model import Pluvian
    m = Pluvian(
        radar_channels=1, pwv_enabled=True, pwv_concat_only=False,
        era5_enabled=False, mfd_channel_enabled=False,
        forecast_frames=2, input_frames=4, hidden_dim=32, n_layers=2,
        n_heads=4,
    )
    assert isinstance(m.fusion, GatedAsymmFusion)
