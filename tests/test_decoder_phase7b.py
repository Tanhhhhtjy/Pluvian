"""Phase 7b T1 decoder: U-Net skip + per-lead-time queries.

Verifies:
  (a) output shapes match (B, T_out, H, W) for rain & pwv
  (b) skip-fed forward differs numerically from no-skip forward
      (proves skip features actually contribute)
  (c) MC-dropout spread > 0 across K passes
  (d) backward produces finite gradients on every skip-conv weight
"""
from __future__ import annotations

import torch

from model.decoder import PluvianDecoder


def _make_inputs(B=1, T_in=12, T_out=18, C=192, h=32, w=32):
    feats = torch.randn(B, T_in, C, h, w)
    skips = {
        "stage1": torch.randn(B, T_in, C, h * 2, w * 2),  # H/4
        "stage2": torch.randn(B, T_in, C, h, w),          # H/8
    }
    return feats, skips


def test_decoder_shape_with_skips():
    B, T_in, T_out, C, h, w = 1, 12, 18, 192, 32, 32
    dec = PluvianDecoder(dim=C, forecast_frames=T_out, input_frames=T_in,
                         upsample_factor=8, dropout=0.0)
    feats, skips = _make_inputs(B, T_in, T_out, C, h, w)
    rain, pwv = dec(feats, encoder_skips=skips)
    assert rain.shape == (B, T_out, h * 8, w * 8)
    assert pwv.shape == (B, T_out, h * 8, w * 8)


def test_decoder_skips_change_output():
    torch.manual_seed(0)
    dec = PluvianDecoder(dim=192, forecast_frames=18, input_frames=12,
                         upsample_factor=8, dropout=0.0)
    dec.eval()
    feats, skips = _make_inputs()
    with torch.no_grad():
        r_skip, _ = dec(feats, encoder_skips=skips)
        r_noskip, _ = dec(feats, encoder_skips=None)
    diff = (r_skip - r_noskip).abs().mean().item()
    assert diff > 1e-4, f"skip pathway has no effect (diff={diff})"


def test_decoder_mc_dropout_spread():
    torch.manual_seed(1)
    dec = PluvianDecoder(dim=192, forecast_frames=18, input_frames=12,
                         upsample_factor=8, dropout=0.5)
    dec.eval()
    dec.enable_mc_dropout(True)
    feats, skips = _make_inputs()
    with torch.no_grad():
        samples = torch.stack([dec(feats, encoder_skips=skips)[0]
                               for _ in range(4)], dim=0)
    spread = samples.std(dim=0).mean().item()
    assert spread > 1e-4, f"MC-dropout produced zero spread ({spread})"


def test_decoder_eval_mode_deterministic():
    torch.manual_seed(2)
    dec = PluvianDecoder(dim=192, forecast_frames=18, input_frames=12,
                         upsample_factor=8, dropout=0.5)
    dec.eval()
    feats, skips = _make_inputs()
    with torch.no_grad():
        a, _ = dec(feats, encoder_skips=skips)
        b, _ = dec(feats, encoder_skips=skips)
    assert torch.allclose(a, b), "eval-mode forward should be deterministic"


def test_decoder_backward_finite_grads():
    dec = PluvianDecoder(dim=192, forecast_frames=18, input_frames=12,
                         upsample_factor=8, dropout=0.0)
    feats, skips = _make_inputs()
    rain, pwv = dec(feats, encoder_skips=skips)
    loss = rain.mean() + pwv.mean()
    loss.backward()
    for name in ("skip2_proj.weight", "skip1_proj.weight",
                 "fuse_stage2.conv.weight", "fuse_stage1.conv.weight",
                 "skip2_time.weight", "skip1_time.weight",
                 "lead_attn.queries"):
        p = dict(dec.named_parameters())[name]
        assert p.grad is not None, f"{name} has no grad"
        assert torch.isfinite(p.grad).all(), f"{name} grad not finite"
        assert p.grad.abs().sum().item() > 0, f"{name} grad all-zero"


def test_decoder_no_skips_path():
    """Backward compat: encoder_skips=None still returns correct shape."""
    dec = PluvianDecoder(dim=192, forecast_frames=18, input_frames=12,
                         upsample_factor=8, dropout=0.0)
    feats, _ = _make_inputs()
    rain, pwv = dec(feats, encoder_skips=None)
    assert rain.shape == (1, 18, 256, 256)
    assert pwv.shape == (1, 18, 256, 256)
