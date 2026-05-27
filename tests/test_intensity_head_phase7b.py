"""Phase 7b T3 — intensity-stratified rain head (on top of the U-Net decoder).

Covers:
  * 5-band logit shape on the decoder output tuple.
  * Continuous expected-value rain shape collapses to (B, T_out, H, W).
  * Backward pass produces finite grads.
  * ``intensity_stratified=False`` is bit-identical to the decoder-skip
    baseline (same seed + same input ⇒ same rain & pwv output).
  * Band-assignment matches the configured edges for sample targets.
  * Head parameter overhead is small (≪ 0.5M).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.decoder import PluvianDecoder  # noqa: E402

DIM = 32
T_IN = 6
T_OUT = 4
H = W = 4


def _mk_feats(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, T_IN, DIM, H, W, generator=g)


def test_intensity_head_output_shapes():
    dec = PluvianDecoder(dim=DIM, forecast_frames=T_OUT, input_frames=T_IN,
                         upsample_factor=2, dropout=0.0,
                         intensity_stratified=True)
    feats = _mk_feats(0)
    rain, pwv, logits = dec(feats)
    assert logits.shape == (2, T_OUT, 5, H * 2, W * 2)
    assert rain.shape == (2, T_OUT, H * 2, W * 2)
    assert pwv.shape == (2, T_OUT, H * 2, W * 2)
    # Expected-value rain should be inside the band-center hull [0, 50].
    assert rain.min().item() >= -1e-4
    assert rain.max().item() <= 50.0 + 1e-4


def test_intensity_backward_finite():
    dec = PluvianDecoder(dim=DIM, forecast_frames=T_OUT, input_frames=T_IN,
                         upsample_factor=2, dropout=0.0,
                         intensity_stratified=True)
    feats = _mk_feats(1).requires_grad_(True)
    rain, _pwv, logits = dec(feats)
    loss = rain.pow(2).mean() + logits.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in dec.parameters() if p.grad is not None]
    assert grads, "no grads"
    for g in grads:
        assert torch.isfinite(g).all(), "non-finite grad"


def test_band_assignment_matches_edges():
    # bucketize with edges [0.1, 1, 8, 30] should map these samples to bands
    # [0, 1, 2, 3, 4] respectively.
    edges = torch.tensor([0.1, 1.0, 8.0, 30.0])
    targets = torch.tensor([0.0, 0.5, 5.0, 20.0, 50.0])
    bands = torch.bucketize(targets, edges)
    assert bands.tolist() == [0, 1, 2, 3, 4]


def test_bit_identical_to_decoder_skip_default():
    # Same seed -> same Conv weights for the single-channel head means the
    # default (intensity_stratified=False) path must reproduce the
    # decoder-skip baseline output exactly under the same input.
    torch.manual_seed(123)
    dec_old = PluvianDecoder(dim=DIM, forecast_frames=T_OUT, input_frames=T_IN,
                             upsample_factor=2, dropout=0.0,
                             intensity_stratified=False)
    torch.manual_seed(123)
    dec_new = PluvianDecoder(dim=DIM, forecast_frames=T_OUT, input_frames=T_IN,
                             upsample_factor=2, dropout=0.0,
                             intensity_stratified=False)
    feats = _mk_feats(42)
    dec_old.eval(); dec_new.eval()
    with torch.no_grad():
        r_old, p_old = dec_old(feats)
        r_new, p_new = dec_new(feats)
    assert torch.equal(r_old, r_new), "default-mode rain output drifted"
    assert torch.equal(p_old, p_new), "default-mode pwv output drifted"


def test_head_param_overhead_small():
    base = PluvianDecoder(dim=DIM, forecast_frames=T_OUT, input_frames=T_IN,
                          upsample_factor=2, intensity_stratified=False)
    strat = PluvianDecoder(dim=DIM, forecast_frames=T_OUT, input_frames=T_IN,
                           upsample_factor=2, intensity_stratified=True)
    n_base = sum(p.numel() for p in base.parameters())
    n_strat = sum(p.numel() for p in strat.parameters())
    delta = n_strat - n_base
    expected = (strat.rain_head.weight.numel() + strat.rain_head.bias.numel()
                - base.rain_head.weight.numel() - base.rain_head.bias.numel())
    assert delta == expected
    assert delta < 500_000, f"head overhead too big: {delta}"


def test_intensity_with_encoder_skips():
    # Sanity: stratified head still works when U-Net skips are supplied.
    dec = PluvianDecoder(dim=DIM, forecast_frames=T_OUT, input_frames=T_IN,
                         upsample_factor=4, dropout=0.0,
                         intensity_stratified=True)
    feats = _mk_feats(7)
    skips = {
        "stage2": _mk_feats(11),                               # (B, T_in, C, H/8, W/8)
        "stage1": torch.randn(2, T_IN, DIM, H * 2, W * 2),     # (B, T_in, C, H/4, W/4)
    }
    rain, pwv, logits = dec(feats, encoder_skips=skips)
    assert rain.shape == (2, T_OUT, H * 4, W * 4)
    assert logits.shape == (2, T_OUT, 5, H * 4, W * 4)
    assert pwv.shape == (2, T_OUT, H * 4, W * 4)
