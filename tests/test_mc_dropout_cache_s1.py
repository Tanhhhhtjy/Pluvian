"""Phase 7b audit S1: val loop must call _encode_and_fuse exactly once per
batch — not twice (deterministic forward + mc_dropout_predict was a 2x).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.pluvian import Pluvian


def _build():
    return Pluvian(
        radar_channels=1, pwv_enabled=False, pwv_concat_only=False,
        era5_enabled=False, mfd_channel_enabled=False,
        forecast_frames=4, input_frames=4,
        hidden_dim=32, n_layers=2, n_heads=4,
        gated_fusion=False, intensity_stratified=False,
    )


def _fake_batch(B=1, T=4, H=24, W=24):
    return {
        "radar": torch.zeros(B, T, H, W),
        "pwv_grid": torch.zeros(B, T, 176),
        "pwv_mask": torch.zeros(B, T, 176),
        "pwv_coords": torch.zeros(B, 176, 2),
        "station_grid": torch.zeros(B, T, 289, 6),
        "station_mask": torch.zeros(B, T, 289),
        "station_coords": torch.zeros(B, 289, 3),
    }


def test_mc_dropout_from_fused_no_encoder_recompute():
    model = _build().eval()
    batch = _fake_batch()
    calls = [0]
    real = model._encode_and_fuse

    def counting(*a, **kw):
        calls[0] += 1
        return real(*a, **kw)

    model._encode_and_fuse = counting

    # Simulate the new val-loop pattern
    fused, skips = model._encode_and_fuse(batch, return_skips=True)
    _ = model.decoder(fused, encoder_skips=skips)
    samples = model.mc_dropout_predict_from_fused(fused, skips, n_samples=3)

    assert calls[0] == 1, f"encoder called {calls[0]} times, expected 1"
    assert samples.shape[0] == 3


def test_mc_dropout_from_fused_matches_full_path_shape():
    model = _build().eval()
    batch = _fake_batch()
    samples_full = model.mc_dropout_predict(batch, n_samples=2)
    fused, skips = model._encode_and_fuse(batch, return_skips=True)
    samples_cached = model.mc_dropout_predict_from_fused(fused, skips, n_samples=2)
    assert samples_full.shape == samples_cached.shape
