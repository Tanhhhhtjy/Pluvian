"""Unit tests for Phase 7a loss fixes.

Run with:
    /ssd-disk/lih/miniconda3/bin/python -m pytest tests/test_losses.py -xvs
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from model.losses import (  # noqa: E402
    FocalRainLoss,
    FSSProxy,
    TweedieDevianceLoss,
    WeightedMSE,
)
from scripts.train import _crps_ensemble  # noqa: E402


# ---------------------------------------------------------------------------
# Tweedie deviance — sanity values at y = 0 / 5 / 50
# ---------------------------------------------------------------------------

def _tweedie_reference(y: float, mu: float, p: float = 1.5) -> float:
    """Closed-form Tweedie deviance (unreduced) for scalar inputs."""
    term2 = mu ** (2.0 - p) / (2.0 - p)
    term1 = y * mu ** (1.0 - p) / (1.0 - p)
    return 2.0 * (term2 - term1)


@pytest.mark.parametrize("y,pred,expected_sign", [
    (0.0,  0.0,  "small"),     # pred near 0 floor -> deviance small (~0)
    (0.0, 10.0,  "positive"),  # overshoot when truth is 0
    (5.0,  5.0,  "small"),     # well-calibrated
    (5.0,  0.0,  "positive"),  # under-shoot
    (50.0, 5.0,  "positive"),  # gross under-shoot of extreme
    (50.0, 50.0, "small"),
])
def test_tweedie_sanity(y, pred, expected_sign):
    loss = TweedieDevianceLoss(p=1.5)
    y_t = torch.tensor([[y]])
    p_t = torch.tensor([[pred]])
    val = loss(p_t, y_t).item()
    assert math.isfinite(val), f"non-finite deviance at y={y}, pred={pred}"
    if expected_sign == "positive":
        assert val > 0.01, f"expected positive deviance, got {val}"
    if expected_sign == "small":
        # well-calibrated case still has tiny floor due to softplus offset
        assert val < 5.0, f"expected small deviance, got {val}"


def test_tweedie_pred_le_zero_safe():
    """Strongly negative network outputs (-50) should still give finite loss."""
    loss = TweedieDevianceLoss(p=1.5)
    y = torch.tensor([[0.0, 5.0, 50.0]])
    pred = torch.tensor([[-50.0, -10.0, -3.0]])
    val = loss(pred, y).item()
    assert math.isfinite(val), f"loss exploded on negative preds: {val}"


def test_tweedie_grad_flows():
    pred = torch.tensor([0.3, 1.0, 5.0], requires_grad=True)
    y = torch.tensor([0.0, 3.0, 50.0])
    loss = TweedieDevianceLoss(p=1.5)(pred, y)
    loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()


# ---------------------------------------------------------------------------
# 4-member CRPS — match independent reference implementation
# ---------------------------------------------------------------------------

def _crps_brute(samples: torch.Tensor, target: torch.Tensor) -> float:
    """Brute-force CRPS via nested loops over members (correct, slow)."""
    K = samples.shape[0]
    flat = samples.reshape(K, -1)
    y = target.reshape(-1)
    N = y.numel()
    mae = 0.0
    for i in range(K):
        mae += (flat[i] - y).abs().mean().item()
    mae /= K
    pw = 0.0
    for i in range(K):
        for j in range(K):
            pw += (flat[i] - flat[j]).abs().mean().item()
    pw /= (K * K)
    return mae - 0.5 * pw


def test_crps_ensemble_matches_brute_force():
    torch.manual_seed(0)
    samples = torch.randn(4, 2, 3, 5, 7)
    target = torch.randn(2, 3, 5, 7)
    val = _crps_ensemble(samples, target)
    ref = _crps_brute(samples, target)
    assert abs(val - ref) < 1e-5, f"got {val}, ref {ref}"


def test_crps_degenerate_equals_mae():
    """K=1 ensemble: CRPS == mean(|x - y|) (the old behavior)."""
    pred = torch.randn(2, 3, 4, 5)
    target = torch.randn(2, 3, 4, 5)
    samples = pred.unsqueeze(0)
    crps = _crps_ensemble(samples, target)
    mae = (pred - target).abs().mean().item()
    assert abs(crps - mae) < 1e-6


def test_crps_perfect_ensemble_is_zero():
    target = torch.randn(2, 4, 8, 8)
    # 4 identical samples == target -> CRPS should be 0
    samples = torch.stack([target] * 4, dim=0)
    val = _crps_ensemble(samples, target)
    assert abs(val) < 1e-6


# ---------------------------------------------------------------------------
# FocalRainLoss + FSS — single-batch forward smoke
# ---------------------------------------------------------------------------

def test_focal_rain_loss_forward():
    pred = torch.randn(2, 3, 16, 16) * 5.0 + 5.0
    target = (torch.rand(2, 3, 16, 16) * 20.0).clamp(min=0)
    loss = FocalRainLoss(rain_threshold=10.0, alpha=0.75, gamma=2.0)
    val = loss(pred, target)
    assert val.dim() == 0
    assert torch.isfinite(val).item()
    # gradient
    pred2 = pred.detach().requires_grad_(True)
    loss(pred2, target).backward()
    assert torch.isfinite(pred2.grad).all()


def test_fss_proxy_multi_threshold():
    pred = torch.rand(2, 3, 16, 16) * 30.0
    target = torch.rand(2, 3, 16, 16) * 30.0
    for thr in [1.0, 10.0, 30.0]:
        f = FSSProxy(threshold=thr, window=9)
        v = f(pred, target).item()
        assert math.isfinite(v)
        assert 0.0 <= v <= 1.0 + 1e-3


# ---------------------------------------------------------------------------
# Decoder dropout / Pluvian.mc_dropout_predict — varies outputs
# ---------------------------------------------------------------------------

def test_mc_dropout_predict_produces_varied_samples():
    """In eval mode, mc_dropout_predict should still produce varying outputs."""
    from model.decoder import PluvianDecoder
    torch.manual_seed(0)
    dec = PluvianDecoder(dim=32, forecast_frames=3, input_frames=4,
                         upsample_factor=8, dropout=0.5)
    dec.eval()
    feats = torch.randn(1, 4, 32, 4, 4)
    a, _ = dec(feats)
    b, _ = dec(feats)
    # Without MC-dropout, two eval passes are identical
    assert torch.allclose(a, b)
    dec.enable_mc_dropout(True)
    a, _ = dec(feats)
    b, _ = dec(feats)
    # With MC-dropout, masks differ across passes
    assert not torch.allclose(a, b), "MC-dropout failed to randomize"
    dec.enable_mc_dropout(False)


# ---------------------------------------------------------------------------
# build_losses integration: FocalRainLoss is now wired
# ---------------------------------------------------------------------------

def test_build_losses_focal_rain_wired():
    from scripts.train import build_losses
    cfg = {
        "loss": {
            "data": {"type": "focal_rain", "weight": 1.0,
                     "focal_rain": {"threshold": 10.0, "alpha": 0.75,
                                    "gamma": 2.0, "sharpness": 5.0}},
            "fss": {"enabled": True, "thresholds": [1.0, 10.0],
                    "threshold_weights": [1.0, 0.1],
                    "window": 9, "weight": 0.1},
            "budget": {"enabled": False, "weight": 0.0,
                       "delta_p_pa": 30000.0, "smooth_sigma": 2.0,
                       "huber_delta": 1e-4, "level_idx": 1},
        }
    }
    losses = build_losses(cfg)
    assert isinstance(losses["data"], FocalRainLoss)
    assert len(losses["fss"]) == 2
    pred = torch.randn(2, 3, 16, 16)
    target = (torch.rand(2, 3, 16, 16) * 20.0).clamp(min=0)
    v = losses["data"](pred, target)
    assert torch.isfinite(v).item()


def test_build_losses_tweedie_wired():
    from scripts.train import build_losses
    cfg = {
        "loss": {
            "data": {"type": "tweedie", "weight": 1.0,
                     "tweedie": {"p": 1.5, "eps": 1e-6}},
            "fss": {"enabled": False, "thresholds": [1.0],
                    "threshold_weights": [1.0], "window": 9, "weight": 0.1},
            "budget": {"enabled": False, "weight": 0.0,
                       "delta_p_pa": 30000.0, "smooth_sigma": 2.0,
                       "huber_delta": 1e-4, "level_idx": 1},
        }
    }
    losses = build_losses(cfg)
    assert isinstance(losses["data"], TweedieDevianceLoss)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-xvs"]))
