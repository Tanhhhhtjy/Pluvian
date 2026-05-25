"""Data-fidelity losses for Pluvian: weighted MSE, BMAE, differentiable FSS,
focal-rain loss.  All are nn.Modules so they can be composed with the physics
loss inside a training step.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _intensity_weight(rain_true: torch.Tensor) -> torch.Tensor:
    """Piecewise weight by ground-truth rain rate (mm/h).

    <1 mm/h    : 1.0
    1-10 mm/h  : 1.0
    10-30 mm/h : 3.0
    >30 mm/h   : 5.0
    """
    w = torch.ones_like(rain_true)
    w = torch.where(rain_true > 10.0, torch.full_like(w, 3.0), w)
    w = torch.where(rain_true > 30.0, torch.full_like(w, 5.0), w)
    return w


class WeightedMSE(nn.Module):
    """Intensity-weighted MSE for rain-rate regression."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(reduction)
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = _intensity_weight(target)
        if mask is not None:
            w = w * mask.to(dtype=w.dtype)
        sq = (pred - target) ** 2 * w
        if self.reduction == "none":
            return sq
        if self.reduction == "sum":
            return sq.sum()
        denom = w.sum().clamp(min=1.0)
        return sq.sum() / denom


class BMAE(nn.Module):
    """Balanced MAE: equal weight to wet vs dry pixels using a rain threshold."""

    def __init__(self, rain_threshold: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.rain_threshold = float(rain_threshold)
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(reduction)
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        wet = (target > self.rain_threshold).to(dtype=pred.dtype)
        dry = 1.0 - wet
        if mask is not None:
            m = mask.to(dtype=pred.dtype)
            wet = wet * m
            dry = dry * m
        err = (pred - target).abs()
        n_wet = wet.sum().clamp(min=1.0)
        n_dry = dry.sum().clamp(min=1.0)
        wet_loss = (err * wet).sum() / n_wet
        dry_loss = (err * dry).sum() / n_dry
        per_pixel = 0.5 * (wet_loss + dry_loss)
        if self.reduction == "sum":
            return per_pixel * (n_wet + n_dry)
        if self.reduction == "none":
            # broadcast a per-class weight back so the caller can mask
            w = 0.5 * (wet / n_wet + dry / n_dry)
            return err * w
        return per_pixel


class FSSProxy(nn.Module):
    """Differentiable Fractions Skill Score (FSS) proxy.

    Standard FSS uses uniform-box neighbourhood fraction of pixels exceeding a
    threshold.  Here we replace the hard threshold with a sigmoid so gradients
    flow:

        f(x) = sigmoid(k * (x - thresh))

    and average f(x) over a (window x window) box using F.avg_pool2d.  The
    returned loss is `1 - FSS`, so lower is better.

    Inputs are (B, T, H, W); pooling is applied on (H, W) per (B*T) slice.
    """

    def __init__(self,
                 threshold: float = 1.0,
                 window: int = 9,
                 sharpness: float = 5.0):
        super().__init__()
        if window < 1 or window % 2 == 0:
            raise ValueError("window must be odd >=1")
        self.threshold = float(threshold)
        self.window = int(window)
        self.sharpness = float(sharpness)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W) -> (B*T, 1, H, W) -> avg pool -> back
        B, T, H, W = x.shape
        x2 = x.reshape(B * T, 1, H, W)
        pad = self.window // 2
        x2 = F.avg_pool2d(x2, kernel_size=self.window, stride=1, padding=pad,
                          count_include_pad=False)
        return x2.reshape(B, T, H, W)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        if pred.dim() != 4:
            raise ValueError(f"expected (B,T,H,W), got {pred.shape}")
        f_pred = torch.sigmoid(self.sharpness * (pred - self.threshold))
        f_targ = torch.sigmoid(self.sharpness * (target - self.threshold))
        p_pred = self._pool(f_pred)
        p_targ = self._pool(f_targ)
        num = ((p_pred - p_targ) ** 2).mean()
        den = (p_pred ** 2 + p_targ ** 2).mean().clamp(min=1e-12)
        fss = 1.0 - num / den
        return 1.0 - fss  # loss: smaller is better


class FocalRainLoss(nn.Module):
    """Focal loss for rain/no-rain classification (uses a soft threshold).

    Treats pixels with rain rate > rain_threshold as positive.  The prediction
    is converted to a probability via a sigmoid on `(pred_mm_per_h - thresh)`
    scaled by `sharpness`.  Standard focal loss with `alpha`/`gamma`.
    """

    def __init__(self,
                 rain_threshold: float = 0.1,
                 sharpness: float = 5.0,
                 alpha: float = 0.75,
                 gamma: float = 2.0):
        super().__init__()
        self.rain_threshold = float(rain_threshold)
        self.sharpness = float(sharpness)
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        p = torch.sigmoid(self.sharpness * (pred - self.rain_threshold))
        y = (target > self.rain_threshold).to(dtype=pred.dtype)
        p = p.clamp(min=1e-6, max=1.0 - 1e-6)
        ce_pos = -torch.log(p)
        ce_neg = -torch.log(1.0 - p)
        loss = (self.alpha * (1.0 - p) ** self.gamma * y * ce_pos
                + (1.0 - self.alpha) * p ** self.gamma * (1.0 - y) * ce_neg)
        if mask is not None:
            loss = loss * mask.to(dtype=loss.dtype)
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom
        return loss.mean()
