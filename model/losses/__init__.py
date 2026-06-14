"""Pluvian loss modules: physics (water budget) + data (MSE/BMAE/FSS/focal)."""
from .water_budget import (
    SphericalDivergence,
    TimeDerivative,
    GaussianSmooth2d,
    WaterBudgetLoss,
)
from .data_losses import (
    WeightedMSE,
    BMAE,
    FSSProxy,
    FocalRainLoss,
    SpectralLoss,
)
from .tweedie import TweedieDevianceLoss

__all__ = [
    "SphericalDivergence",
    "TimeDerivative",
    "GaussianSmooth2d",
    "WaterBudgetLoss",
    "WeightedMSE",
    "BMAE",
    "FSSProxy",
    "FocalRainLoss",
    "SpectralLoss",
    "TweedieDevianceLoss",
]
