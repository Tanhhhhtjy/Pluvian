"""Tweedie deviance loss for precipitation regression.

Following Hunt (2025) "Stop using RMSE for precipitation forecasting"
(arXiv:2509.08369): Tweedie compound Poisson-Gamma deviance better matches
the zero-inflated heavy-tailed distribution of rain rate than MSE / MAE.

For p in (1, 2) (commonly 1.5), the (unit) deviance is::

    d(y, mu) = 2 * [ y^(2-p) / ((1-p)(2-p))
                    - y * mu^(1-p) / (1-p)
                    + mu^(2-p) / (2-p) ]

Special-cased at y = 0 to::

    d(0, mu) = 2 * mu^(2-p) / (2-p)

Because mu^(1-p) and mu^(2-p) are undefined for mu <= 0, predictions are
mapped through softplus(.) + eps before being plugged into the formula.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TweedieDevianceLoss(nn.Module):
    """Tweedie deviance loss for precipitation (mm / h).

    Parameters
    ----------
    p   : Tweedie power parameter in (1, 2). 1.5 is the common compound
          Poisson-Gamma default and what the Hunt 2025 paper recommends.
    eps : floor added after softplus(pred) so the deviance is finite when
          the network outputs very negative numbers (softplus saturates
          smoothly to 0).
    reduction : ``mean`` | ``sum`` | ``none``.
    """

    def __init__(self, p: float = 1.5, eps: float = 1e-6,
                 reduction: str = "mean"):
        super().__init__()
        if not (1.0 < p < 2.0):
            raise ValueError(f"Tweedie p must be in (1, 2), got {p}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(reduction)
        self.p = float(p)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        mu = F.softplus(pred) + self.eps
        y = target.clamp(min=0.0)
        p = self.p
        # Saturated deviance for Tweedie (1 < p < 2):
        #   d(y, mu) = 2 * [ y^(2-p) / ((2-p)(1-p))
        #                   - y * mu^(1-p) / (1-p)
        #                   + mu^(2-p) / (2-p) ]
        # The y^(2-p) term is the saturated likelihood — it does not depend on
        # mu so does not change the gradient, but it makes d(y, mu) = 0 at the
        # optimum mu = y and keeps the loss interpretable / non-negative.
        # At y = 0 it vanishes via the floor below.
        term_mu_2 = torch.pow(mu, 2.0 - p) / (2.0 - p)
        term_y_mu = y * torch.pow(mu, 1.0 - p) / (1.0 - p)
        # y^(2-p) is fine for y >= 0 (0 maps to 0 since 2-p > 0)
        term_y_sat = torch.pow(y.clamp(min=0.0), 2.0 - p) / ((2.0 - p) * (1.0 - p))
        dev = 2.0 * (term_y_sat - term_y_mu + term_mu_2)
        if self.reduction == "sum":
            return dev.sum()
        if self.reduction == "none":
            return dev
        return dev.mean()
