"""Phase 7d Tier-1: stochastic latent head (CVAE) for de-blurring.

The frozen Pluvian backbone predicts the *conditional mean* of all plausible
futures, which is smooth by construction. To recover sharp, realistic small-
scale structure (incl. intense echo cores) we must *sample one future instead
of averaging*. This head injects a sampled latent ``z`` into the fused
bottleneck via FiLM (per-channel affine), so a single decode commits to one
realisation.

Training (CVAE / SVG-style):
  * a posterior encoder q(z | future GT frames) sees the truth and emits a
    Gaussian; we reparam-sample z, FiLM the bottleneck, decode, and minimise
    reconstruction (+spectral) + beta * KL(q || N(0,I)).
  * only this head trains; the backbone (encoder+fusion+decoder) is frozen,
    so gradients flow through the decoder but update nothing in it.

Inference:
  * z ~ N(0, I) gives genuinely distinct sharp samples; report the ensemble
    mean for CSI and individual samples for realism (RAPSD / CRPS).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PosteriorEncoder(nn.Module):
    """q(z | future GT frames). Input (B, T_out, H, W) -> (mu, logvar) (B, z_dim)."""

    def __init__(self, t_out: int = 18, z_dim: int = 64, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(t_out, base, 4, stride=2, padding=1), nn.GroupNorm(8, base), nn.GELU(),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.GroupNorm(8, base * 2), nn.GELU(),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.GroupNorm(8, base * 4), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(base * 4, 2 * z_dim)
        self.z_dim = z_dim

    def forward(self, future: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(future).flatten(1)
        mu, logvar = self.fc(h).chunk(2, dim=-1)
        logvar = logvar.clamp(-8.0, 8.0)
        return mu, logvar


class FiLMGen(nn.Module):
    """z -> per-channel (gamma, beta) for the fused bottleneck (C channels)."""

    def __init__(self, z_dim: int, n_ch: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * n_ch),
        )
        self.n_ch = n_ch
        # init last layer ~0 so the head starts as identity (gamma=0, beta=0)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.net(z).chunk(2, dim=-1)
        return gamma, beta


class LatentSharpHead(nn.Module):
    """Posterior encoder + FiLM generator. Trains on a frozen backbone."""

    def __init__(self, n_ch: int = 192, z_dim: int = 64, t_out: int = 18):
        super().__init__()
        self.z_dim = z_dim
        self.post = PosteriorEncoder(t_out=t_out, z_dim=z_dim)
        self.film = FiLMGen(z_dim=z_dim, n_ch=n_ch)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    @staticmethod
    def kl(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

    def sample_posterior(self, future: torch.Tensor):
        mu, logvar = self.post(future)
        z = self.reparam(mu, logvar)
        return z, self.kl(mu, logvar)

    def sample_prior(self, batch: int, device) -> torch.Tensor:
        return torch.randn(batch, self.z_dim, device=device)

    def apply_film(self, fused: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # fused: (B, T_in, C, h, w); gamma/beta: (B, C)
        gamma, beta = self.film(z)
        g = gamma[:, None, :, None, None]
        b = beta[:, None, :, None, None]
        return fused * (1.0 + g) + b
