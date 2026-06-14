"""Phase 7d Tier-2: DiffCast-style residual diffusion head.

The frozen Pluvian backbone is an excellent *deterministic* predictor but its
MSE/regression objective drives it to the conditional mean -> blur, intense
echo cores averaged away. DiffCast (Yu et al., NeurIPS 2024) decomposes a
forecast into a global deterministic part + a local stochastic residual modeled
by a conditional diffusion process. We do exactly that:

    target = mu(backbone, frozen)  +  residual
    residual ~ p_theta(r | mu)            (learned by DDPM, predict-eps)

Sampling draws a *fresh* residual field -> sharp, physically-plausible cores
that are NOT averaged out, while mu keeps the storm correctly located. We model
the residual per-frame (each of the 18 lead frames is an independent 2D field
conditioned on its own mu frame); this keeps the denoiser small and the 180x180
spatial resolution affordable. DDIM (Song et al. 2021) gives fast 20-50 step
sampling at inference.

This module is self-contained: a conditional eps-UNet + a GaussianDiffusion
helper (cosine schedule, q_sample, p_losses, ddim_sample). Train/eval wiring
lives in scripts/train_diffusion.py and scripts/plot_diffusion_samples.py.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  conditional eps-UNet                                                         #
# --------------------------------------------------------------------------- #
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):  # t: (B,) float
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / max(1, half - 1)
        )
        a = t.float()[:, None] * freqs[None, :]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class ResBlock(nn.Module):
    """GroupNorm + SiLU conv block with additive time embedding (FiLM-free,
    simple shift) and a residual connection."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class CondEpsUNet(nn.Module):
    """Predicts the diffusion noise eps for a noisy residual frame, conditioned
    on the backbone-mean frame (channel-concatenated) and the timestep t.

    in  : x_t (B,1,H,W)  +  cond mu (B,1,H,W)  -> 2 in-channels
    out : eps (B,1,H,W)
    """

    def __init__(self, base: int = 64, t_dim: int = 256, cond_ch: int = 1):
        super().__init__()
        self.t_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim), nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        c1, c2, c3 = base, base * 2, base * 4
        self.in_conv = nn.Conv2d(1 + cond_ch, c1, 3, padding=1)
        # down
        self.d1 = ResBlock(c1, c1, t_dim)
        self.dn1 = nn.Conv2d(c1, c1, 4, 2, 1)
        self.d2 = ResBlock(c1, c2, t_dim)
        self.dn2 = nn.Conv2d(c2, c2, 4, 2, 1)
        self.d3 = ResBlock(c2, c3, t_dim)
        self.dn3 = nn.Conv2d(c3, c3, 4, 2, 1)
        # mid
        self.m1 = ResBlock(c3, c3, t_dim)
        self.m2 = ResBlock(c3, c3, t_dim)
        # up
        self.up3 = nn.ConvTranspose2d(c3, c3, 4, 2, 1)
        self.u3 = ResBlock(c3 + c3, c2, t_dim)
        self.up2 = nn.ConvTranspose2d(c2, c2, 4, 2, 1)
        self.u2 = ResBlock(c2 + c2, c1, t_dim)
        self.up1 = nn.ConvTranspose2d(c1, c1, 4, 2, 1)
        self.u1 = ResBlock(c1 + c1, c1, t_dim)
        self.out_norm = nn.GroupNorm(min(8, c1), c1)
        self.out_conv = nn.Conv2d(c1, 1, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_t, cond, t):
        temb = self.t_mlp(t)
        h = self.in_conv(torch.cat([x_t, cond], dim=1))
        s1 = self.d1(h, temb)
        h = self.dn1(s1)
        s2 = self.d2(h, temb)
        h = self.dn2(s2)
        s3 = self.d3(h, temb)
        h = self.dn3(s3)
        h = self.m2(self.m1(h, temb), temb)
        h = self.up3(h)
        h = self.u3(torch.cat([h, s3], dim=1), temb)
        h = self.up2(h)
        h = self.u2(torch.cat([h, s2], dim=1), temb)
        h = self.up1(h)
        h = self.u1(torch.cat([h, s1], dim=1), temb)
        return self.out_conv(F.silu(self.out_norm(h)))


# --------------------------------------------------------------------------- #
#  Gaussian diffusion (predict-eps, cosine schedule, DDIM sampling)            #
# --------------------------------------------------------------------------- #
def _cosine_alpha_bar(T: int, s: float = 0.008):
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    ab = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    return ab.clamp(1e-6, 1.0)  # (T+1,)


class GaussianDiffusion(nn.Module):
    """Minimal DDPM with cosine schedule. Operates on a residual field already
    normalised to ~unit scale by the caller. Buffers registered so .to(device)
    moves the schedule."""

    def __init__(self, model: CondEpsUNet, timesteps: int = 1000):
        super().__init__()
        self.model = model
        self.T = timesteps
        ab = _cosine_alpha_bar(timesteps)          # (T+1,)
        alpha_bar = ab[1:].float()                 # (T,)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_ab", alpha_bar.sqrt())
        self.register_buffer("sqrt_1m_ab", (1 - alpha_bar).sqrt())

    def q_sample(self, x0, t, noise):
        return self.sqrt_ab[t][:, None, None, None] * x0 + \
            self.sqrt_1m_ab[t][:, None, None, None] * noise

    def p_losses(self, x0, cond):
        """Standard eps-prediction MSE. x0,cond: (N,1,H,W)."""
        N = x0.shape[0]
        t = torch.randint(0, self.T, (N,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps = self.model(x_t, cond, t.float())
        return F.mse_loss(eps, noise)

    @torch.no_grad()
    def ddim_sample(self, cond, n_steps: int = 30, eta: float = 0.0, shape=None):
        """DDIM sampling. cond: (N,1,H,W). Returns x0_hat residual (N,1,H,W)."""
        device = cond.device
        N = cond.shape[0]
        H, W = (cond.shape[-2], cond.shape[-1]) if shape is None else shape
        x = torch.randn(N, 1, H, W, device=device)
        ts = torch.linspace(self.T - 1, 0, n_steps, dtype=torch.long, device=device)
        for i in range(n_steps):
            t = ts[i]
            tb = t.repeat(N)
            ab_t = self.alpha_bar[t]
            eps = self.model(x, cond, tb.float())
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            x0 = x0.clamp(-4, 4)
            if i == n_steps - 1:
                x = x0
                break
            t_next = ts[i + 1]
            ab_n = self.alpha_bar[t_next]
            sigma = eta * ((1 - ab_n) / (1 - ab_t) * (1 - ab_t / ab_n)).clamp(min=0).sqrt()
            dir_xt = (1 - ab_n - sigma ** 2).clamp(min=0).sqrt() * eps
            x = ab_n.sqrt() * x0 + dir_xt + sigma * torch.randn_like(x)
        return x


def pad_to_multiple(x, m: int = 8):
    """Pad last two dims up to a multiple of m (reflect). Returns padded, (H,W)."""
    H, W = x.shape[-2:]
    ph, pw = (m - H % m) % m, (m - W % m) % m
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    return x, (H, W)
