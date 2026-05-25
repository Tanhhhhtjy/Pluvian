"""Decoder: fused features → (T_out, H, W) rain & PWV predictions.

We aggregate the input-time axis into a per-frame "forecast bank" with a
small temporal projection (so the decoder doesn't need to be recurrent),
then upsample 8x with two PixelShuffle stages and split into two heads.

Dual heads:
  * ``rain_head``  → predicted dBZ-equivalent (or directly mm/h) future frames
  * ``pwv_head``   → predicted column water-vapour field (used by the
                     water-budget physical loss in T4.2)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange


class _UpBlock(nn.Module):
    """PixelShuffle x2 + conv + GN + GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, 3, padding=1),
            nn.PixelShuffle(2),
        )
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.up(x)))


class PluvianDecoder(nn.Module):
    """Convert fused (B, T_in, C, H/8, W/8) features into two prediction fields.

    Parameters
    ----------
    dim         : hidden channel count (matches encoder dim)
    forecast_frames : number of frames to predict (T_out, typically 18)
    upsample_factor : total upsample factor (encoder downsamples 8x).
    """

    def __init__(self, dim: int = 192, forecast_frames: int = 18,
                 input_frames: int = 30, upsample_factor: int = 8):
        super().__init__()
        self.forecast_frames = forecast_frames
        self.input_frames = input_frames
        # learn a (T_in -> T_out) projection on the time axis. A simple
        # Linear keeps the parameter count negligible (~30*18 = 540 weights)
        # while letting the decoder pick which input frames matter for each
        # forecast horizon.
        self.time_proj = nn.Linear(input_frames, forecast_frames, bias=True)

        ups = []
        ch = dim
        n_ups = 0
        f = upsample_factor
        while f > 1:
            next_ch = max(ch // 2, 32)
            ups.append(_UpBlock(ch, next_ch))
            ch = next_ch
            f //= 2
            n_ups += 1
        self.up = nn.Sequential(*ups)
        self.last_ch = ch
        self.rain_head = nn.Conv2d(ch, 1, 1)
        self.pwv_head = nn.Conv2d(ch, 1, 1)

    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # feats: (B, T_in, C, H/8, W/8)
        B, T_in, C, h, w = feats.shape
        if T_in != self.input_frames:
            # Resample along time to the expected input length so the
            # Linear projection still applies. Linear interp is fine — it
            # just keeps the decoder agnostic to mild T_in changes.
            x = rearrange(feats, "b t c h w -> b (c h w) t")
            x = F.interpolate(x, size=self.input_frames, mode="linear",
                              align_corners=False)
            feats = rearrange(x, "b (c h w) t -> b t c h w", c=C, h=h, w=w)
        x = rearrange(feats, "b t c h w -> b c h w t")
        x = self.time_proj(x)                                     # (B, C, H, W, T_out)
        x = rearrange(x, "b c h w t -> (b t) c h w")
        # Decoder upsample is the heaviest activation in the model
        # (going from H/8 W/8 to H W). Checkpoint each stage in training
        # to trade ~10% compute for ~3x activation-memory savings.
        if self.training and torch.is_grad_enabled():
            for stage in self.up:
                x = torch.utils.checkpoint.checkpoint(stage, x, use_reentrant=False)
        else:
            x = self.up(x)                                        # (B*T_out, C', H, W)
        rain = self.rain_head(x)                                  # (B*T_out, 1, H, W)
        pwv = self.pwv_head(x)
        rain = rearrange(rain, "(b t) () h w -> b t h w",
                         b=B, t=self.forecast_frames)
        pwv = rearrange(pwv, "(b t) () h w -> b t h w",
                        b=B, t=self.forecast_frames)
        return rain, pwv
