"""Decoder: fused features → (T_out, H, W) rain & PWV predictions.

Phase 7b T1 refactor (audit Lever #1):
  * Replace Linear(T_in -> T_out) time projection with per-lead-time
    learnable queries cross-attending to fused T_in tokens (per spatial
    location). Each forecast frame attends to its own subset of inputs.
  * Add U-Net skip connections from the radar encoder at H/8 (stage2)
    and H/4 (stage1) so high-frequency structure survives the upsample stack.

Backward compatibility: passing ``encoder_skips=None`` runs the cross-attention
path without skips; old ckpts (with ``time_proj.weight``) will fail to load
strict, as designed — new training starts from scratch.

Dual heads:
  * ``rain_head``  → predicted dBZ-equivalent (or directly mm/h) future frames
  * ``pwv_head``   → predicted column water-vapour field

Phase 7b T3 (MTLDM-style intensity-stratified head):
When ``intensity_stratified=True`` the rain head emits ``n_bands`` logit
channels instead of one. The continuous rain map returned to downstream
losses/metrics is the softmax-weighted expectation over fixed band centers.
Default (``intensity_stratified=False``) is numerically bit-identical to the
pre-T3 decoder-skip baseline.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class _SkipFuse(nn.Module):
    """Fuse a concatenated (main, skip) feature map via 3x3 conv + GN + GELU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _LeadTimeCrossAttn(nn.Module):
    """Per-spatial-location cross-attention from T_out learnable queries to
    T_in fused tokens. Output: (B, T_out, C, H, W).
    """

    def __init__(self, dim: int, forecast_frames: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.forecast_frames = forecast_frames
        self.queries = nn.Parameter(torch.randn(forecast_frames, dim) * 0.02)
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T_in, C, H, W)
        B, T_in, C, H, W = feats.shape
        kv = rearrange(feats, "b t c h w -> (b h w) t c")
        kv = self.kv_norm(kv)
        q = self.queries.unsqueeze(0).expand(kv.shape[0], -1, -1)
        q = self.q_norm(q)
        out, _ = self.attn(q, kv, kv, need_weights=False)
        out = rearrange(out, "(b h w) t c -> b t c h w", b=B, h=H, w=W)
        return out


def _temporal_aggregate(skip: torch.Tensor, weights: nn.Linear) -> torch.Tensor:
    """Project skip (B, T_in, C, H, W) to (B, T_out, C, H, W) by linear mixing
    over the time axis."""
    B, T_in, C, H, W = skip.shape
    x = rearrange(skip, "b t c h w -> b c h w t")
    x = weights(x)
    return rearrange(x, "b c h w t -> b t c h w")


class PluvianDecoder(nn.Module):
    """Convert fused (B, T_in, C, H/8, W/8) features into two prediction fields.

    Parameters
    ----------
    dim         : hidden channel count (matches encoder dim)
    forecast_frames : number of frames to predict (T_out, typically 18)
    upsample_factor : total upsample factor (encoder downsamples 8x).
    """

    def __init__(self, dim: int = 192, forecast_frames: int = 18,
                 input_frames: int = 30, upsample_factor: int = 8,
                 dropout: float = 0.1, n_query_heads: int = 4,
                 skip_channels: int | None = None,
                 intensity_stratified: bool = False,
                 band_centers: Sequence[float] = (0.0, 0.5, 4.5, 19.0, 50.0)):
        super().__init__()
        self.forecast_frames = forecast_frames
        self.input_frames = input_frames
        self.dropout_p = float(dropout)
        self.dim = dim
        self.skip_channels = skip_channels if skip_channels is not None else dim

        self.lead_attn = _LeadTimeCrossAttn(dim, forecast_frames, n_query_heads)

        self.skip2_time = nn.Linear(input_frames, forecast_frames, bias=False)
        self.skip1_time = nn.Linear(input_frames, forecast_frames, bias=False)

        self.skip2_proj = nn.Conv2d(self.skip_channels, dim, 1)
        ch_after_up1 = max(dim // 2, 32)
        self.skip1_proj = nn.Conv2d(self.skip_channels, ch_after_up1, 1)

        self.fuse_stage2 = _SkipFuse(dim * 2, dim)
        self.fuse_stage1 = _SkipFuse(ch_after_up1 * 2, ch_after_up1)

        self.up_blocks = nn.ModuleList()
        ch = dim
        f = upsample_factor
        while f > 1:
            next_ch = max(ch // 2, 32)
            self.up_blocks.append(_UpBlock(ch, next_ch))
            ch = next_ch
            f //= 2
        self.last_ch = ch

        self.rain_head = nn.Conv2d(ch, 1, 1)
        self.pwv_head = nn.Conv2d(ch, 1, 1)
        self._force_dropout = False

        # Phase 7b T3 — optional MTLDM-style multi-band head.
        # Constructed AFTER the single-channel head so the default-mode RNG
        # consumption (and therefore weight init) matches the decoder-skip
        # baseline exactly; ``rain_head`` is reassigned only when stratified.
        self.intensity_stratified = bool(intensity_stratified)
        if self.intensity_stratified:
            self.n_bands = len(band_centers)
            self.rain_head = nn.Conv2d(ch, self.n_bands, 1)
            self.register_buffer(
                "band_centers",
                torch.tensor(tuple(band_centers), dtype=torch.float32),
            )
        else:
            self.n_bands = 1

    def enable_mc_dropout(self, flag: bool = True) -> None:
        """Toggle MC-dropout sampling. Idempotent."""
        self._force_dropout = bool(flag)

    def _maybe_resize_time(self, feats: torch.Tensor) -> torch.Tensor:
        B, T_in, C, h, w = feats.shape
        if T_in == self.input_frames:
            return feats
        x = rearrange(feats, "b t c h w -> b (c h w) t")
        x = F.interpolate(x, size=self.input_frames, mode="linear",
                          align_corners=False)
        return rearrange(x, "b (c h w) t -> b t c h w", c=C, h=h, w=w)

    def forward(self, feats: torch.Tensor,
                encoder_skips: dict | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # feats: (B, T_in, C, H/8, W/8)
        feats = self._maybe_resize_time(feats)
        B = feats.shape[0]

        # per-lead-time cross-attention
        x = self.lead_attn(feats)                          # (B, T_out, C, h, w)
        x = rearrange(x, "b t c h w -> (b t) c h w")

        if self.dropout_p > 0.0:
            x = F.dropout(x, p=self.dropout_p,
                          training=self.training or self._force_dropout)

        # skip 2 at H/8 before up_blocks[0]
        if encoder_skips is not None and "stage2" in encoder_skips:
            s2 = self._maybe_resize_time(encoder_skips["stage2"])
            s2 = _temporal_aggregate(s2, self.skip2_time)
            s2 = rearrange(s2, "b t c h w -> (b t) c h w")
            s2 = self.skip2_proj(s2)
            x = torch.cat([x, s2], dim=1)
            x = self.fuse_stage2(x)

        x = self.up_blocks[0](x)

        # skip 1 at H/4 before up_blocks[1]
        if (len(self.up_blocks) > 1 and encoder_skips is not None
                and "stage1" in encoder_skips):
            s1 = self._maybe_resize_time(encoder_skips["stage1"])
            s1 = _temporal_aggregate(s1, self.skip1_time)
            s1 = rearrange(s1, "b t c h w -> (b t) c h w")
            s1 = self.skip1_proj(s1)
            x = torch.cat([x, s1], dim=1)
            x = self.fuse_stage1(x)

        for blk in self.up_blocks[1:]:
            x = blk(x)

        rain_out = self.rain_head(x)
        pwv = self.pwv_head(x)
        pwv = rearrange(pwv, "(b t) () h w -> b t h w",
                        b=B, t=self.forecast_frames)
        if self.intensity_stratified:
            logits = rearrange(rain_out, "(b t) k h w -> b t k h w",
                               b=B, t=self.forecast_frames)
            # Phase 7b audit S3: bf16 softmax + (probs*centers) where the top
            # band centre is 50 mm/h gives ~0.4 mm quantization noise — that
            # is the same order as the CSI@30mm decision boundary. Cast to
            # fp32 for the expectation so the head topology is not what we
            # are measuring at the high-intensity end. F.cross_entropy already
            # upcasts internally, so logits stay in their autocast dtype.
            probs = F.softmax(logits, dim=2).float()
            centers = self.band_centers.float().view(1, 1, -1, 1, 1)
            rain = (probs * centers).sum(dim=2)
            return rain, pwv, logits
        rain = rearrange(rain_out, "(b t) () h w -> b t h w",
                         b=B, t=self.forecast_frames)
        return rain, pwv
