"""Encoders for the three input modalities.

Design notes
------------
* **Radar** is the heaviest input: (B, T=30, H, W) on the 0.01° grid.
  We stride-down with a Conv2d patch stem (factor 4) and one further factor-2
  downsample, so the latent grid is H/8 × W/8 — small enough that a couple
  of windowed-attention blocks fit comfortably on A800 40GB.
* **ERA5** is 4 vars × 8 levels × T frames. Even on the 49×49 native grid
  this is 32 channels per frame; we collapse vars/levels into the channel
  dim, run a small 2D conv tower, and *up*-sample at the fusion stage —
  not here — to keep the heavy stuff on the coarse grid.
* **Sparse tokens** (PWV stations and surface stations) are tiny in count
  (~176 and ~289) but spatially & temporally irregular; we tokenise with
  Fourier positional encoding on (lon, lat, [alti]) plus a per-time-step
  value embedding, and carry an attention mask for missing observations.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# small reusable bits
# ---------------------------------------------------------------------------


class _ConvBlock(nn.Module):
    """Conv2d + GroupNorm + GELU. Stride configurable."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=pad)
        groups = math.gcd(8, out_ch) or 1
        self.norm = nn.GroupNorm(groups, out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class _WindowAttention(nn.Module):
    """Non-shifted window self-attention (Swin-style but simpler).

    We don't bother with the shift trick — for a 4-block tower the absolute
    receptive field is large enough through the downsample stages, and the
    test shapes are small enough that windowing is mostly an OOM safeguard.
    """

    def __init__(self, dim: int, n_heads: int, window: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.window = window
        self.scale = (dim // n_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        ws = self.window
        # pad to multiple of window
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        # partition windows
        x = rearrange(x, "b (nh ws1) (nw ws2) c -> (b nh nw) (ws1 ws2) c",
                      ws1=ws, ws2=ws)
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B*, h, N, d)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        out = self.proj(out)
        out = rearrange(out, "(b nh nw) (ws1 ws2) c -> b (nh ws1) (nw ws2) c",
                        b=B, nh=Hp // ws, nw=Wp // ws, ws1=ws, ws2=ws)
        if pad_h or pad_w:
            out = out[:, :H, :W, :]
        return out


class _SpatialBlock(nn.Module):
    """Window-attn + MLP, pre-norm."""

    def __init__(self, dim: int, n_heads: int, window: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, n_heads, window)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _TemporalBlock(nn.Module):
    """Per-spatial-location attention across the time axis."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*H*W, T, C)
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


def fourier_features(coords: torch.Tensor, n_bands: int = 8,
                     max_freq: float = 64.0) -> torch.Tensor:
    """Sinusoidal positional encoding for (B, N, D) coord tensors.

    Returns (B, N, D * 2 * n_bands). ``coords`` should be roughly in [-1, 1].
    """
    freqs = torch.linspace(1.0, max_freq, n_bands, device=coords.device)
    # (B, N, D, n_bands)
    angles = coords.unsqueeze(-1) * freqs * math.pi
    feats = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return rearrange(feats, "b n d k -> b n (d k)")


# ---------------------------------------------------------------------------
# radar encoder
# ---------------------------------------------------------------------------


class RadarEncoder(nn.Module):
    """Encode (B, T, H, W) dBZ radar into (B, T, C, H/8, W/8) features.

    Stem: stride-4 patch embed → 1 spatial block at H/4 → stride-2 conv → 1
    spatial block at H/8. Temporal mixing happens AFTER spatial encoding in
    one shared temporal-attention block so we don't blow up memory.
    """

    def __init__(self, in_channels: int = 1, dim: int = 192,
                 n_heads: int = 8, window: int = 8,
                 n_spatial_blocks: int = 2, n_temporal_blocks: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            _ConvBlock(in_channels, dim // 2, stride=4, kernel=7),
            _ConvBlock(dim // 2, dim, stride=1),
        )
        self.spatial1 = nn.ModuleList(
            [_SpatialBlock(dim, n_heads, window) for _ in range(n_spatial_blocks)]
        )
        self.down = _ConvBlock(dim, dim, stride=2)
        self.spatial2 = nn.ModuleList(
            [_SpatialBlock(dim, n_heads, window) for _ in range(n_spatial_blocks)]
        )
        self.temporal = nn.ModuleList(
            [_TemporalBlock(dim, n_heads) for _ in range(n_temporal_blocks)]
        )
        self.dim = dim

    def forward(self, radar: torch.Tensor, return_skips: bool = False):
        # radar: (B, T, H, W)
        B, T, H, W = radar.shape
        x = rearrange(radar, "b t h w -> (b t) () h w")
        x = self.stem(x)                                    # (B*T, C, H/4, W/4)
        x = rearrange(x, "n c h w -> n h w c")
        for blk in self.spatial1:
            x = blk(x)
        # stage1 skip: post-spatial1 at H/4
        skip_stage1 = rearrange(x, "(b t) h w c -> b t c h w", b=B, t=T)
        x = rearrange(x, "n h w c -> n c h w")
        x = self.down(x)                                    # (B*T, C, H/8, W/8)
        x = rearrange(x, "n c h w -> n h w c")
        for blk in self.spatial2:
            x = blk(x)
        # stage2 skip: post-spatial2 at H/8 (before temporal mixing)
        h2, w2 = x.shape[1], x.shape[2]
        skip_stage2 = rearrange(x, "(b t) h w c -> b t c h w", b=B, t=T)
        # temporal mixing across T
        x = rearrange(x, "(b t) h w c -> (b h w) t c", b=B, t=T)
        for blk in self.temporal:
            x = blk(x)
        x = rearrange(x, "(b h w) t c -> b t c h w", b=B, h=h2, w=w2)
        if return_skips:
            return x, {"stage1": skip_stage1, "stage2": skip_stage2}
        return x


# ---------------------------------------------------------------------------
# ERA5 encoder
# ---------------------------------------------------------------------------


class Era5Encoder(nn.Module):
    """Encode the 4-variable, L-level ERA5 cube into a coarse spatial feature.

    Input is a mapping {var: (B, T, L, H, W)} produced by ``era5_io``; we
    stack along the channel dim, downsample to a coarse grid (target_size,
    default 32) to save FLOPs, then run a small conv tower.
    Optional ``mfd`` channel can be appended: (B, T, L_mfd, H, W).
    """

    def __init__(self, n_vars: int = 4, n_levels: int = 8, dim: int = 192,
                 mfd_channels: int = 0, target_size: int = 32):
        super().__init__()
        in_ch = n_vars * n_levels + mfd_channels
        self.target_size = target_size
        self.tower = nn.Sequential(
            _ConvBlock(in_ch, dim // 2, stride=1),
            _ConvBlock(dim // 2, dim, stride=2),
            _ConvBlock(dim, dim, stride=1),
        )
        self.dim = dim
        self.mfd_channels = mfd_channels

    def forward(self,
                era5: Mapping[str, torch.Tensor],
                mfd: torch.Tensor | None = None) -> torch.Tensor:
        # era5: dict of (B, T, L, H, W); stack -> (B, T, V*L, H, W)
        keys = sorted(era5.keys())
        stacked = torch.stack([era5[k] for k in keys], dim=2)   # (B, T, V, L, H, W)
        B, T, V, L, H, W = stacked.shape
        x = stacked.reshape(B, T, V * L, H, W)
        if mfd is not None and self.mfd_channels > 0:
            # mfd may be at a different spatial size; resize to match
            if mfd.shape[-2:] != (H, W):
                mfd = F.interpolate(
                    mfd.reshape(B * T, mfd.shape[2], *mfd.shape[-2:]),
                    size=(H, W), mode="bilinear", align_corners=False
                ).reshape(B, T, mfd.shape[2], H, W)
            x = torch.cat([x, mfd], dim=2)
        x = rearrange(x, "b t c h w -> (b t) c h w")
        # downsample to target coarse grid before the conv tower
        x = F.adaptive_avg_pool2d(x, self.target_size)
        x = self.tower(x)                                       # (B*T, C, S/2, S/2)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T)
        return x


# ---------------------------------------------------------------------------
# sparse PWV / station tokens
# ---------------------------------------------------------------------------


class SparseTokenEncoder(nn.Module):
    """Token-ise sparse observations with Fourier positional encoding.

    Two streams, both share the projection head:
      * PWV: (B, T_pwv, N_pwv) values + (N_pwv, 2) coords  (lon, lat)
      * Stations: (B, T_sta, N_sta, F) values + (N_sta, 3) coords  (lon, lat, alti)

    Output token bank has shape (B, T, N_tot, dim) where T is the radar
    frame count and we ``last-fill`` the lower-rate streams onto the radar
    clock — see DATA_HANDLING.md §7.
    """

    def __init__(self, dim: int = 192, pwv_n_features: int = 1,
                 station_n_features: int = 6, n_fourier_bands: int = 8):
        super().__init__()
        self.dim = dim
        self.n_fourier_bands = n_fourier_bands
        # PE dims: 2 coords -> 2*2*B floats; 3 coords -> 3*2*B
        pwv_pe = 2 * 2 * n_fourier_bands
        sta_pe = 3 * 2 * n_fourier_bands
        # value+mask channels: value, mask. Stations have F features + 1 mask
        self.pwv_proj = nn.Sequential(
            nn.Linear(pwv_n_features + 1 + pwv_pe, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.sta_proj = nn.Sequential(
            nn.Linear(station_n_features + 1 + sta_pe, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # learnable stream-type embedding (PWV vs station)
        self.type_embed = nn.Embedding(2, dim)

    @staticmethod
    def _normalise_coords(coords: torch.Tensor) -> torch.Tensor:
        """Map raw (lon, lat, [alti]) into a roughly [-1, 1] range.

        Lon/Lat use the radar bounding box (113-120°E, 36-42.6°N); altitude
        is divided by 4000 m as a rough max for the region (Taihang peaks).
        """
        c = coords.clone().float()
        # (B, N, D) or (N, D)
        if c.shape[-1] >= 1:
            c[..., 0] = (c[..., 0] - 116.5) / 3.5     # lon
        if c.shape[-1] >= 2:
            c[..., 1] = (c[..., 1] - 39.3) / 3.3      # lat
        if c.shape[-1] >= 3:
            c[..., 2] = c[..., 2] / 4000.0            # alti
        return c

    def _expand_time(self, x: torch.Tensor, T_target: int) -> torch.Tensor:
        """Repeat a (B, T_src, ...) tensor on the time axis to T_target via last-fill."""
        B, T_src = x.shape[:2]
        # we use last-fill: each radar frame inherits the most recent obs frame.
        # mapping: src_idx = min(floor(t * T_src / T_target), T_src-1)
        idx = torch.arange(T_target, device=x.device)
        idx = (idx.float() * T_src / T_target).floor().clamp_max(T_src - 1).long()
        return x.index_select(1, idx)

    def forward(self,
                pwv_vals: torch.Tensor, pwv_mask: torch.Tensor, pwv_coords: torch.Tensor,
                sta_vals: torch.Tensor, sta_mask: torch.Tensor, sta_coords: torch.Tensor,
                T_target: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (tokens, key_padding_mask).

        tokens: (B, T_target, N_pwv + N_sta, dim)
        key_padding_mask: (B, T_target, N_pwv + N_sta) — True = padding/missing.
        """
        # ---- PWV ----
        # pwv_vals: (B, T_pwv, N) -> add feature dim
        pv = pwv_vals.unsqueeze(-1).float()                       # (B, T_pwv, N, 1)
        pm = pwv_mask.unsqueeze(-1).float()                       # (B, T_pwv, N, 1)
        if pwv_coords.dim() == 2:
            pc = pwv_coords.unsqueeze(0).expand(pv.shape[0], -1, -1)
        else:
            pc = pwv_coords
        pc = self._normalise_coords(pc)                           # (B, N, 2)
        pwv_pe = fourier_features(pc, self.n_fourier_bands)       # (B, N, pe)
        pwv_pe = pwv_pe.unsqueeze(1).expand(-1, pv.shape[1], -1, -1)
        pwv_input = torch.cat([pv, pm, pwv_pe], dim=-1)
        pwv_tok = self.pwv_proj(pwv_input)                        # (B, T_pwv, N, C)
        pwv_tok = pwv_tok + self.type_embed.weight[0]

        # ---- Stations ----
        sv = sta_vals.float()                                     # (B, T_sta, N, F)
        sm = sta_mask.unsqueeze(-1).float()                       # (B, T_sta, N, 1)
        if sta_coords.dim() == 2:
            sc = sta_coords.unsqueeze(0).expand(sv.shape[0], -1, -1)
        else:
            sc = sta_coords
        sc = self._normalise_coords(sc)
        sta_pe = fourier_features(sc, self.n_fourier_bands)       # (B, N, pe)
        sta_pe = sta_pe.unsqueeze(1).expand(-1, sv.shape[1], -1, -1)
        sta_input = torch.cat([sv, sm, sta_pe], dim=-1)
        sta_tok = self.sta_proj(sta_input)                        # (B, T_sta, N, C)
        sta_tok = sta_tok + self.type_embed.weight[1]

        # ---- broadcast to radar clock ----
        pwv_tok_T = self._expand_time(pwv_tok, T_target)          # (B, T, N_pwv, C)
        sta_tok_T = self._expand_time(sta_tok, T_target)
        pwv_mask_T = self._expand_time(pwv_mask, T_target)
        sta_mask_T = self._expand_time(sta_mask, T_target)

        tokens = torch.cat([pwv_tok_T, sta_tok_T], dim=2)         # (B, T, N_tot, C)
        # key_padding_mask: True where the token is invalid (missing observation)
        kpm = torch.cat([pwv_mask_T, sta_mask_T], dim=2) <= 0.5
        return tokens, kpm
