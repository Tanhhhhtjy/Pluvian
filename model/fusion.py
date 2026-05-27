"""Cross-attention fusion between dense (radar+ERA5) features and sparse tokens.

Two implementations are provided:

* ``CrossAttentionFusion`` — original symmetric design (PWV + station tokens
  share one bag). Kept for backward compatibility with the ``pwv_concat_only``
  ablation path.
* ``GatedAsymmFusion`` — Phase 7b T2 design. PWV and station tokens are
  routed through separate cross-attention branches, each modulated by a
  per-pixel sigmoid gate over the radar feature. PWV tokens get a small
  refinement tower (2 spatial self-attn + 1 temporal self-attn) before they
  reach the cross-attention block, addressing the audit finding that the PWV
  signal was diluted by the much larger station bag.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class _CrossAttnBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # q: (B, Nq, C); k: (B, Nk, C)
        a, _ = self.cross(self.norm_q(q), self.norm_k(k), self.norm_k(k),
                          key_padding_mask=key_padding_mask, need_weights=False)
        q = q + a
        q = q + self.mlp(self.norm2(q))
        return q


def _safe_kpm(kpm: torch.Tensor) -> torch.Tensor:
    """Un-mask the first token on rows that would otherwise be fully masked.

    nn.MultiheadAttention emits NaN for entirely-masked rows. The first token
    carries the learned type embedding so substituting it is benign.
    """
    all_masked = kpm.all(dim=-1)
    if all_masked.any():
        kpm = kpm.clone()
        kpm[all_masked, 0] = False
    return kpm


class CrossAttentionFusion(nn.Module):
    """Multi-layer cross-attention with per-frame sparse token slicing.

    Inputs
    ------
    dense:  (B, T, C, H, W)
    sparse: (B, T, N, C)
    sparse_mask: (B, T, N) — True where the token is invalid.
    """

    def __init__(self, dim: int = 192, n_heads: int = 8, n_layers: int = 2,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [_CrossAttnBlock(dim, n_heads, mlp_ratio) for _ in range(n_layers)]
        )
        self.dim = dim

    def forward(self,
                dense: torch.Tensor,
                sparse: torch.Tensor,
                sparse_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C, H, W = dense.shape
        q = rearrange(dense, "b t c h w -> (b t) (h w) c")
        k = rearrange(sparse, "b t n c -> (b t) n c")
        kpm = None
        if sparse_mask is not None:
            kpm = _safe_kpm(rearrange(sparse_mask, "b t n -> (b t) n"))
        for layer in self.layers:
            q = layer(q, k, key_padding_mask=kpm)
        out = rearrange(q, "(b t) (h w) c -> b t c h w",
                        b=B, t=T, h=H, w=W)
        return out


# ---------------------------------------------------------------------------
# Phase 7b T2: asymmetric gated fusion
# ---------------------------------------------------------------------------


class _SelfAttnBlock(nn.Module):
    """Pre-norm self-attention + MLP."""

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 2.0):
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

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class _PwvRefiner(nn.Module):
    """PWV refinement tower: 2 spatial self-attn + 1 temporal self-attn.

    Spatial attention mixes the N_pwv stations within each time frame;
    temporal attention mixes the T frames at each station. Gives the PWV
    stream a chance to integrate internally before the cross-attention path
    queries it.
    """

    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 2.0):
        super().__init__()
        self.spatial = nn.ModuleList(
            [_SelfAttnBlock(dim, n_heads, mlp_ratio) for _ in range(2)]
        )
        self.temporal = _SelfAttnBlock(dim, n_heads, mlp_ratio)

    def forward(self, tokens: torch.Tensor,
                kpm: torch.Tensor | None = None) -> torch.Tensor:
        # tokens: (B, T, N, C); kpm: (B, T, N)
        B, T, N, _ = tokens.shape
        x = rearrange(tokens, "b t n c -> (b t) n c")
        sp_kpm = None
        if kpm is not None:
            sp_kpm = _safe_kpm(rearrange(kpm, "b t n -> (b t) n"))
        for blk in self.spatial:
            x = blk(x, key_padding_mask=sp_kpm)
        x = rearrange(x, "(b t) n c -> (b n) t c", b=B, t=T)
        tp_kpm = None
        if kpm is not None:
            tp_kpm = _safe_kpm(rearrange(kpm, "b t n -> (b n) t"))
        x = self.temporal(x, key_padding_mask=tp_kpm)
        return rearrange(x, "(b n) t c -> b t n c", b=B, n=N)


class GatedAsymmFusion(nn.Module):
    """Asymmetric gated radar / PWV / station fusion (FusionCast-style).

    Each modality gets its own cross-attention branch with the radar dense
    feature as query, then a per-pixel sigmoid gate decides how much of that
    branch is added back onto the radar residual:

        fused = radar + gate_pwv * (pwv_branch - radar)
                      + gate_sta * (sta_branch - radar)

    Forward
    -------
    radar_feat: (B, T, C, H, W)
    pwv_tokens: (B, T, N_pwv, C),  pwv_kpm: (B, T, N_pwv) | None
    sta_tokens: (B, T, N_sta, C),  sta_kpm: (B, T, N_sta) | None
    """

    def __init__(self, dim: int = 192, n_heads: int = 8,
                 n_cross_layers: int = 1, mlp_ratio: float = 2.0,
                 gate_init_bias: float = -2.0):
        super().__init__()
        self.dim = dim
        self.pwv_refiner = _PwvRefiner(dim, n_heads, mlp_ratio)
        self.pwv_cross = nn.ModuleList(
            [_CrossAttnBlock(dim, n_heads, mlp_ratio) for _ in range(n_cross_layers)]
        )
        self.sta_cross = nn.ModuleList(
            [_CrossAttnBlock(dim, n_heads, mlp_ratio) for _ in range(n_cross_layers)]
        )
        # Per-pixel gates: Conv2d(C -> 1) + sigmoid. Weights zeroed and bias
        # negative so each gate starts at ~sigmoid(-2)≈0.12 — the model begins
        # close to a radar-only forecast and learns to open the gates.
        self.gate_pwv = nn.Conv2d(dim, 1, kernel_size=1)
        self.gate_sta = nn.Conv2d(dim, 1, kernel_size=1)
        nn.init.zeros_(self.gate_pwv.weight)
        nn.init.zeros_(self.gate_sta.weight)
        nn.init.constant_(self.gate_pwv.bias, gate_init_bias)
        nn.init.constant_(self.gate_sta.bias, gate_init_bias)

    def _branch(self, q_flat: torch.Tensor, tokens: torch.Tensor,
                kpm: torch.Tensor | None,
                blocks: nn.ModuleList) -> torch.Tensor:
        k = rearrange(tokens, "b t n c -> (b t) n c")
        pad = None
        if kpm is not None:
            pad = _safe_kpm(rearrange(kpm, "b t n -> (b t) n"))
        out = q_flat
        for blk in blocks:
            out = blk(out, k, key_padding_mask=pad)
        return out - q_flat

    def forward(self,
                radar_feat: torch.Tensor,
                pwv_tokens: torch.Tensor,
                pwv_kpm: torch.Tensor | None,
                sta_tokens: torch.Tensor,
                sta_kpm: torch.Tensor | None) -> torch.Tensor:
        B, T, C, H, W = radar_feat.shape
        q = rearrange(radar_feat, "b t c h w -> (b t) (h w) c")
        pwv_tokens = self.pwv_refiner(pwv_tokens, pwv_kpm)
        pwv_delta = self._branch(q, pwv_tokens, pwv_kpm, self.pwv_cross)
        sta_delta = self._branch(q, sta_tokens, sta_kpm, self.sta_cross)
        pwv_delta = rearrange(pwv_delta, "n (h w) c -> n c h w", h=H, w=W)
        sta_delta = rearrange(sta_delta, "n (h w) c -> n c h w", h=H, w=W)
        radar_flat = rearrange(radar_feat, "b t c h w -> (b t) c h w")
        g_pwv = torch.sigmoid(self.gate_pwv(radar_flat))
        g_sta = torch.sigmoid(self.gate_sta(radar_flat))
        fused = radar_flat + g_pwv * pwv_delta + g_sta * sta_delta
        return rearrange(fused, "(b t) c h w -> b t c h w", b=B, t=T)
