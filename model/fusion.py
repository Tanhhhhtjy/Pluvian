"""Cross-attention fusion between dense (radar+ERA5) features and sparse tokens.

Dense features serve as queries — they ask the sparse network "what do you
know about my location/time?". Sparse tokens are the keys/values. We use a
per-time-step attention pattern (each radar frame attends only to its own
sparse-token slice) to keep memory linear in T.
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
        # Flatten dense to (B*T, H*W, C) for per-frame attention.
        q = rearrange(dense, "b t c h w -> (b t) (h w) c")
        k = rearrange(sparse, "b t n c -> (b t) n c")
        kpm = None
        if sparse_mask is not None:
            kpm = rearrange(sparse_mask, "b t n -> (b t) n")
            # If a row is ENTIRELY masked, nn.MultiheadAttention returns NaN.
            # Detect those rows and un-mask their first token as a safety valve.
            all_masked = kpm.all(dim=-1)
            if all_masked.any():
                kpm = kpm.clone()
                kpm[all_masked, 0] = False
        for layer in self.layers:
            q = layer(q, k, key_padding_mask=kpm)
        out = rearrange(q, "(b t) (h w) c -> b t c h w",
                        b=B, t=T, h=H, w=W)
        return out
