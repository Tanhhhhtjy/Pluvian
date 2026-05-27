"""Pluvian — main multi-source nowcasting model.

The model glues four pieces together:
  1. ``RadarEncoder``      — dense (B, T, H, W) -> (B, T, C, H/8, W/8)
  2. ``Era5Encoder``       — pressure-level cube  -> (B, T, C, S, S) at
                              coarse 0.25° grid; upsampled & added to radar
                              features at the fusion stage
  3. ``SparseTokenEncoder`` — GNSS-PWV + surface station tokens with
                              Fourier PE & validity masks
  4. ``CrossAttentionFusion`` — dense (radar+ERA5) queries sparse tokens
  5. ``PluvianDecoder``    — upsample 8x with two heads (rain, pwv)

Configuration knobs for ablations (see RESEARCH_PROPOSAL_v1.md §I-2/I-3):
  * ``pwv_enabled``         — turn off sparse PWV tokens entirely (I-3 ablation)
  * ``pwv_concat_only``     — treat PWV identically to other stations
                               (no separate cross-attn branch — I-2 ablation)
  * ``era5_enabled``        — turn off ERA5 input (radar-only baseline)
  * ``mfd_channel_enabled`` — add the derived MFD channel to the ERA5 branch
                               (I-2 main proposal)
"""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import RadarEncoder, Era5Encoder, SparseTokenEncoder
from .fusion import CrossAttentionFusion
from .decoder import PluvianDecoder


SIZE_PRESETS = {
    "small": dict(hidden_dim=256, n_layers=4, n_heads=8),
    "base":  dict(hidden_dim=384, n_layers=8, n_heads=8),
    "large": dict(hidden_dim=512, n_layers=12, n_heads=16),
}


class Pluvian(nn.Module):
    """Multi-source short-range precipitation forecaster.

    Forward signature accepts the dict produced by ``pipeline.NPJDataset``.
    See ``__init__`` for the ablation switches; defaults match the main
    "I-1+I-2+I-3" proposal model.
    """

    def __init__(
        self,
        radar_channels: int = 1,
        pwv_enabled: bool = True,
        pwv_concat_only: bool = False,
        era5_enabled: bool = True,
        mfd_channel_enabled: bool = False,
        forecast_frames: int = 18,
        input_frames: int = 30,
        hidden_dim: int = 192,
        n_layers: int = 4,
        n_heads: int = 8,
        n_era5_vars: int = 4,
        n_era5_levels: int = 8,
        mfd_channels: int = 3,
        decoder_dropout: float = 0.1,
        size: str | None = None,
    ):
        super().__init__()
        if size is not None:
            preset = SIZE_PRESETS[size]
            hidden_dim = preset["hidden_dim"]
            n_layers = preset["n_layers"]
            n_heads = preset["n_heads"]

        self.pwv_enabled = pwv_enabled
        self.pwv_concat_only = pwv_concat_only
        self.era5_enabled = era5_enabled
        self.mfd_channel_enabled = mfd_channel_enabled
        self.forecast_frames = forecast_frames
        self.input_frames = input_frames
        self.hidden_dim = hidden_dim

        # ---- encoders ----
        n_spatial_blocks = max(1, n_layers // 2)
        self.radar_encoder = RadarEncoder(
            in_channels=radar_channels, dim=hidden_dim, n_heads=n_heads,
            n_spatial_blocks=n_spatial_blocks, n_temporal_blocks=1,
        )
        if era5_enabled:
            self.era5_encoder = Era5Encoder(
                n_vars=n_era5_vars, n_levels=n_era5_levels, dim=hidden_dim,
                mfd_channels=mfd_channels if mfd_channel_enabled else 0,
                target_size=32,
            )
        else:
            self.era5_encoder = None

        # Sparse-token encoder is always built (the dataset still emits the
        # station stream). When ``pwv_enabled`` is False we just zero out
        # the PWV portion before encoding.
        self.sparse_encoder = SparseTokenEncoder(dim=hidden_dim)

        # ---- fusion ----
        self.fusion = CrossAttentionFusion(
            dim=hidden_dim, n_heads=n_heads,
            n_layers=max(1, n_layers // 2),
        )

        # ---- decoder ----
        self.decoder = PluvianDecoder(
            dim=hidden_dim, forecast_frames=forecast_frames,
            input_frames=input_frames, upsample_factor=8,
            dropout=decoder_dropout,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _fuse_era5_into_radar(self, radar_feat: torch.Tensor,
                              era5_feat: torch.Tensor) -> torch.Tensor:
        """Upsample ERA5 coarse features to radar grid and add residually."""
        B, T, C, H, W = radar_feat.shape
        x = era5_feat.reshape(B * T, C, *era5_feat.shape[-2:])
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        x = x.reshape(B, T, C, H, W)
        return radar_feat + x

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Predict rain & PWV from a NPJDataset batch.

        Returns
        -------
        dict with keys:
          * ``rain_pred`` : (B, T_out, H, W) — predicted reflectivity / rain
          * ``pwv_pred``  : (B, T_out, H, W) — predicted column water vapour
          * ``radar_feat``: (B, T_in, C, H/8, W/8) intermediate dense feature
                            (exposed for future physics-loss hooks)
        """
        fused = self._encode_and_fuse(batch)
        rain_pred, pwv_pred = self.decoder(fused)
        return {
            "rain_pred": rain_pred,
            "pwv_pred": pwv_pred,
            "radar_feat": fused,
        }

    def _encode_and_fuse(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Run radar + ERA5 + sparse-token encoders + fusion. Deterministic
        (no dropout). Returns the fused dense feature tensor that the decoder
        consumes. Factored out of ``forward`` so MC-dropout sampling can reuse
        the result across K decoder passes (audit #11: ~4x val speedup).
        """
        radar = batch["radar"].float()
        if radar.dim() == 4:
            # NPJDataset returns (B, T, H, W); add channel dim
            radar_in = radar
        else:
            raise ValueError(f"radar tensor has unexpected ndim={radar.dim()}")

        B, T_in, H, W = radar_in.shape

        # ---- radar branch ----
        radar_feat = self.radar_encoder(radar_in)            # (B, T, C, H/8, W/8)

        # ---- ERA5 branch ----
        if self.era5_enabled and self.era5_encoder is not None:
            era5 = batch["era5"]
            mfd = batch.get("era5_mfd") if self.mfd_channel_enabled else None
            era5_feat = self.era5_encoder(era5, mfd=mfd)
            radar_feat = self._fuse_era5_into_radar(radar_feat, era5_feat)

        # ---- sparse tokens ----
        pwv_vals = batch["pwv_grid"].float()
        pwv_mask = batch["pwv_mask"].float()
        pwv_coords = batch["pwv_coords"].float()
        sta_vals = batch["station_grid"].float()
        sta_mask = batch["station_mask"].float()
        sta_coords = batch["station_coords"].float()

        if not self.pwv_enabled:
            # zero the PWV channel entirely (and mask)
            pwv_vals = torch.zeros_like(pwv_vals)
            pwv_mask = torch.zeros_like(pwv_mask)
        elif self.pwv_concat_only:
            # ablation: feed PWV as just another station-like stream by
            # stripping its values to mean-only. This keeps the architecture
            # the same but disables the spatial-coverage signal that the
            # dedicated PWV stream encodes.
            pwv_vals = pwv_vals.mean(dim=-1, keepdim=True).expand_as(pwv_vals)

        sparse_tokens, sparse_kpm = self.sparse_encoder(
            pwv_vals, pwv_mask, pwv_coords,
            sta_vals, sta_mask, sta_coords,
            T_target=T_in,
        )

        # ---- fusion ----
        fused = self.fusion(radar_feat, sparse_tokens, sparse_kpm)
        return fused

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # MC-dropout ensemble for real CRPS
    # ------------------------------------------------------------------

    def mc_dropout_predict(self, batch: Mapping[str, torch.Tensor],
                           n_samples: int = 4) -> torch.Tensor:
        """Run ``n_samples`` stochastic forward passes with decoder dropout
        kept active. Returns a tensor of shape (K, B, T_out, H, W) of
        ``rain_pred`` samples. The rest of the network is run in eval mode;
        only the decoder Dropout is forced on.

        Audit #11 (2026-05-27): cache the deterministic encoder+fusion output
        and only re-run the decoder K times. Previously the encoder ran K
        times despite being deterministic, accounting for ~75% of MC val
        cost at K=4.
        """
        was_training = self.training
        self.eval()
        self.decoder.enable_mc_dropout(True)
        try:
            fused = self._encode_and_fuse(batch)
            samples = []
            for _ in range(n_samples):
                rain_pred, _ = self.decoder(fused)
                samples.append(rain_pred)
            return torch.stack(samples, dim=0)
        finally:
            self.decoder.enable_mc_dropout(False)
            if was_training:
                self.train()
