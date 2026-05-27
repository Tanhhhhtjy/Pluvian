"""Phase 7b audit M2: legacy ckpts must load non-strict; same-branch ckpts
must still load strict=True without warnings.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from model.pluvian import Pluvian


def _build_tiny_model(gated_fusion: bool = False) -> Pluvian:
    return Pluvian(
        radar_channels=1, pwv_enabled=True, pwv_concat_only=False,
        era5_enabled=False, mfd_channel_enabled=False,
        forecast_frames=4, input_frames=4,
        hidden_dim=32, n_layers=2, n_heads=4,
        gated_fusion=gated_fusion, intensity_stratified=False,
    )


def test_same_branch_ckpt_loads_strict():
    m1 = _build_tiny_model()
    m2 = _build_tiny_model()
    result = m2.load_state_dict(m1.state_dict(), strict=True)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def test_legacy_ckpt_missing_new_keys_loads_nonstrict():
    m = _build_tiny_model()
    state = m.state_dict()
    legacy = {k: v for k, v in state.items()
              if "lead_attn" not in k and "skip" not in k and "fuse_stage" not in k}
    assert len(legacy) < len(state)

    m2 = _build_tiny_model()
    result = m2.load_state_dict(legacy, strict=False)
    assert len(result.missing_keys) > 0
    assert result.unexpected_keys == []


def test_legacy_ckpt_unexpected_keys_loads_nonstrict():
    m = _build_tiny_model()
    state = m.state_dict()
    state["decoder.time_proj.weight"] = torch.zeros(1)
    result = _build_tiny_model().load_state_dict(state, strict=False)
    assert "decoder.time_proj.weight" in result.unexpected_keys
