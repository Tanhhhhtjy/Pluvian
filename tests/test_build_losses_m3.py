"""Phase 7b audit M3: build_losses must fail loudly when the data-loss type
conflicts with the intensity-stratified model head.
"""
from __future__ import annotations
import sys
from pathlib import Path
from copy import deepcopy

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.train import build_losses, load_config


def _intensity_cfg():
    cfg = load_config(Path("configs/ablation_7b_intensity.yaml"))
    return deepcopy(cfg)


def test_intensity_on_with_weighted_mse_raises():
    cfg = _intensity_cfg()
    cfg["loss"]["data"]["type"] = "weighted_mse"
    with pytest.raises(ValueError, match="intensity_stratified"):
        build_losses(cfg)


def test_intensity_on_with_tweedie_raises():
    cfg = _intensity_cfg()
    cfg["loss"]["data"]["type"] = "tweedie"
    with pytest.raises(ValueError, match="intensity_stratified"):
        build_losses(cfg)


def test_intensity_off_with_intensity_loss_raises():
    cfg = _intensity_cfg()
    cfg["model"]["intensity_stratified"] = False
    cfg["loss"]["data"]["type"] = "intensity"
    with pytest.raises(ValueError, match="intensity_stratified=True"):
        build_losses(cfg)


def test_intensity_on_with_intensity_loss_ok():
    cfg = _intensity_cfg()
    cfg["loss"]["data"]["type"] = "intensity"
    losses = build_losses(cfg)
    assert losses["data"] is None  # _intensity_loss is invoked directly


def test_legacy_weighted_mse_still_ok():
    cfg = load_config(Path("configs/ablation_1_radar_only.yaml"))
    losses = build_losses(cfg)
    assert losses["data"] is not None
