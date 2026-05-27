"""Audit item #1 (2026-05-27): CkptManager.update() used to overwrite
``best.pt`` with the CURRENT epoch state, regardless of which epoch
actually held the best monitored score. ab2 reported best CSI@10mm = 0.625
at epoch 24 but its ``best.pt`` held the epoch-59 (CSI 0.619) weights.
This test simulates three epochs with scores [0.5, 0.7, 0.3] and asserts
that ``best.pt`` loads back the epoch-1 state (the 0.7 winner).
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.train import CkptManager


def test_best_pt_holds_highest_score_epoch(tmp_path):
    mgr = CkptManager(tmp_path, monitor="val/csi", mode="max", top_k=5)
    states = [{"epoch": 0, "tag": "A"},
              {"epoch": 1, "tag": "B"},
              {"epoch": 2, "tag": "C"}]
    scores = [0.5, 0.7, 0.3]
    for ep, (s, st) in enumerate(zip(scores, states)):
        mgr.update(st, score=s, epoch=ep)

    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["epoch"] == 1
    assert best["tag"] == "B"

    last = torch.load(tmp_path / "last.pt", weights_only=False)
    assert last["epoch"] == 2


def test_best_pt_min_mode(tmp_path):
    mgr = CkptManager(tmp_path, monitor="val/loss", mode="min", top_k=5)
    for ep, s in enumerate([1.0, 0.4, 0.8]):
        mgr.update({"epoch": ep}, score=s, epoch=ep)
    best = torch.load(tmp_path / "best.pt", weights_only=False)
    assert best["epoch"] == 1


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_best_pt_holds_highest_score_epoch(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_best_pt_min_mode(Path(d))
    print("OK")
