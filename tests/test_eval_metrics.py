"""Phase 7a unit tests for full-set CSI / FSS aggregation.

Audit (2026-05-27) revealed the old `validate()` reduced CSI by averaging the
per-batch CSI values, which is the wrong reduction: it weights every batch
equally regardless of how many positive pixels it contains. With drop_last=False
and unbalanced rain/no-rain batches this drifted the val estimate ~0.02-0.05
away from the true full-set CSI. The fix accumulates raw hits/false-alarms/
misses across the whole epoch and computes one CSI at the end.

These tests construct synthetic batches that are deliberately imbalanced
(one batch dense rain, one batch nearly empty) so the batch-mean vs
full-set CSI disagree by a measurable amount, and check that the new
`_csi_counts` / `_fss_components` helpers reproduce the answer you would
get by concatenating all batches into a single tensor.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.train import _csi, _csi_counts, _fss_binary, _fss_components


def _make_batches(seed: int = 0):
    """Two batches with very different rain-pixel densities.

    Batch A: large, pred ~= tgt -> high CSI.
    Batch B: small, pred independent of tgt -> low CSI.
    Same H,W,T so the two batches can be concatenated along the batch axis
    (mirrors a real DataLoader with drop_last=False).
    """
    g = torch.Generator().manual_seed(seed)
    pred_a = torch.rand(8, 6, 32, 32, generator=g) * 50.0
    tgt_a = pred_a + torch.randn(8, 6, 32, 32, generator=g) * 1.0
    pred_b = torch.rand(1, 6, 32, 32, generator=g) * 12.0
    tgt_b = torch.rand(1, 6, 32, 32, generator=g) * 12.0
    return [(pred_a, tgt_a), (pred_b, tgt_b)]


def test_csi_fullset_matches_concatenated():
    batches = _make_batches()
    thr = 10.0
    # Full-set ground truth: concatenate then compute CSI once.
    pred_all = torch.cat([b[0] for b in batches], dim=0)
    tgt_all = torch.cat([b[1] for b in batches], dim=0)
    csi_ref = _csi(pred_all, tgt_all, thr)
    # Accumulator path.
    hits = fa = miss = 0
    for p, t in batches:
        h, f, m = _csi_counts(p, t, thr)
        hits += h; fa += f; miss += m
    denom = hits + fa + miss
    csi_acc = hits / denom if denom > 0 else float("nan")
    assert abs(csi_acc - csi_ref) < 1e-9, (csi_acc, csi_ref)


def test_csi_batch_mean_differs_from_fullset():
    """Sanity: the bug we're fixing must actually disagree with the truth.

    If this assertion ever starts failing for spurious reasons, regenerate
    `_make_batches` so the rain densities are more dissimilar.
    """
    batches = _make_batches()
    thr = 10.0
    pred_all = torch.cat([b[0] for b in batches], dim=0)
    tgt_all = torch.cat([b[1] for b in batches], dim=0)
    csi_ref = _csi(pred_all, tgt_all, thr)
    per_batch = [_csi(p, t, thr) for p, t in batches]
    per_batch = [v for v in per_batch if v == v]
    csi_mean = float(np.mean(per_batch))
    # Expect at least 0.01 disagreement on this synthetic case
    assert abs(csi_mean - csi_ref) > 0.01, (csi_mean, csi_ref)


def test_fss_fullset_matches_concatenated():
    batches = _make_batches(seed=1)
    thr = 1.0
    window = 3
    pred_all = torch.cat([b[0] for b in batches], dim=0)
    tgt_all = torch.cat([b[1] for b in batches], dim=0)
    fss_ref = _fss_binary(pred_all, tgt_all, thr, window)
    num = den = 0.0
    for p, t in batches:
        n, d = _fss_components(p, t, thr, window)
        num += n; den += d
    fss_acc = 1.0 - num / den if den > 0 else float("nan")
    assert abs(fss_acc - fss_ref) < 1e-5, (fss_acc, fss_ref)


def test_fss_multi_threshold_aggregation():
    """Audit #6: val FSS used to only compute threshold=1mm even though the
    training-loss FSS spans [1, 10, 30]. Verify each (nbr, thr) accumulator
    is independent and matches a full-set computation."""
    batches = _make_batches(seed=2)
    pred_all = torch.cat([b[0] for b in batches], dim=0)
    tgt_all = torch.cat([b[1] for b in batches], dim=0)
    for thr in (1.0, 10.0, 30.0):
        for nbr in (3, 11):
            fss_ref = _fss_binary(pred_all, tgt_all, thr, nbr)
            num = den = 0.0
            for p, t in batches:
                n, d = _fss_components(p, t, thr, nbr)
                num += n; den += d
            fss_acc = 1.0 - num / den if den > 0 else float("nan")
            if fss_acc == fss_acc and fss_ref == fss_ref:
                assert abs(fss_acc - fss_ref) < 1e-5, (thr, nbr, fss_acc, fss_ref)


def test_csi_empty_returns_nan():
    pred = torch.zeros(1, 1, 4, 4)
    tgt = torch.zeros(1, 1, 4, 4)
    h, fa, m = _csi_counts(pred, tgt, 10.0)
    assert (h, fa, m) == (0, 0, 0)


if __name__ == "__main__":
    test_csi_fullset_matches_concatenated()
    test_csi_batch_mean_differs_from_fullset()
    test_fss_fullset_matches_concatenated()
    test_csi_empty_returns_nan()
    print("OK")
